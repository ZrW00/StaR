import torch
from transformers import AutoConfig, Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, Qwen2VLProcessor, AutoModelForCausalLM

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from zhipuai import ZhipuAI
import base64
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import *

import logging
import json
import os
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("annotate.log", mode='a', encoding='utf-8')  # 输出到文件
    ]
)

logger = logging.getLogger(__name__)

class Agent:
    def __init__(
        self, 
        path:str, cache_dir='~/.cache', 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=128,
        benchmarkSetting="high"
    ):
        self.modelPath = path
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        assert benchmarkSetting in ["high", "low"], "benchmarkSetting must be either 'high' or 'low'"
        self.benchmarkSetting = benchmarkSetting
        
    
    def generate(self, messages:List[Dict[str, str]]):
        raise NotImplementedError()
    
    def input_lenth(self, sample:Dict[str, str]):
        text = sample["messages"][0]["content"]
        image = sample["images"][0]
        messages = [{
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "text":text,
                },
                {
                    "type":"image",
                    "image":image
                }
            ]
        }]
        context = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[context],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        input_length = len(inputs.input_ids[0])
        
        return input_length
        
        

class AtlasAgent(Agent):
    def __init__(
        self, 
        path:str, cache_dir='~/.cache', 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=512,
        benchmarkSetting="high"
    ):
        super().__init__(path=path, cache_dir=cache_dir, device=device, max_new_tokens=max_new_tokens, benchmarkSetting=benchmarkSetting)
        self.model_config = AutoConfig.from_pretrained(path)
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                path, 
                config=self.model_config, 
                device_map="auto", 
                trust_remote_code=True, 
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
            logger.info("load with flash attention")
        except Exception:
            logger.info("disable flash attention")
            print("disable flash attention")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(path, config=self.model_config, device_map="auto", trust_remote_code=True)
        self.processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(path)
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = param.data.clone().contiguous()
                
        
        
    def generate(self, messages:List[Dict[str, str]]):
        # messages = sample["messages"]
        context = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[context],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            generatedIds = self.model.generate(
                **inputs, 
                pad_token_id=self.processor.tokenizer.pad_token_id, 
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                
            ).to(self.device)
        trueGenIds = [
            genIds[len(inIds):] for inIds, genIds in zip(inputs.input_ids, generatedIds)
        ]
        outputText = self.processor.batch_decode(trueGenIds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputText


class UITARSAgent(Agent):
    def __init__(
        self, 
        path:str, cache_dir='~/.cache', 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=512,
        benchmarkSetting = "high"
    ):
        super().__init__(path=path, cache_dir=cache_dir, device=device, max_new_tokens=max_new_tokens, benchmarkSetting=benchmarkSetting)
        self.model_config = AutoConfig.from_pretrained(path)
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                path, 
                config=self.model_config, 
                device_map="auto", 
                trust_remote_code=True, 
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
            logger.info("load with flash attention")
        except Exception:
            logger.info("disable flash attention")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(path, config=self.model_config, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(path)
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = param.data.clone().contiguous()
                
        self.model = self.model.eval()
        
        
    def generate(self, messages:List[Dict[str, str]]):
        # messages = sample["messages"]
        context = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.benchmarkSetting == "low":
            context = context.rsplit("<|im_end|>", 1)[0].strip() + "\nAction:"
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[context],
            images=[image_inputs],
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generatedIds = self.model.generate(
                **inputs, 
                pad_token_id=self.processor.tokenizer.pad_token_id, 
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                use_cache=True
            ).to(self.device)
        trueGenIds = [
            genIds[len(inIds):] for inIds, genIds in zip(inputs.input_ids, generatedIds)
        ]
        outputText = self.processor.batch_decode(trueGenIds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if self.benchmarkSetting == "low":
            action = (
                outputText[0].strip().split("\n")[-1] 
                if "Action:" not in outputText[0] 
                else outputText[0].split("Action:")[-1].strip()
            )
            outputText = [f"Action: {action}"]
        return outputText


class GLM4VFlashAgent:
    def __init__(
        self,
        path:str, cache_dir='~/.cache', 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=512,
        benchmarkSetting = "high"
    ):
        apiKey = "30adcfd54712c617f290fe9beb513029.RRktJf0dcgvk53GO"
        self.model = ZhipuAI(api_key=apiKey)
        if path is None:
            path = "glm-4v-flash"
        self.modelType = path
        self.max_new_tokens = max_new_tokens
        
    def generate(self, sample:Dict[str, str]):
        text = sample["messages"][0]["content"]
        image = sample["images"][0]
        with open(image, "rb") as f:
            imgBase = base64.b64encode(f.read()).decode('utf-8')
        messages = [{
            "role":"user",
            "content":[
                {
                    "type":"image_url",
                    "image_url": {
                        "url": imgBase
                    }
                },
                {
                    "type":"text",
                    "text":text,
                }
            ]
        }]
        outputs = self.model.chat.completions.create(
            model=self.modelType,
            messages=messages,
            max_tokens=self.max_new_tokens
        ).choices[0].message.content
        outputText = [outputs]
        return outputText
  
class AgentCPMGUIAgent(Agent):
    def __init__(
        self, 
        path: str, 
        cache_dir='~/.cache',
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_new_tokens: int = 512,
        benchmarkSetting = "high",
        apiKey=None,
        **kwargs
    ):
        super().__init__(path=path, cache_dir=cache_dir, device=device, max_new_tokens=max_new_tokens, benchmarkSetting=benchmarkSetting, **kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        self.model.to(device)
        self.model.eval()

        schema_path = "/data1/models/OpenBMB/AgentCPM-GUI/schema.json"
        with open(schema_path, encoding="utf-8") as f:
            action_schema = json.load(f)
        items = list(action_schema.items())
        items.insert(3, ("required", ["thought"]))
        self.schema = dict(items)

        self.system_prompt_template = f"""# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。
# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。
# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束
# Schema
{json.dumps(self.schema, ensure_ascii=False, separators=(',', ':'))}"""

    def _resize_image(self, origin_img: Image.Image) -> Image.Image:
        """固定长边 1120 像素"""
        w, h = origin_img.size
        max_line = 1120
        if h > max_line:
            w = int(w * max_line / h)
            h = max_line
        if w > max_line:
            h = int(h * max_line / w)
            w = max_line
        return origin_img.resize((w, h), resample=Image.Resampling.LANCZOS)
    
    def _load_image_if_needed(self, img):
        """确保图像是 PIL.Image 对象"""
        if isinstance(img, str):
            if not os.path.exists(img):
                raise FileNotFoundError(f"Image path not found: {img}")
            img = Image.open(img).convert("RGB")
        return img

    def generate(self, messages):
        # instruction = messages["messages"][1]["content"]
        instruction = messages[1]["content"] # aitz等
        # instruction = messages[0]["content"] # state_cot
        # image = messages["images"][0] # PIL.Image 格式
        image = messages[2]["content"][0]["image"] # PIL.Image 格式 aitz等
        # image = messages[1]["content"][0]["image"] # PIL.Image 格式 state_cot等
        image = self._load_image_if_needed(image)
        image = self._resize_image(image)

        
        message = [{
            "role": "user",
            "content": [
                f"<Question>{instruction}</Question>\n当前屏幕截图：",
                image
            ]
        }]
        # 模型 chat 接口
        outputs = self.model.chat(
            image=None,
            msgs=message,
            system_prompt=self.system_prompt_template,
            tokenizer=self.tokenizer,
            temperature=0.1,
            top_p=0.3,
            n=1,
        )
        if isinstance(outputs, list):
            outputs = outputs[0]  # 取第一个字符串
        if isinstance(outputs, str):
            outputs = json.loads(outputs)  # 转换成 dict
        return outputs
    

class APIAgentForState:
    def __init__(
        self, 
        path:str=None, cache_dir='~/.cache', 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=128,
        benchmarkSetting="high"
    ):
        self.modelPath = path
        self.apiKey = "sk-lk2OlVIyEsuaJi0VvPw0mDf8bQt9LGy9siillORONrKnN9Od"
        self.apiUrl = "https://xinyun.ai/v1/chat/completions"
        
        assert benchmarkSetting in ["high", "low"], "benchmarkSetting must be either 'high' or 'low'"
        self.benchmarkSetting = benchmarkSetting
    
    def generate(self, messages:List[Dict[str, str]]):
        assert len(messages) == 2, "messages must contain 2 elements, first is text prompt, second is the screenshot"
        text = messages[0]["content"]
        imagePath = messages[1]['content'][0]["image"]
        
        with open(imagePath, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        input_messages = [
            {
                "role":"user",
                "content":text
            },
            {
                "role":"user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        data = {
            "model": self.modelPath,
            "messages": input_messages,
        }
        
        headers = {
            "Authorization": f"Bearer {self.apiKey}",
            "Content-Type": "application/json"
        }
    
        try:
        # 发送 POST 请求
            response = requests.post(self.apiUrl, headers=headers, json=data)
            
            if response.status_code == 200:
                # 解析响应并提取 Gemini 的回复
                result = response.json()
                return [result["choices"][0]["message"]["content"].strip()]
            else:
                return [f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}"]
        except Exception as e:
            return [f"请求出错: {str(e)}"]

class GUIR1Agent(Agent):
    def __init__(
        self, 
        path:str, cache_dir='~/.cache', 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=128,
        benchmarkSetting="high",
        **kwargs
    ):
        super().__init__(path=path, cache_dir=cache_dir, device=device, max_new_tokens=max_new_tokens, benchmarkSetting=benchmarkSetting, **kwargs)
        self.model_config = AutoConfig.from_pretrained(path)
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, 
                config=self.model_config, 
                device_map="auto", 
                trust_remote_code=True, 
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
        except Exception:
            print("disable flash attention")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, config=self.model_config, device_map="auto", trust_remote_code=True)
        self.processor: Qwen2_5_VLProcessor = AutoProcessor.from_pretrained(path)
        
    def generate(self, messages):
        context = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[context],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        
        with torch.no_grad():
            generatedIds = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens
            ).to(self.device)
        trueGenIds = [
            genIds[len(inIds):] for inIds, genIds in zip(inputs.input_ids, generatedIds)
        ]
        outputText = self.processor.batch_decode(trueGenIds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
        
        return outputText