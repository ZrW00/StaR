from typing import *
import re
from android_world.agents import infer
from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import json_action
from android_world.env.adb_utils import _PATTERN_TO_ACTIVITY
from transformers import AutoConfig, Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, Qwen2VLProcessor, AutoModelForCausalLM
import torch
from qwen_vl_utils import process_vision_info

from absl import logging # Betula added here
import time




UITARS_ONLINE_SYS_PROMPT = """
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')
long_press(start_box=\'<|box_start|>(x1,y1)<|box_end|>\', time=\'\')
type(content=\'\')
scroll(direction=\'down or up or right or left\')
open_app(app_name=\'\')
press_back()
press_home()
wait()
finished() # Submit the task regardless of whether it succeeds or fails.


## Note
- Use English in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""



def _match_app_correctly(agent_output: str) -> str:
    normalized_input = re.sub(r"[^a-zA-Z\s]", "", agent_output.lower())

    letter_sequences = re.findall(r"[a-zA-Z]{3,}", normalized_input)
    if not letter_sequences:
        return ""
    
    letter_sequences.sort(key=len, reverse=True)
    for pattern, _activity in _PATTERN_TO_ACTIVITY.items():
        aliases = pattern.split("|")
        for alias in aliases:
            clean_alias = re.sub(r"[^a-zA-Z]", "", alias.lower())
            for seq in letter_sequences:
                if len(seq) < 3:
                    continue
                if seq in clean_alias:
                    return pattern
    return ""

class ParseActionError(ValueError):
    pass

def convert_uitars_action_to_json_action(
    action: str,
    width, height
) -> json_action.JSONAction:
    """Converts a UITARS object to a JSONAction object.

    Args:
        action: The UITARS object to convert.

    Returns:
        The corresponding JSONAction object.

    Raises:
        ParseActionError: If cannot convert action.
    """
    text = None
    direction = None
    goal_status = None
    app_name = None
    x, y = 0, 0
  
    if action.startswith("click") or action.startswith("long_press"):
        assert width is not None and height is not None, "Please provide screen size to convert relative coordinates to absolute coordinates"
        pattern = r'\(([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)\)'
        match = re.search(pattern, action)
        if match:
            ux, uy = float(match.group(1)), float(match.group(2))
            x, y = round(ux / 1000 * width), round(uy / 1000 * height)
        action_type = json_action.CLICK if action.startswith("click") else json_action.LONG_PRESS
    elif action.startswith("finished"):
        goal_status = "task_complete"
        action_type = json_action.STATUS
    elif action.startswith("press_back"):
        action_type = json_action.NAVIGATE_BACK # Betula edited here, not PRESS_* but NAVIGATE_*
    elif action.startswith("press_home"):
        action_type = json_action.NAVIGATE_HOME # Betula edited here, not PRESS_* but NAVIGATE_*
    elif action.startswith("scroll"):
        # attention to the directions
        pattern = r"scroll\(direction=\\'([a-zA-Z]+)\\'\)"
        match = re.search(pattern, action)
        if match:
            origin_direction = match.group(1)
            if origin_direction == "up":
                direction = "down"
            else:
                direction = "up"
        else:
            direction = "down"
        action_type = json_action.SCROLL
    elif action.startswith("open_app"):
        pattern = r"open_app\(app_name='([^']+)'\)"
        match = re.search(pattern, action)
        if match:
            app_name = match.group(1)
            # Betula edited here: to match app correctly
            app_name = app_name.replace("The", "")
            app_name = app_name.replace("Simple", "")
            app_name = app_name.replace("Pro", "")
            app_name = app_name.replace("Google", "")
            app_name = _match_app_correctly(app_name)
        else:
            app_name = ""
        action_type = json_action.OPEN_APP  
    elif action.startswith("type"):
        pattern = r"type\(content='([^']+)'\)"
        match = re.search(pattern, action)
        if match:
            text = match.group(1)
        else:
            text = None
        action_type = json_action.INPUT_TEXT
    elif action.startswith("enter") or action.startswith("press_enter"):
        action_type = json_action.KEYBOARD_ENTER
    elif action.startswith("wait"):
        action_type = json_action.WAIT
    elif action.startswith("drag") or action.startswith("noneaction"):
        action_type = json_action.UNKNOWN
    else:
        raise ParseActionError(f"action: {action} does not have correpsonding translation target")
    return json_action.JSONAction(
        action_type=action_type,
        x=x,
        y=y,
        text=text,
        direction=direction,
        goal_status=goal_status,
        app_name=app_name,
    )
  
  
def extractUITARSPrediction(text:str):
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action: |$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action: |$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
            
    assert "Action:" in text, "no action results!"
    action_str = text.split("Action: ")[-1]
    return {"thought":thought, "action":action_str}



class UITARS(base_agent.EnvironmentInteractingAgent):
    def __init__(self, env: interface.AsyncEnv, name: str = "UITARS", modelPath=None):
        super().__init__(env, name)
        self._actions = []
        self._history = []
        
        self.additional_guidelines = None
        self.modelPath = modelPath
        self.cache_dir = '~/.cache'

        self.model_config = AutoConfig.from_pretrained(self.modelPath)

        self.tokenizer = AutoTokenizer.from_pretrained(self.modelPath, trust_remote_code=True, cache_dir=self.cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_new_tokens = 512
        
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.modelPath, 
                config=self.model_config, 
                device_map="auto", 
                trust_remote_code=True, 
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
            logging.info("load with flash attention")
        except Exception:
            logging.info("disable flash attention")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.modelPath, config=self.model_config, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
        
        self.processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(self.modelPath)
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = param.data.clone().contiguous()
                
        self.model = self.model.eval()
        
    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.env.hide_automation_ui()
        self._actions.clear()
        self._history.clear()
        
    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines
        
    def create_action_generation_messages_payload(self, action_gen_prompt, image_array):
        base64_image = infer.Gpt4Wrapper.encode_image(image_array)
        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "You are a helpful assistant."
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": action_gen_prompt
                }]
            }
        ]
        
        messages.extend(self._history[-8:])
        messages.append({
            "role": "user",
            "content": 
                [{
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }]
        })
        
        return messages
    
    def generate(self, messages:List[Dict]):
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
        return outputText[0]
    
    def updateHistory(self, image_array, uitars_action, uitars_thought):
        base64_image = infer.Gpt4Wrapper.encode_image(image_array)
        self._history.extend([
            {
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Thought: {uitars_thought}\nAction: {uitars_action}"
                    }
                ]
            }
        ]) 
    
    def step(
      self, goal: str, verbose: bool = True
    ) -> base_agent.AgentInteractionResult: 
        result = {
            "screenshot": None,
            "action_gen_payload": None,
            "action_gen_response": None,
            "uitars_action": None,
            "action": None,
            "thought": None,
        }
        state = self.get_post_transition_state()
        result["screenshot"] = state.pixels.copy()
        
        action_gen_prompt = UITARS_ONLINE_SYS_PROMPT.format(instruction=goal)
        
        payload = self.create_action_generation_messages_payload(action_gen_prompt, state.pixels.copy())
        
        result["action_gen_payload"] = payload
        
        t0 = time.time()
        action_gen_response = self.generate(payload)
        print(">>> generation use: ", time.time() - t0)
        
        result["action_gen_response"] = action_gen_response
        
        if verbose:
            (
                seeact_utils.display_prompt(
                    result["action_gen_payload"],
                    extra_text="\n~~~ANSWER~~~:" + action_gen_response,
                )
            )
        
        parsedDict = extractUITARSPrediction(action_gen_response)
        thought, uitars_action = parsedDict["thought"], parsedDict["action"]
        result["uitars_action"] = uitars_action
        result["thought"] = thought
        
        self.updateHistory(state.pixels.copy(), uitars_action, thought)
        
        
        try:
            action = convert_uitars_action_to_json_action(uitars_action, *self.env.logical_screen_size)
            result["action"] = action
        except ParseActionError as e:
            action = json_action.JSONAction(action_type=json_action.UNKNOWN)
            result["uitars_action"] = None
            result["action"] = action
        else:
            actuation.execute_adb_action(
                action,
                [],
                self.env.logical_screen_size,
                self.env.controller,
            ) 
        return base_agent.AgentInteractionResult(
            done=action.action_type == json_action.STATUS,
            data=result
        )
