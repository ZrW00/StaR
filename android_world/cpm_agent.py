from typing import *
import re
from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import json_action
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from android_world.env.adb_utils import _PATTERN_TO_ACTIVITY

import numpy as np
from PIL import Image
import json


CPM_ONLINE_SYS_PROMPT = """
# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。
# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。
# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束
# Schema
{
  "type": "object",
  "description": "执行操作并决定当前任务状态",
  "additionalProperties": false,
  "properties": {
    "thought": {
      "type": "string",
      "description": "智能体的思维过程"
    },
    "POINT": {
      "$ref": "#/$defs/Location",
      "description": "点击屏幕上的指定位置"
    },
    "to": {
      "description": "移动，组合手势参数",
      "oneOf": [
        {
          "enum": [
            "up",
            "down",
            "left",
            "right"
          ],
          "description": "从当前点（POINT）出发，执行滑动手势操作，方向包括向上、向下、向左、向右"
        },
        {
          "$ref": "#/$defs/Location",
          "description": "移动到某个位置"
        }
      ]
    },
    "duration": {
      "type": "integer",
      "description": "动作执行的时间或等待时间，毫秒",
      "minimum": 0,
      "default": 200
    },
    "PRESS": {
      "type": "string",
      "description": "触发特殊按键，HOME为回到主页按钮，BACK为返回按钮，ENTER为回车按钮",
      "enum": [
        "HOME",
        "BACK",
        "ENTER"
      ]
    },
    "TYPE": {
      "type": "string",
      "description": "输入文本"
    },
    "STATUS": {
      "type": "string",
      "description": "当前任务的状态。特殊情况：satisfied，无需操作；impossible，任务无法完成；interrupt，任务中断；need_feedback，需要用户反馈；",
      "enum": [
        "continue",
        "finish",
        "satisfied",
        "impossible",
        "interrupt",
        "need_feedback"
      ],
      "default": "continue"
    }
  },
  "$defs": {
    "Location": {
      "type": "array",
      "description": "坐标为相对于屏幕左上角位原点的相对位置，并且按照宽高比例缩放到0～1000，数组第一个元素为横坐标x，第二个元素为纵坐标y",
      "items": {
        "type": "integer",
        "minimum": 0,
        "maximum": 1000
      },
      "minItems": 2,
      "maxItems": 2
    }
  }
}
"""


class ParseActionError(ValueError):
    pass


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


def convert_uitars_action_to_json_action(
    action: str, width, height
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
        assert (
            width is not None and height is not None
        ), "Please provide screen size to convert relative coordinates to absolute coordinates"
        pattern = r"\(([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)\)"
        match = re.search(pattern, action)
        if match:
            ux, uy = float(match.group(1)), float(match.group(2))
            x, y = round(ux / 1000 * width), round(uy / 1000 * height)
        else:
            x, y = 0, 0
        action_type = (
            json_action.CLICK if action.startswith("click") else json_action.LONG_PRESS
        )
    elif action.startswith("finished"):
        goal_status = "task_complete"
        action_type = json_action.STATUS
    elif action.startswith("press_back"):
        action_type = (
            json_action.NAVIGATE_BACK
        )  
    elif action.startswith("press_home"):
        action_type = (
            json_action.NAVIGATE_HOME
        )  
    elif action.startswith("scroll"):
        
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
        raise ParseActionError(
            f"action: {action} does not have correpsonding translation target"
        )
    return json_action.JSONAction(
        action_type=action_type,
        x=x,
        y=y,
        text=text,
        direction=direction,
        goal_status=goal_status,
        app_name=app_name,
    )


def translateACG2Atlas(action: Union[Dict, List[Dict]]) -> str:
    if isinstance(action, dict):
        act = action
    elif isinstance(action, list):
        if len(action) == 0:
            return "NONEACTTION"
        act = action[0]
    else:
        return "NONEACTTION"

    atlas_action = None

    if "POINT" in act and "to" not in act and "TYPE" not in act and "PRESS" not in act:
        if "duration" in act:
            atlas_action = (
                f"LONG_CLICK <point>[[{act['POINT'][0]}, {act['POINT'][1]}]]</point>"
            )
        else:
            atlas_action = (
                f"CLICK <point>[[{act['POINT'][0]}, {act['POINT'][1]}]]</point>"
            )

    elif "to" in act:
        direction = act["to"].upper()
        atlas_action = f"SCROLL [{direction}]"

    elif "TYPE" in act:
        content = act["TYPE"]
        atlas_action = f"TYPE [{content}]"

    elif "PRESS" in act:
        key = act["PRESS"].upper()
        if key == "HOME":
            atlas_action = "PRESS_HOME"
        elif key == "BACK":
            atlas_action = "PRESS_BACK"
        elif key == "ENTER":
            atlas_action = "PRESS_ENTER"
        else:
            atlas_action = f"PRESS_{key}"

    elif "STATUS" in act:
        if act["STATUS"] == "finish":
            atlas_action = "COMPLETE"

    elif "duration" in act and "POINT" not in act and "TYPE" not in act:
        atlas_action = "WAIT"

    else:
        atlas_action = "NONEACTTION"

    return atlas_action


def translateAtlas2Uitars(action: str):
    if action.startswith("CLICK"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        uitarsAction = f"click(start_box='<|box_start|>({x},{y})<|box_end|>')"
    elif action.startswith("COMPLETE"):
        uitarsAction = "finished()"
    elif action.startswith("LONG_CLICK"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        atlasAction = f"LONG_CLICK <point>[[{x}, {y}]]</point>"
        uitarsAction = (
            f"long_press(start_box='<|box_start|>({x},{y})<|box_end|>', time='')"
        )
    elif action.startswith("PRESS_BACK"):
        uitarsAction = "press_back()"
    elif action.startswith("PRESS_HOME"):
        uitarsAction = "press_home()"
    elif action.startswith("SCROLL"):
        pattern = r"SCROLL\s+\[([A-Z]+)\]"
        match = re.search(pattern, action)
        if match:
            direction = match.group(1).lower()
        else:
            direction = "down"
        atlasAction = f"SCROLL [{direction}]"
        uitarsAction = f"scroll(direction={direction})"
    elif action.startswith("OPENAPP"):
        appName = action.replace("OPENAPP", "").strip()
        atlasAction = f"OPENAPP {appName}"
        uitarsAction = f"open_app(app_name='{appName}')"
    elif action.startswith("WAIT"):
        atlasAction = "WAIT"
        uitarsAction = "wait()"
    elif action.startswith("TYPE"):
        pattern = r"TYPE \[(.*?)\]"
        match = re.search(pattern, action)
        if match:
            content = match.group(1)
        else:
            content = None
        atlasAction = f"TYPE [{content}]"
        uitarsAction = f"type(content='{content}')"
    elif action.startswith("ENTER"):
        uitarsAction = "enter()"
    else:
        raise NotImplementedError(
            f"action: {action} does not have correpsonding translation target"
        )

    return uitarsAction


class CPM(base_agent.EnvironmentInteractingAgent):
    def __init__(self, env: interface.AsyncEnv, name: str = "CPM", modelPath=None):
        super().__init__(env, name)
        self._actions = []
        
        self.additional_guidelines = None
        self.modelPath = modelPath
        self.cache_dir = "~/.cache"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.modelPath, trust_remote_code=True, cache_dir=self.cache_dir
        )
        self.tokenizer.truncation_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_new_tokens = 512

        self.model_config = AutoConfig.from_pretrained(
            self.modelPath, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.modelPath,
                config=self.model_config,  # Betula edited here
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.modelPath,
                config=self.model_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = param.data.clone().contiguous()

        self.model = self.model.eval()

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.env.hide_automation_ui()
        self._actions.clear()

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def create_action_generation_messages_payload(self, action_gen_prompt, image_array):

        if isinstance(image_array, np.ndarray):
            if image_array.ndim == 3 and image_array.shape[-1] == 3:
                pil_img = Image.fromarray(image_array.astype("uint8"), mode="RGB")
            else:
                raise ValueError("image_array must be HxWx3 uint8 RGB numpy array.")
        elif isinstance(image_array, Image.Image):
            pil_img = image_array
        else:
            raise TypeError(f"Unsupported image type: {type(image_array)}")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": action_gen_prompt}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": pil_img}]},
        ]
        return messages

    def generate(self, messages: List[Dict]):

        instruction = messages[1]["content"]

        img_block = messages[2]["content"][0]
        if "image" not in img_block:
            image: Image.Image = img_block["image_url"]
        else:
            image: Image.Image = img_block["image"]
        if not isinstance(image, Image.Image):
            raise TypeError("The 'image' in messages must be a PIL.Image object.")

        def _resize_image(origin_img: Image.Image) -> Image.Image:
            w, h = origin_img.size
            max_line = 1120
            if h > max_line:
                w = int(w * max_line / h)
                h = max_line
            if w > max_line:
                h = int(h * max_line / w)
                w = max_line
            return origin_img.resize((w, h), resample=Image.Resampling.LANCZOS)

        image = _resize_image(image)

        msg = [
            {
                "role": "user",
                "content": [
                    f"<Question>{instruction}</Question>\n当前屏幕截图：",
                    image,
                ],
            }
        ]

        outputs = self.model.chat(
            image=None,
            msgs=msg,
            system_prompt=CPM_ONLINE_SYS_PROMPT,
            tokenizer=self.tokenizer,
            temperature=0.1,
            top_p=0.3,
            n=1,
        )

        if isinstance(outputs, list):
            outputs = outputs[0]

        if isinstance(outputs, str):
            try:
                outputs = json.loads(outputs)
                return outputs
            except Exception:
                return outputs

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

        action_gen_prompt = goal

        payload = self.create_action_generation_messages_payload(
            action_gen_prompt, state.pixels.copy()
        )

        result["action_gen_payload"] = payload

        action_gen_response = self.generate(payload)

        result["action_gen_response"] = action_gen_response

        if verbose:
            (
                seeact_utils.display_prompt(
                    result["action_gen_payload"],
                    extra_text="\n~~~ANSWER~~~:" + str(action_gen_response),
                )
            )

        uitars_action = translateAtlas2Uitars(translateACG2Atlas(action_gen_response))
        result["uitars_action"] = uitars_action

        try:
            action = convert_uitars_action_to_json_action(
                uitars_action, *self.env.logical_screen_size
            )
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
            done=action.action_type == json_action.STATUS, data=result
        )
