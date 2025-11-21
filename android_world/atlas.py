from typing import *
import re
from android_world.agents import infer
from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import json_action
from transformers import (
    AutoConfig,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLProcessor,
    AutoModelForCausalLM,
)
import torch
from qwen_vl_utils import process_vision_info
from android_world.env.adb_utils import _PATTERN_TO_ACTIVITY  

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("annotate.log", mode="a", encoding="utf-8"),  
    ],
)

logger = logging.getLogger(__name__)

ATLAS_ONLINE_SYS_PROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
       
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]
    
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 2: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 3: IMPOSSIBLE
    - purpose: Indicate the task is impossible.
    - format: IMPOSSIBLE
    - example usage: IMPOSSIBLE

Custom Action 4: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

Custom Action 5: OPENAPP
    - purpose: Open an app.
    - format: OPENAPP <APP_NAME>
    - example usage: OPENAPP Zoho Meeting

Custom Action 6: WAIT
    - purpose: Wait a set number of seconds for something on screen (e.g., a loading bar).
    - format: WAIT
    - example usage: WAIT

Custom Action 7: LONG_CLICK
    - purpose: Long click at the specified position.
    - format: LONG_CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_CLICK <point>[[101, 872]]</point>

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thought and Action.
Thought: Clearly outline your reasoning process for current step.
Action: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

And your final goal, previous actions and associated screenshot are as follows:

Final goal: {finalGoal}
Previous actions: {previousActions}
Screenshot: <image>

# Instructions for Determining the Next Action
- Carefully analyze the final goal, previous actions, and the current screenshot.
- Identify the most suitable action based on the context and the goal.
- Make sure the action you suggest aligns with the desired outcome, considering the previous steps.
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


def convert_atlas_action_to_json_action(
    action: str, width, height
) -> json_action.JSONAction:
    text = None
    direction = None
    goal_status = None
    app_name = None
    x, y = 0, 0

    if action.startswith("CLICK") or action.startswith("LONG_CLICK"):
        assert (
            width is not None and height is not None
        ), "Please provide screen size to convert relative coordinates to absolute coordinates"
        pattern = r"([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)"
        match = re.search(pattern, action)
        if match:
            ux, uy = float(match.group(1)), float(match.group(2))
            x, y = round(ux / 1000 * width), round(uy / 1000 * height)
        else:
            x, y = 0, 0
        action_type = (
            json_action.CLICK if action.startswith("CLICK") else json_action.LONG_PRESS
        )
    elif action.startswith("COMPLETE"):
        goal_status = "task_complete"
        action_type = json_action.STATUS
    elif action.startswith("PRESS_BACK"):
        action_type = json_action.NAVIGATE_BACK
    elif action.startswith("PRESS_HOME"):
        action_type = json_action.NAVIGATE_HOME
    elif action.startswith("SCROLL"):
        pattern = r"SCROLL ([a-zA-Z]+)"
        match = re.search(pattern, action)
        if match:
            origin_direction = match.group(1)
            if origin_direction == "UP":
                direction = "down"
            else:
                direction = "up"
        else:
            direction = "down"
        action_type = json_action.SCROLL
    elif action.startswith("OPENAPP"):
        pattern = r"OPENAPP ([^']+)"
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
    elif action.startswith("TYPE"):
        pattern = r"TYPE ([^']+)"
        match = re.search(pattern, action)
        if match:
            text = match.group(1)
        else:
            text = None
        action_type = json_action.INPUT_TEXT
    elif action.startswith("enter") or action.startswith("press_enter"):
        action_type = json_action.KEYBOARD_ENTER
    elif action.startswith("WAIT"):
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


def extractAtlasPrediction(text: str):
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
    return {"thought": thought, "action": action_str}


class Atlas(base_agent.EnvironmentInteractingAgent):
    def __init__(self, env: interface.AsyncEnv, name: str = "Atlas", modelPath=None):
        super().__init__(env, name)
        self._actions = []
        self._history = []

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

        self.model_config = AutoConfig.from_pretrained(self.modelPath)
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.modelPath,
                config=self.model_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            logger.info("load with flash attention")
        except Exception:
            logger.info("disable flash attention")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.modelPath,
                config=self.model_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
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
        imagePixel = image_array.copy()
        base64_image = infer.Gpt4Wrapper.encode_image(imagePixel)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": action_gen_prompt}]},
        ]

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ],
            }
        )

        return messages

    def generate(self, messages: List[Dict]):
        context = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

       
        inputs = self.processor(
            text=[context],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generatedIds = self.model.generate(
                **inputs,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
            ).to(self.device)
        trueGenIds = [
            genIds[len(inIds) :]
            for inIds, genIds in zip(inputs.input_ids, generatedIds)
        ]
        outputText = self.processor.batch_decode(
            trueGenIds, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return outputText[0]

    def updateHistory(self, image_array, atlas_action, atlas_thought):

        self._history.append(atlas_thought)

    def step(
        self, goal: str, verbose: bool = True
    ) -> base_agent.AgentInteractionResult:
        result = {
            "screenshot": None,
            "action_gen_payload": None,
            "action_gen_response": None,
            "atlas_action": None,
            "action": None,
            "thought": None,
        }
        state = self.get_post_transition_state()
        result["screenshot"] = state.pixels.copy()

        action_gen_prompt = ATLAS_ONLINE_SYS_PROMPT.format(
            finalGoal=goal, previousActions=self._history
        )

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
                    extra_text="\n~~~ANSWER~~~:" + action_gen_response,
                )
            )

        parsedDict = extractAtlasPrediction(action_gen_response)
        thought, atlas_action = parsedDict["thought"], parsedDict["action"]
        result["atlas_action"] = atlas_action
        result["thought"] = thought

        self.updateHistory(state.pixels.copy(), atlas_action, thought)

        try:
            action = convert_atlas_action_to_json_action(
                atlas_action, *self.env.logical_screen_size
            )
            result["action"] = action

        except ParseActionError as e:
            action = json_action.JSONAction(action_type=json_action.UNKNOWN)
            result["atlas_action"] = None
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
