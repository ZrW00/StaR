from .templates import *
from .actions import *
import os, json, json5, re, base64, time
import argparse
import zhipuai, base64
from zhipuai import ZhipuAI
import copy
from tqdm import tqdm
import math, traceback
from PIL import Image
from pathlib import Path
import multiprocessing as mp
from typing import *
# apiKey = "30adcfd54712c617f290fe9beb513029.RRktJf0dcgvk53GO"

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler("annotate.log", mode='a', encoding='utf-8')  
    ]
)



logger = logging.getLogger(__name__)

SAMPLET = {
    "messages": [
        {
            "role": "user",
            "content": None
        },
        {
            "role": "assistant",
            "content": None
        }
    ],
    "images": [],
    "layouts": [],
    "episodeID":None,
    "stepID":None,
    "width":None,
    "height":None
}

MESSAGET = [
    {
        "role": "user",
        "content": ""
    },
    {
        "role": "assistant",
        "content": None
    }
]

sys_prompt = {
    "role": "system",
    "content": "You are a helpful assistant."
}

def __in_bbox(x, y, bbox):
    return bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]

def __get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

def convertAndroidControlAction(action:dict, layoutPath:str, width:int, height:int) -> str:
    actionType = action["action_type"]
    with open(layoutPath, "r", encoding="utf-8") as f:
        layouts = [json.loads(line.strip()) for line in f]

    
    if actionType == "click":
        x, y = action["x"], action["y"]
        availableBboxes = []
        for layout in layouts:
            if layout["is_clickable"] and "bbox_pixels" in layout and __in_bbox(x, y, layout["bbox_pixels"]):
                availableBboxes.append(layout["bbox_pixels"])
        smallestBbox = None if availableBboxes == [] else \
            min(
                availableBboxes,
                key=lambda b: (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])
            )
        if smallestBbox is not None:
            center_x, center_y = __get_bbox_center(smallestBbox)
            if ((center_x - x) / width * 1000) ** 2 + ((center_y - y) / height * 1000) ** 2 <= 40:
                x, y = center_x, center_y
        ux, uy = (x / width) * 1000, (y / height) * 1000 
        result = f"CLICK <point>[[{ux:.1f}, {uy:.1f}]]</point>"
    elif actionType == "open_app":
        result = f"OPENAPP {action['app_name']}"
    elif actionType == "input_text":
        result = f"TYPE [{action['text']}]"
    elif actionType == "scroll":
        result = f"SCROLL [{action['direction'].upper()}]"
    elif actionType == "wait":
        result = "WAIT"
    elif actionType == "navigate_home":
        result = "PRESS_HOME"
    elif actionType == "navigate_back":
        result = "PRESS_BACK"   
    elif actionType == "long_press":
        x, y = action["x"], action["y"]
        availableBboxes = []
        for layout in layouts:
            if layout["is_clickable"] and "bbox_pixels" in layout and __in_bbox(x, y, layout["bbox_pixels"]):
                availableBboxes.append(layout["bbox_pixels"])
        smallestBbox = None if availableBboxes == [] else \
            min(
                availableBboxes,
                key=lambda b: (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])
            )
        if smallestBbox is not None:
            center_x, center_y = __get_bbox_center(smallestBbox)
            if math.sqrt(((center_x - x) / width * 1000) ** 2 + ((center_y - y) / height * 1000) ** 2) <= 40:
                x, y = center_x, center_y
        ux, uy = (x / width) * 1000, (y / height) * 1000 
        result = f"LONG_CLICK <point>[[{ux:.1f}, {uy:.1f}]]</point>"
    return result

def convertAITZAction(item:dict, width:int, height:int) -> str:
    from PIL import Image
    action_type = item['result_action_type']
    if action_type == 3:
        action = f"TYPE [{item['result_action_text']}]"
    elif action_type == 4:
        # Extract coordinates
        touch_coords = eval(item['result_touch_yx'])
        lift_coords = eval(item['result_lift_yx'])
        touch_y, touch_x = touch_coords
        lift_y, lift_x = lift_coords
                
        # Check if it's a click (same coordinates)
        if (touch_x == lift_x and touch_y == lift_y) or math.sqrt((touch_x - lift_x)**2 + (touch_y - lift_y)**2) < 0.04:
            bboxes = eval(item["ui_positions"])
            # convertBboxes = [[x, y , (x + w), (y + h)] for y, x, h, w in bboxes]
            convertBboxes = [[x / width * 1000, y / height * 1000, (x + w) / width * 1000, (y + h) / height * 1000] for y, x, h, w in bboxes]
            availableBboxes = []
            for bbox in convertBboxes:
                x1, y1, x2, y2 = bbox
                if x1 <= touch_x * 1000 <= x2 and y1 <= touch_y * 1000 <= y2:
                    availableBboxes.append(bbox)
            smallestBbox = None if availableBboxes == [] else \
                min(
                    availableBboxes,
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
                )
            if smallestBbox is not None:
                center_x, center_y = (smallestBbox[0] + smallestBbox[2]) / 2000, (smallestBbox[1] + smallestBbox[3]) / 2000
                if math.sqrt((center_x - touch_x) ** 2 + (center_y - touch_y) ** 2) <= 0.04:
                    touch_x, touch_y = center_x, center_y
            
            action = f"CLICK <point>[[{touch_x * 1000:.1f}, {touch_y * 1000:.1f}]]</point>"
        # elif math.sqrt((touch_x - lift_x)**2 + (touch_y - lift_y)**2) < 0.04:
        #     action = f"CLICK <point>[[{touch_x*1000:.1f}, {touch_y*1000:.1f}]]</point>"
        else:
            # Calculate deltas
            delta_x = abs(touch_x - lift_x)
            delta_y = abs(touch_y - lift_y)
            
            if delta_y > delta_x:
                if lift_y < touch_y:
                    action = "SCROLL [UP]"
                else:
                    action = "SCROLL [DOWN]"
            else:
                if lift_x < touch_x:
                    action = "SCROLL [LEFT]"
                else:
                    action = "SCROLL [RIGHT]"
                    
    elif action_type == 5:
        action = "PRESS_BACK"
    elif action_type == 6:
        action = "PRESS_HOME"
    elif action_type == 7:
        action = "ENTER"
    elif action_type == 10:
        action = "COMPLETE"
    elif action_type == 11:
        action = "IMPOSSIBLE"
    else:
        raise NotImplementedError(f"{action_type} cannot be parsed")
    return action


def convertGuiOdysseyAction(step:dict):
    actionType = step["action"]
    
    if actionType == "CLICK":
        if step["info"] not in ["KEY_HOME", "KEY_BACK", "KEY_APPSELECT"]:
            x, y = step["info"][0]
            bbox = step["sam2_bbox"]
            center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            if math.sqrt((x - center_x)**2 + (y - center_y)**2) < 40:
                x, y = center_x, center_y
            result = f"CLICK <point>[[{x:.1f}, {y:.1f}]]</point>"
        elif step["info"] == "KEY_HOME":
            result = "PRESS_HOME"
        elif step["info"] == "KEY_BACK":
            result = "PRESS_BACK"
        else:
            raise NotImplementedError(f"action: {actionType}: {step['info']} not implemented")        
    elif actionType == "LONG_PRESS":
        x, y = step["info"][0]
        result = f"LONG_CLICK <point>[[{x:.1f}, {y:.1f}]]</point>"
    elif actionType == "SCROLL":
        x1, y1 = step["info"][0]
        x2, y2 = step["info"][1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) > abs(dy):  # 水平滑动
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        result = f"SCROLL [{direction}]"
    elif actionType == "TEXT":
        result = f"TYPE [{step['info']}]"
    
    elif actionType == "COMPLETE":
        result = "COMPLETE"
    else:
        raise NotImplementedError(f"action: {actionType}: {step['info']} not implemented")
    return result


def stateFrameJudge(client:ZhipuAI, goal, stepInstruction, imgPath):
    model = "glm-4v-flash"
    with open(imgPath, 'rb') as f:
        imgBase = base64.b64encode(f.read()).decode('utf-8')
        
    prompt = """
You are given:
1. A screenshot of a user interface.
2. A goal string that has already been identified as a state-change command: __GOAL__.
3. A step-by-step instruction representing the current step toward the goal: __STEP_INSTRUCT__.

Your task:
Determine whether the current stepInstruction is attempting to change the state of a toggle (such as a switch, button, or setting) as described in the goal, and whether that toggle is visible in the screenshot.

Specifically, do the following:
1. From the goal string, extract:
    - The name of the feature being toggled (target_feature).
    - The expected or desired state ("On" or "Off") that the user wants.
2. Examine the screenshot to determine:
    - Whether the target_feature is visible.
    - If visible, what is its current state ("On" or "Off").
3. Carefully analyze the stepInstruction:
    - Determine whether the current stepInstruction is referring to interacting with this toggle. 

Output:
- If the target_feature is visible in the screenshot and the stepInstruction corresponds to changing its state, return:
{
    "state_step": true,
    "target_feature": <name of the feature being toggled, from the goal>,
    "desired_state": "On" or "Off",
    "current_state": "On" or "Off"
}

- Otherwise, return:
{
    "state_step": false,
    "target_feature": <name of the feature being toggled, from the goal>,
    "desired_state": "On" or "Off"
}
"""


    inputText = prompt.replace("__GOAL__", goal).replace("__STEP_INSTRUCT__", stepInstruction)
    
    try:
        output = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": imgBase
                            }
                        },
                        {
                            "type": "text",
                            "text": inputText
                        }
                    ]
                }
            ],
            response_format={
                'type': 'json_object'
            },
            max_tokens=512
        ).choices[0].message.content
    except zhipuai.core._errors.APIRequestFailedError:
        output = "{\"state_step\": false}"
        
    try:
        outputJson = json.loads(output.replace("True", "true").replace("False", "false"))
    except Exception as e:
        print(f"Error parsing JSON for record: {e}")
        return None

    return outputJson
      

