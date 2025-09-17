from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict,Union
import re, json, math
import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer

class ActionType(Enum):
    CLICK=1
    TYPE=2
    SCROLLUP=3
    SCROLLDOWN=4
    SCROLLLEFT=5
    SCROLLRIGHT=6
    PRESSBACK=7
    PRESSHOME=8
    TASKIMPOSSIBLE=9
    TASKCOMPLETE=10
    NONEACTION=11
    WAIT=12
    LONGCLICK=13
    OPENAPP=14
    ENTER=15
    
@dataclass
class AndroidAction():
    action_type: ActionType
    point: Tuple[float, float] = None
    typed_text: str = None
    app_name:str = None

    def __str__(self):
        # Construct the basic action type string.
        components = [f"Action Type: {self.action_type.name}"]

        # Format and add touch_point if it's not None.
        if self.point:
            touch_point_str = f"({self.point[0]:.1f}, {self.point[1]:.1f})"
            components.append(f"Click Point: {touch_point_str}")
        if self.typed_text:
            components.append(f"Typed Text: '{self.typed_text}'")
        return ", ".join(components)

    def to_act(self):
        pass


def qwen_translate_action(action:str):
    if action == "PRESS_BACK":
        return AndroidAction(action_type=ActionType.PRESSBACK)
    elif action == "PRESS_HOME":
        return AndroidAction(action_type=ActionType.PRESSHOME)
    elif action == "ENTER":
        return AndroidAction(action_type=ActionType.ENTER)
    elif action == "COMPLETE":
        return AndroidAction(action_type=ActionType.TASKCOMPLETE)
    elif action == "IMPOSSIBLE":
        return AndroidAction(action_type=ActionType.TASKIMPOSSIBLE)
    elif action == "SCROLL [RIGHT]":
        return AndroidAction(action_type=ActionType.SCROLLRIGHT)
    elif action == "SCROLL [LEFT]":
        return AndroidAction(action_type=ActionType.SCROLLLEFT)
    elif action == "SCROLL [UP]":
        return AndroidAction(action_type=ActionType.SCROLLUP)
    elif action == "SCROLL [DOWN]":
        return AndroidAction(action_type=ActionType.SCROLLDOWN)
    elif action == "WAIT":
        return AndroidAction(action_type=ActionType.WAIT)
    elif action.startswith("TYPE [") and action.endswith("]"):
        pattern = r"TYPE \[(.*?)\]"
        match = re.search(pattern, action)
        text = match.group(1) if match else ""
        return AndroidAction(action_type=ActionType.TYPE, typed_text=text)
    elif action.startswith("OPENAPP"):
        appName = action.replace("OPENAPP", "").strip().lower()
        return AndroidAction(action_type=ActionType.OPENAPP, app_name=appName)
    elif action.startswith("CLICK <point>[[") and action.endswith("]]</point>"):
        # pattern = r"\[\[([0-9.-]+),([0-9.-]+)\]\]"
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        return AndroidAction(action_type=ActionType.CLICK, point=(x, y))  
    elif action.startswith("LONG_CLICK <point>[[") and action.endswith("]]</point>"):
        # pattern = r"\[\[([0-9.-]+),([0-9.-]+)\]\]"
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        return AndroidAction(action_type=ActionType.LONGCLICK, point=(x, y))  
    else:
        # raise ValueError(f"translation of action {action} is not implemented")
        return AndroidAction(action_type=ActionType.NONEACTION)

def extractUITARSPrediction(text:str):
    # text example:  "Thought: Click the button\nAction: click(start_box='(100,200)')"
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
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
            
    action_pattern = r"(?i)(?:Output|Action|Actions)[：:][\s\n]*(?:```(?:plaintext)?\n)?(.+?)(?:\n```)?$"
    try:
        actionMatch = re.search(action_pattern, text, re.DOTALL)
        if actionMatch:
            action = actionMatch.group(1).strip() 
        else:
            action = re.sub(r'^(action|actions)[：:]?\s*', '', text, flags=re.IGNORECASE).strip()
    except Exception as e:
        action = re.sub(r'^(action|actions)[：:]?\s*', '', text, flags=re.IGNORECASE).strip()
    return {"thought":thought, "action":action}

def extractGUIR1Prediction(text:str):
    thought, answer = "", None
    answerPattern = r'<answer>(.*?)</answer>'
    thoughtPattern = r'<think>(.*?)</think>'
    answerMatch = re.search(answerPattern, text, re.DOTALL)
    thoughtMatch = re.search(thoughtPattern, text, re.DOTALL)
    
    if answerMatch:
        answer_content = answerMatch.group(1)
        try:
            answer = json.loads(answer_content.replace("'", '"'))
        except json.JSONDecodeError as e:
            pass
    if thoughtMatch:
        thought = thoughtMatch.group(1)
        
    return {"thought":thought, "action":answer}



def extractCPMPrediction(text:dict):
    predThought = text.get("thought", "")
    predAction = text
    return predThought, predAction