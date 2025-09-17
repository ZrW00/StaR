from typing import *
from .utils import *
import re, json

def translateAtlas2Uitars(action:str):
    """
    click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')
    long_press(start_box=\'<|box_start|>(x1,y1)<|box_end|>\', time=\'\')
    type(content=\'\')
    scroll(direction=\'down or up or right or left\')
    enter()
    open_app(app_name=\'\')
    press_back()
    press_home()
    wait()
    finished()
    """
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
        uitarsAction = f"long_press(start_box=\'<|box_start|>({x},{y})<|box_end|>\', time=\'\')"
    elif action.startswith("PRESS_BACK"):
        uitarsAction = "press_back()"
    elif action.startswith("PRESS_HOME"):
        uitarsAction = "press_home()"
    elif action.startswith("SCROLL"):
        pattern = r'SCROLL\s+\[([A-Z]+)\]'
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
        raise NotImplementedError(f"action: {action} does not have correpsonding translation target")
        
    return uitarsAction

def translateAtlas2Atlas(action:str):
    return action


def translateUITARS2Atlas(action:str):
    """
    click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')
    long_press(start_box=\'<|box_start|>(x1,y1)<|box_end|>\', time=\'\')
    type(content=\'\')
    scroll(direction=\'down or up or right or left\')
    open_app(app_name=\'\')
    press_back()
    enter()
    press_home()
    wait()
    finished()
    drag(start_box='<|box_start|>(600,820)<|box_end|>', end_box='<|box_start|>(600,451)<|box_end|>')
    """
    if action.startswith("click"):
        pattern = r'\(([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)\)'
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        atlasAction = f"CLICK <point>[[{x}, {y}]]</point>"
    elif action.startswith("finished"):
        atlasAction = "COMPLETE"
    elif action.startswith("long_press"):
        pattern = r'\(([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)\)'
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        atlasAction = f"LONG_CLICK <point>[[{x}, {y}]]</point>"
    elif action.startswith("press_back"):
        atlasAction = "PRESS_BACK"
    elif action.startswith("press_home"):
        atlasAction = "PRESS_HOME"
    elif action.startswith("scroll"):
        pattern = r"scroll\(direction=\\'([a-zA-Z]+)\\'\)"
        match = re.search(pattern, action)
        if match:
            direction = match.group(1).upper()
        else:
            direction = "DOWN"
        atlasAction = f"SCROLL [{direction}]"
    elif action.startswith("open_app"):
        # pattern = r"open_app\(app_name=\\'([^\\']+)\\'\)"
        pattern = r"open_app\(app_name='([^']+)'\)"
        match = re.search(pattern, action)
        if match:
            appName = match.group(1)
        else:
            appName = None
        atlasAction = f"OPENAPP {appName}"
    elif action.startswith("wait"):
        atlasAction = "WAIT"
    elif action.startswith("type"):
        # pattern = r"type\(content=\\'([^\\']+)\\'\)"
        pattern = r"type\(content='([^']+)'\)"
        match = re.search(pattern, action)
        if match:
            content = match.group(1)
        else:
            content = None
        atlasAction = f"TYPE [{content}]"
    elif action.startswith("enter"):
        atlasAction = f"ENTER"
    elif action.startswith("press_enter"):
        atlasAction = f"ENTER"
    elif action.startswith("drag"):
        atlasAction = f"NONEACTTION"
    elif action.startswith("noneaction"):
        atlasAction = f"NONEACTTION"
    else:
        raise NotImplementedError(f"action: {action} does not have correpsonding translation target")
        
    return atlasAction

def translateGUIR12Atlas(action:List[dict], imageWidth, imageHeight):
    """
    "[{'action': enum['complete','press_back', 'press_recent', 'press_home','impossible'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
    "[{'action': enum['click','long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
    "[{'action': enum['type'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
    "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    """
    if action[0]['action'] == "click":
        clickCoordinate = action[0]['point']
        x = round(clickCoordinate[0] / imageWidth * 1000, 1)
        y = round(clickCoordinate[1] / imageHeight * 1000, 1)
        atlasAction = f"CLICK <point>[[{x}, {y}]]</point>"
    elif action[0]['action'] == "complete":
        atlasAction = "COMPLETE"
    elif action[0]['action'] == "long_press":
        clickCoordinate = action[0]['point']
        x = round(clickCoordinate[0] / imageWidth * 1000, 1)
        y = round(clickCoordinate[1] / imageHeight * 1000, 1)
        atlasAction = f"LONG_CLICK <point>[[{x}, {y}]]</point>"
    elif action[0]['action'] == "press_back":
        atlasAction = "PRESS_BACK"
    elif action[0]['action'] == "press_home":
        atlasAction = "PRESS_HOME"
    elif action[0]['action'] == "scroll":
        direction = action[0]['input_text'].upper()
        atlasAction = f"SCROLL [{direction}]"
    elif action[0]['action'] == "wait":
        atlasAction = "WAIT"
    elif action[0]['action'] == "open_app":
        appName = action[0]["input_text"]
        atlasAction = f"OPENAPP {appName}"
    elif action[0]['action'] == "type":
        # pattern = r"type\(content=\\'([^\\']+)\\'\)"
        content = action[0]['input_text']
        atlasAction = f"TYPE [{content}]"
    elif action[0]['action'] in ["press_recent", "impossible"]:
        atlasAction = f"NONEACTTION"
    elif action[0]['action'] == "noneaction":
        atlasAction = f"NONEACTTION"
    else:
        raise NotImplementedError(f"action: {action} does not have correpsonding translation target")
        
    return atlasAction


def translateACG2Atlas(action: Union[Dict, List[Dict]]) -> str:
    """
    "{\"thought\": \"Open the Cx file Explorer and rename the Flowers folder to Flora.\", \"POINT\": [951, 256]}"
    "{\"thought\": \"Share my favorite Book \\\"the Queen's Gambit\\\" to my Friend Natalie larson over her gmail address -natalie.larson1998@gmail.com from the  PocketBook app.\", \"POINT\": [108, 272], \"duration\": 1000}"
    "{\"thought\": \"Show me some of the sustainability art pieces on the Pinterest app for my research on sustainable energy.\", \"duration\": 500}"
    "{\"thought\": \"In Google News listen the \\\"Kevin Cahoon:Let's Get Shucked! \\\" podcast on Broadway Podcast Network\", \"POINT\": [500, 500], \"to\": \"down\"}"
    "{\"thought\": \"Open the Cx file Explorer and rename the Flowers folder to Flora.\", \"TYPE\": \"Flora\"}"
    "{\"thought\": \"Turn on device location & give the location access to google app\", \"PRESS\": \"BACK\"}"
    "{\"thought\": \"Open the Cx file Explorer and rename the Flowers folder to Flora.\", \"STATUS\": \"finish\"}"
    """
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
        if "duration" in act : 
            atlas_action = f"LONG_CLICK <point>[[{act['POINT'][0]}, {act['POINT'][1]}]]</point>"
        else:
            atlas_action = f"CLICK <point>[[{act['POINT'][0]}, {act['POINT'][1]}]]</point>"

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

def translateAtlas2ACG(action: str) -> Dict:
    """
    将 Atlas 风格的动作字符串转换为 ACG 格式的 JSON 字典
    例如：
        "CLICK <point>[[951, 256]]</point>" 
        => {"POINT": [951, 256]}
        "LONG_CLICK <point>[[951, 256]]</point>"
        => {"POINT": [951, 256], "duration": 1000}
        "SCROLL [DOWN]"
        => {"to": "down"}
        "TYPE [Flora]"
        => {"TYPE": "Flora"}
        "PRESS_HOME"
        => {"PRESS": "HOME"}
        "COMPLETE"
        => {"STATUS": "finish"}
        "WAIT"
        => {"duration": 500}
    """
    action = action.strip()
    acg_action = {}

    # CLICK
    if action.startswith("CLICK"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            acg_action["POINT"] = [x, y]

    # LONG_CLICK
    elif action.startswith("LONG_CLICK"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            acg_action["POINT"] = [x, y]
            acg_action["duration"] = 1000

    # SCROLL
    elif action.startswith("SCROLL"):
        match = re.search(r"SCROLL\s*\[([A-Z]+)\]", action)
        if match:
            raw_direction = match.group(1).lower()
            if raw_direction == "up":
                direction = "down"
            elif raw_direction == "down":
                direction = "up"
            elif raw_direction == "left":
                direction = "right"
            elif raw_direction == "right":
                direction = "left"
        acg_action["to"] = direction

    # TYPE
    elif action.startswith("TYPE"):
        match = re.search(r"TYPE\s*\[(.*?)\]", action)
        if match:
            acg_action["TYPE"] = match.group(1)

    # PRESS
    elif action.startswith("PRESS_"):
        key = action.replace("PRESS_", "")
        if key in ["HOME", "BACK", "ENTER"]:
            acg_action["PRESS"] = key
        else:
            acg_action["PRESS"] = key

    # COMPLETE
    elif action == "COMPLETE":
        acg_action["STATUS"] = "finish"

    # WAIT
    elif action == "WAIT":
        acg_action["duration"] = 500

    else:
        return None

    return acg_action
 
def translateAtlas2GUIR1(action:List[dict], imageWidth, imageHeight):
    """
    "[{'action': enum['complete','press_back', 'press_recent', 'press_home','impossible'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
    "[{'action': enum['click','long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
    "[{'action': enum['type'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
    "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    "[{'action': enum['wait', 'press_back'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
    "[{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
    "[{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
    "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    """
    if action.startswith("CLICK"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        px, py = round(x / 1000 * imageWidth), round(y / 1000 * imageHeight)
        guir1Action = "[{'action': 'click', 'point': [__X__, __Y__], 'input_text': 'no input text'}]".replace("__X__", str(px)).replace("__Y__", str(py))
    elif action.startswith("COMPLETE"):
        guir1Action = "[{'action': 'complete', 'point': [-100, -100], 'input_text': 'no input text'}]"
    elif action.startswith("LONG_CLICK"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        px, py = round(x / 1000 * imageWidth), round(y / 1000 * imageHeight)
        guir1Action = "[{'action': 'long_press', 'point': [__X__, __Y__], 'input_text': 'no input text'}]".replace("__X__", str(px)).replace("__Y__", str(py))
    elif action.startswith("PRESS_BACK"):
        guir1Action = "[{'action': 'press_back', 'point': [-100, -100], 'input_text': 'no input text'}]"
    elif action.startswith("PRESS_HOME"):
        guir1Action = "[{'action': 'press_home', 'point': [-100, -100], 'input_text': 'no input text'}]"
    elif action.startswith("SCROLL"):
        pattern = r'SCROLL\s+\[([A-Z]+)\]'
        match = re.search(pattern, action)
        if match:
            direction = match.group(1).lower()
        else:
            direction = "down"
        guir1Action = "[{'action': 'scroll', 'point': [-100, -100], 'input_text': '__DIRECTION__'}]".replace("__DIRECTION__", direction)
    elif action.startswith("OPENAPP"):
        appName = action.replace("OPENAPP", "").strip()
        guir1Action = "[{'action': 'open_app', 'point': [-100, -100], 'input_text': '__APPNAME__'}]".replace("__APPNAME__", appName)
    elif action.startswith("WAIT"):
        guir1Action = "[{'action': 'wait', 'point': [-100, -100], 'input_text': 'no input text'}]"
    elif action.startswith("TYPE"):
        pattern = r"TYPE \[(.*?)\]"
        match = re.search(pattern, action)
        if match:
            content = match.group(1)
        else:
            content = None
        guir1Action = "[{'action': 'type', 'point': [-100, -100], 'input_text': '__CONTENT__'}]".replace("__CONTENT__", content)
    else:
        raise NotImplementedError(f"action: {action} does not have correpsonding translation target")
    
    return guir1Action

def translateAtlas2Qwen25vl(action:List[dict], imageWidth, imageHeight):
    """
    "[{'action': enum['complete','press_back', 'press_recent', 'press_home','impossible'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
    "[{'action': enum['click','long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
    "[{'action': enum['type'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
    "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    "[{'action': enum['wait', 'press_back'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
    "[{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
    "[{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
    "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    """
    if action.startswith("CLICK"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        px, py = round(x / 1000 * imageWidth), round(y / 1000 * imageHeight)
        qwen25vlAction = f"CLICK <point>[[{px}, {py}]]</point>"
        
    elif action.startswith("COMPLETE"):
        qwen25vlAction = "COMPLETE"
    elif action.startswith("LONG_CLICK"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        px, py = round(x / 1000 * imageWidth), round(y / 1000 * imageHeight)
        qwen25vlAction = f"LONG_CLICK <point>[[{px}, {py}]]</point>"
    elif action.startswith("PRESS_BACK"):
        qwen25vlAction = "PRESS_BACK"
    elif action.startswith("PRESS_HOME"):
        qwen25vlAction = "PRESS_HOME"
    elif action.startswith("SCROLL"):
        pattern = r'SCROLL\s+\[([A-Z]+)\]'
        match = re.search(pattern, action)
        if match:
            direction = match.group(1).upper()
        else:
            direction = "down"
        qwen25vlAction = f"SCROLL [{direction}]"
    elif action.startswith("OPENAPP"):
        appName = action.replace("OPENAPP", "").strip()
        qwen25vlAction = f"OPENAPP {appName}"
    elif action.startswith("WAIT"):
        qwen25vlAction = "WAIT"
    elif action.startswith("TYPE"):
        pattern = r"TYPE \[(.*?)\]"
        match = re.search(pattern, action)
        if match:
            content = match.group(1)
        else:
            content = None
        qwen25vlAction = f"TYPE [{content}]"
    else:
        raise NotImplementedError(f"action: {action} does not have correpsonding translation target")
    
    return qwen25vlAction

def translateQwen25vl2Atlas(action:List[dict], imageWidth, imageHeight):
    """
    "[{'action': enum['complete','press_back', 'press_recent', 'press_home','impossible'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
    "[{'action': enum['click','long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
    "[{'action': enum['type'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
    "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    "[{'action': enum['wait', 'press_back'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
    "[{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
    "[{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
    "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    """
    if action.startswith("CLICK"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            px, py = float(match.group(1)), float(match.group(2))
        else:
            px, py = None, None
        ux, uy = round(px / imageWidth * 1000, 1), round(py / imageHeight * 1000, 1)
        atlasAction = f"CLICK <point>[[{ux}, {uy}]]</point>"
        
    elif action.startswith("COMPLETE"):
        atlasAction = "COMPLETE"
    elif action.startswith("LONG_CLICK"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            px, py = float(match.group(1)), float(match.group(2))
        else:
            px, py = None, None
        ux, uy = round(px / imageWidth * 1000, 1), round(py / imageHeight * 1000, 1)
        atlasAction = f"LONG_CLICK <point>[[{ux}, {uy}]]</point>"
    elif action.startswith("PRESS_BACK"):
        atlasAction = "PRESS_BACK"
    elif action.startswith("PRESS_HOME"):
        atlasAction = "PRESS_HOME"
    elif action.startswith("SCROLL"):
        pattern = r'SCROLL\s+\[([A-Z]+)\]'
        match = re.search(pattern, action)
        if match:
            direction = match.group(1).upper()
        else:
            direction = "down"
        atlasAction = f"SCROLL [{direction}]"
    elif action.startswith("OPENAPP"):
        appName = action.replace("OPENAPP", "").strip()
        atlasAction = f"OPENAPP {appName}"
    elif action.startswith("WAIT"):
        atlasAction = "WAIT"
    elif action.startswith("TYPE"):
        pattern = r"TYPE \[(.*?)\]"
        match = re.search(pattern, action)
        if match:
            content = match.group(1)
        else:
            content = None
        atlasAction = f"TYPE [{content}]"
    else:
        raise NotImplementedError(f"action: {action} does not have correpsonding translation target")
    
    return atlasAction



def actionTypeJudge(rawPredictAction:str, rawGroundTruthAction:str):
    if rawGroundTruthAction.startswith("action:"):
        rawGroundTruthAction = rawGroundTruthAction.replace("action:", "")
    if rawPredictAction.startswith("action:"):
        rawPredictAction = rawPredictAction.replace("action:", "")
    predictAction = qwen_translate_action(rawPredictAction)
    groundTruthAction = qwen_translate_action(rawGroundTruthAction)
    return predictAction.action_type == groundTruthAction.action_type

def androidControlActionJudge(rawPredictAction:str, rawGroundTruthAction:str, record:Dict):
    layoutPath, width, height = record["layouts"][0], record["width"], record["height"]
    def __in_bbox(x, y, bbox):
        return bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]
    def __box_center_distance(x, y, bbox):
        center_x, center_y = (bbox["x_min"] + bbox["x_max"]) / 2, (bbox["y_min"] + bbox["y_max"]) / 2
        return math.sqrt(((center_x - x) / width * 1000) ** 2 + ((center_y - y) / height * 1000) ** 2) <= 40
        return math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2) <= 0.04
    
    if rawGroundTruthAction.startswith("Action:"):
        rawGroundTruthAction = rawGroundTruthAction.replace("Action:", "")
    if rawPredictAction.startswith("Action:"):
        rawPredictAction = rawPredictAction.replace("Action:", "")
    predictAction = qwen_translate_action(rawPredictAction)
    groundTruthAction = qwen_translate_action(rawGroundTruthAction)
    
        
    if predictAction.action_type != groundTruthAction.action_type:
        return False
    elif predictAction.action_type == ActionType.TYPE:
        # return predictAction.typed_text == groundTruthAction.typed_text
        return predictAction.typed_text.strip().lower() == groundTruthAction.typed_text.strip().lower()
    elif predictAction.action_type == ActionType.OPENAPP:
        stemmer = PorterStemmer()
        def preprocess(name):
            words = [word.lower() for word in ''.join(
                e if e.isalnum() or e.isspace() else ' ' 
                for e in name
            ).split()]
            stop_words = {"the", "a", "an"}
            stems = [stemmer.stem(word) for word in words if word not in stop_words]
            return stems

        predict_words = preprocess(predictAction.app_name)
        truth_words = preprocess(groundTruthAction.app_name)

        if predict_words == truth_words:
            return True

        predict_merged = "".join(predict_words)
        truth_merged = "".join(truth_words)
        if predict_merged in truth_merged or truth_merged in predict_merged:
            return True

        if set(predict_words) == set(truth_words):
            return True

        return False
    elif predictAction.action_type == ActionType.CLICK or predictAction.action_type == ActionType.LONGCLICK:
        x1, y1 = predictAction.point
        x2, y2 = groundTruthAction.point
        x2f, y2f = x2 / 1000 * width, y2 / 1000 * height
        
        with open(layoutPath, "r") as f:
            layouts = [json.loads(line.strip()) for line in f]
        
        inboundFlag, distFlag = False, False
        availableBboxes = []
        for layout in layouts:
            if layout["is_clickable"] and "bbox_pixels" in layout and __in_bbox(x2f, y2f, layout["bbox_pixels"]) and __box_center_distance(x2f, y2f, layout["bbox_pixels"]):
                # inboundFlag = __in_bbox(x1 / 1000 * width, y1 / 1000 * width, layout["bbox_pixels"])
                # break
                availableBboxes.append(layout["bbox_pixels"])
        smallestBbox = None if availableBboxes == [] else \
            min(
                availableBboxes,
                key=lambda b: (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])
            )
        if smallestBbox is not None:
            inboundFlag = __in_bbox(x1 / 1000 * width, y1 / 1000 * height, smallestBbox)
    
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distFlag = distance <= 140
        return inboundFlag or distFlag
    elif groundTruthAction.action_type in [ActionType.SCROLLUP, ActionType.SCROLLDOWN, ActionType.SCROLLLEFT, ActionType.SCROLLRIGHT, ActionType.PRESSBACK, ActionType.PRESSHOME, ActionType.TASKIMPOSSIBLE, ActionType.TASKCOMPLETE, ActionType.WAIT, ActionType.ENTER]:
        return predictAction.action_type == groundTruthAction.action_type
    else:
        raise NotImplementedError(f"comparison between pred action: {predictAction}\nand ground truth action: {groundTruthAction}\n is not implemented!")


def guiodysseyActionJudge(rawPredictAction:str, rawGroundTruthAction:str, record:Dict):
    def __in_bbox(x, y, bbox):
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
    bbox = record["bbox"]
    if rawGroundTruthAction.startswith("action:"):
        rawGroundTruthAction = rawGroundTruthAction.replace("action:", "")
    if rawPredictAction.startswith("action:"):
        rawPredictAction = rawPredictAction.replace("action:", "")
    predictAction = qwen_translate_action(rawPredictAction)
    groundTruthAction = qwen_translate_action(rawGroundTruthAction)
    
        
    if predictAction.action_type != groundTruthAction.action_type:
        return False
    elif predictAction.action_type == ActionType.TYPE:
        # return predictAction.typed_text == groundTruthAction.typed_text
        return predictAction.typed_text.strip().lower() == groundTruthAction.typed_text.strip().lower()
    elif predictAction.action_type == ActionType.CLICK or predictAction.action_type == ActionType.LONGCLICK:
        x1, y1 = predictAction.point
        x2, y2 = groundTruthAction.point
        
        inboundFlag, distFlag = False, False
        
        inboundFlag = __in_bbox(x1 , y1 , bbox)
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distFlag = distance <= 140
        return inboundFlag or distFlag
    elif groundTruthAction.action_type in [ActionType.SCROLLUP, ActionType.SCROLLDOWN, ActionType.SCROLLLEFT, ActionType.SCROLLRIGHT, ActionType.PRESSBACK, ActionType.PRESSHOME, ActionType.TASKIMPOSSIBLE, ActionType.TASKCOMPLETE, ActionType.WAIT, ActionType.ENTER]:
        return predictAction.action_type == groundTruthAction.action_type
    else:
        raise NotImplementedError(f"comparison between pred action: {predictAction}\nand ground truth action: {groundTruthAction}\n is not implemented!")
    

def aitzActionJudge(rawPredictAction:str, rawGroundTruthAction:str, record:Dict):
    def __in_bbox(x, y, bbox):
        return bbox[0] <= x  <= bbox[2] and bbox[1] <= y <= bbox[3]
    def __box_center_distance(x, y, bbox):
        center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2) <= 40

    bboxes = record["bboxes"]
    if rawGroundTruthAction.startswith("action:"):
        rawGroundTruthAction = rawGroundTruthAction.replace("action:", "")
    if rawPredictAction.startswith("action:"):
        rawPredictAction = rawPredictAction.replace("action:", "")
    predictAction = qwen_translate_action(rawPredictAction)
    groundTruthAction = qwen_translate_action(rawGroundTruthAction)
    
        
    if predictAction.action_type != groundTruthAction.action_type:
        return False
    elif predictAction.action_type == ActionType.TYPE:
        # return predictAction.typed_text.strip() == groundTruthAction.typed_text.strip()
        return predictAction.typed_text.strip().lower() == groundTruthAction.typed_text.strip().lower()
    elif predictAction.action_type == ActionType.OPENAPP:
        return predictAction.app_name == groundTruthAction.app_name
    elif predictAction.action_type == ActionType.CLICK or predictAction.action_type == ActionType.LONGCLICK:
        x1, y1 = predictAction.point
        x2, y2 = groundTruthAction.point
        
        inboundFlag, distFlag = False, False
        availableBboxes = []
        for bbox in bboxes:
            if __in_bbox(x2, y2, bbox) and __box_center_distance(x2, y2, bbox):
                # inboundFlag = __in_bbox(x1, y1, bbox)
                # break
                availableBboxes.append(bbox)
        smallestBbox = None if availableBboxes == [] else \
            min(
                availableBboxes,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
            )
        if smallestBbox is not None:
            inboundFlag = __in_bbox(x1, y1, smallestBbox)
    
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distFlag = distance <= 140
        return inboundFlag or distFlag
    elif groundTruthAction.action_type in [ActionType.SCROLLUP, ActionType.SCROLLDOWN, ActionType.SCROLLLEFT, ActionType.SCROLLRIGHT, ActionType.PRESSBACK, ActionType.PRESSHOME, ActionType.TASKIMPOSSIBLE, ActionType.TASKCOMPLETE, ActionType.WAIT, ActionType.ENTER]:
        return predictAction.action_type == groundTruthAction.action_type
    else:
        raise NotImplementedError(f"comparison between pred action: {predictAction}\nand ground truth action: {groundTruthAction}\n is not implemented!")
    
def stateActionTypeJudge(rawPredictAction:str, rawGroundTruthAction:str):
    if rawGroundTruthAction.startswith("action:"):
        rawGroundTruthAction = rawGroundTruthAction.replace("action:", "")
    if rawPredictAction.startswith("action:"):
        rawPredictAction = rawPredictAction.replace("action:", "")
    predictAction = qwen_translate_action(rawPredictAction)
    groundTruthAction = qwen_translate_action(rawGroundTruthAction)
    return predictAction.action_type == groundTruthAction.action_type

def stateActionJudge(rawPredictAction:str, rawGroundTruthAction:str, record:dict):
    predictAction = qwen_translate_action(rawPredictAction)
    groundTruthAction = qwen_translate_action(rawGroundTruthAction)
        
    if predictAction.action_type != groundTruthAction.action_type:
        return False
    elif predictAction.action_type == ActionType.TYPE:
        return predictAction.typed_text == groundTruthAction.typed_text
    elif predictAction.action_type == ActionType.CLICK:
        x1, y1 = predictAction.point
        x2, y2 = groundTruthAction.point
        
        inboundFlag, distFlag = False, False
        if record["useBbox"]:
            inboundFlag = record["bbox"][0] <= x1 <= record["bbox"][2] and record["bbox"][1] <= y1 <= record["bbox"][3]
            
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distFlag = distance <= 40
        return inboundFlag or distFlag
    elif groundTruthAction.action_type in [ActionType.SCROLLUP, ActionType.SCROLLDOWN, ActionType.SCROLLLEFT, ActionType.SCROLLRIGHT, ActionType.PRESSBACK, ActionType.PRESSHOME, ActionType.TASKIMPOSSIBLE, ActionType.TASKCOMPLETE, ActionType.WAIT, ActionType.ENTER]:
        return predictAction.action_type == groundTruthAction.action_type
    else:
        raise NotImplementedError(f"comparison between pred action: {predictAction}\nand ground truth action: {groundTruthAction}\n is not implemented!")


