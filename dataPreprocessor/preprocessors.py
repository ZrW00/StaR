from .templates import *
from .actions import *
from .utils import *
import os, json, json5, re, base64, time
import argparse
import zhipuai, base64
from zhipuai import ZhipuAI
import copy, yaml
from tqdm import tqdm
import math, traceback
from PIL import Image
from pathlib import Path
import multiprocessing as mp
from typing import *
import random
random.seed(3407)
from argparse import Namespace



class AgenticDataPreprocessor:
    def __init__(self, args):
        self.args = args
        self.model = args.model
        self.apiKey = args.apiKey if hasattr(args, "apiKey") else "Your API Key"
        
        self.client = ZhipuAI(api_key=self.apiKey)
        self.agentCount = args.agentCount if hasattr(args, "agentCount") else 10
        self.llamafactory = args.llamafactory if hasattr(args, "llamafactory") else False
        self.cot_trained = args.cot_trained if hasattr(args, "cot_trained") else False
        
        
        self.settings = "_".join(["state" if args.state else "", "low" if args.low_level else "high", "cot_trained" if args.cot_trained else "", "llamafactory" if self.llamafactory else ""]).strip("_").replace("__", "_")
        self.trainData, self.testData = [], []
        self.savePathTrain, self.savePathTest = None, None
        self.savePathStateTrain, self.savePathNoneStateTrain = None, None
        
    def preprocessEpisode(self, episode_identifier: Union[int, str, Path]):
        raise NotImplementedError("Please implement this method in the subclass")
        
    def preprocess(self):
        trainProcessedData, testProcessedData = [], []
        
        for episode_identifier in tqdm(self.trainData, total=len(self.trainData), desc="train"):
            trainProcessedData.extend(self.preprocessEpisode(episode_identifier))
        for episode_identifier in tqdm(self.testData, total=len(self.testData), desc="test"):
            testProcessedData.extend(self.preprocessEpisode(episode_identifier))
            
        stateTrainProcessedData = [record for record in trainProcessedData if record['stateEpisode']]
        noneStateTrainProcessedData = [record for record in testProcessedData if not record['stateEpisode']]
        
        trainProcessedData.sort(key=lambda x: x["episodeID"])
        testProcessedData.sort(key=lambda x: x["episodeID"])
        stateTrainProcessedData.sort(key=lambda x: x["episodeID"])
        noneStateTrainProcessedData.sort(key=lambda x: x["episodeID"])

        return trainProcessedData, testProcessedData, stateTrainProcessedData, noneStateTrainProcessedData
    
    
    def preprocessChunk(self, agentID, chunk, chunk_start, preprocessData, listLock, progressCounter, progressLock, model_loaded_event, status_queue):
        model_loaded_event.set()
        try:
            for i, episode_identifier in enumerate(chunk):
                episodeData = self.preprocessEpisode(episode_identifier)
                with listLock:
                    preprocessData.extend(episodeData)
                with progressLock:
                    progressCounter.value += 1
            status_queue.put((agentID, "success", f"Processed {len(chunk)} episodes starting from index {chunk_start}"))
        except Exception as e:
            logger.warning(f"error: {e} occor during predict chunk record {i} at agent {agentID}")
            status_queue.put((agentID, "error", f"{e} at {traceback.format_exc()}"))

    def preprocessMP(self):
        manager = mp.Manager()
        
        trainProcessedData, testProcessedData = manager.list(), manager.list()

        trainProgressCounter, testProgressCounter = manager.Value('i', 0), manager.Value('i', 0)
        trainProgressLock, testProgressLock = manager.Lock(), manager.Lock()
        trainListLock, testListLock = manager.Lock(), manager.Lock()
        train_status_queue, test_status_queue = manager.Queue(), manager.Queue()
        trainChunkSize = math.ceil(len(self.trainData) / self.agentCount)
        testChunkSize = math.ceil(len(self.testData) / self.agentCount)
        trainChunks = [self.trainData[i:i + trainChunkSize] for i in range(0, len(self.trainData), trainChunkSize)]
        
        testChunks = [self.testData[i:i + testChunkSize] for i in range(0, len(self.testData), testChunkSize)]

        loadManagerTrain = mp.Manager()
        model_loaded_events_train = [loadManagerTrain.Event() for _ in range(self.agentCount)]
        
        trainProcesses = []
        
        for i, chunk in enumerate(trainChunks):
            chunk_start = i * trainChunkSize
            p = mp.Process(
                target=self.preprocessChunk, 
                args=(i, chunk, chunk_start, trainProcessedData, trainListLock, trainProgressCounter, trainProgressLock, model_loaded_events_train[i], train_status_queue),
                daemon=False
            )
            p.start()
            trainProcesses.append(p)
            
        for event in model_loaded_events_train:
            event.wait()
            
        with tqdm(total=len(self.trainData), desc="Preprocessing training records") as pbar:
            last_progress = 0
            while any(p.is_alive() for p in trainProcesses):
                with trainProgressLock:
                    current_progress = trainProgressCounter.value

                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(0.5)
            
            for p in trainProcesses:
                p.join(timeout=1080000)
                if p.is_alive():
                    # print(f"Process {p.pid} timed out. Terminating.")
                    logger.warning(f"Process {p.pid} timed out. Terminating.")
                    p.terminate()
                    p.join()
                    
            statuses = []
            while not train_status_queue.empty():
                statuses.append(train_status_queue.get())
            for sid, status, info in statuses:
                logger.info(f"[Agent {sid}] Status: {status}, Info: {info}")
                
            failed_agents = [s for s in statuses if s[1] != "success"]
            if failed_agents:
                logger.warning(f"\n {len(failed_agents)} agents failed. You may need to retry or debug.")
        
        loadManagerTest = mp.Manager()
        model_loaded_events_test = [loadManagerTest.Event() for _ in range(self.agentCount)]
        
        testProcesses = []
        
        for i, chunk in enumerate(testChunks):
            chunk_start = i * testChunkSize
            p = mp.Process(
                target=self.preprocessChunk, 
                args=(i, chunk, chunk_start, testProcessedData, testListLock, testProgressCounter, testProgressLock, model_loaded_events_test[i], test_status_queue),
                daemon=False
            )
            p.start()
            testProcesses.append(p)
            
        for event in model_loaded_events_test:
            event.wait()
            
        with tqdm(total=len(self.testData), desc="Preprocessing test records") as pbar:
            last_progress = 0
            while any(p.is_alive() for p in testProcesses):
                with testProgressLock:
                    current_progress = testProgressCounter.value

                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(0.5)
            
            for p in testProcesses:
                p.join(timeout=1080000)
                if p.is_alive():
                    # print(f"Process {p.pid} timed out. Terminating.")
                    logger.warning(f"Process {p.pid} timed out. Terminating.")
                    p.terminate()
                    p.join()
                    
            statuses = []
            while not test_status_queue.empty():
                statuses.append(test_status_queue.get())
            for sid, status, info in statuses:
                logger.info(f"[Agent {sid}] Status: {status}, Info: {info}")
                
            failed_agents = [s for s in statuses if s[1] != "success"]
            if failed_agents:
                logger.warning(f"\n {len(failed_agents)} agents failed. You may need to retry or debug.")
        
        
        trainProcessedData, testProcessedData = list(trainProcessedData), list(testProcessedData)
        
        stateTrainProcessedData = [record for record in trainProcessedData if record['stateEpisode']]
        noneStateTrainProcessedData = [record for record in trainProcessedData if not record['stateEpisode']]
        
        trainProcessedData.sort(key=lambda x: x["episodeID"])
        testProcessedData.sort(key=lambda x: x["episodeID"])
        stateTrainProcessedData.sort(key=lambda x: x["episodeID"])
        noneStateTrainProcessedData.sort(key=lambda x: x["episodeID"])

        return trainProcessedData, testProcessedData, stateTrainProcessedData, noneStateTrainProcessedData
        
        
    def saveData(self, trainProcessedData=None, testProcessedData=None, stateTrainProcessedData=None, noneStateTrainProcessedData=None):
        if trainProcessedData is not None and len(trainProcessedData) > 0:
            with open(self.savePathTrain, "w") as f:
                print(f"Save {len(trainProcessedData)} records to {self.savePathTrain}")
                json.dump(trainProcessedData, f, indent=4)
        
        if testProcessedData is not None and len(testProcessedData) > 0:
            with open(self.savePathTest, "w") as f:
                print(f"Save {len(testProcessedData)} records to {self.savePathTest}")
                json.dump(testProcessedData, f, indent=4, ensure_ascii=False)
        
        if stateTrainProcessedData is not None and len(stateTrainProcessedData) > 0:
            with open(self.savePathStateTrain, "w") as f:
                print(f"Save {len(stateTrainProcessedData)} records to {self.savePathStateTrain}")
                json.dump(stateTrainProcessedData, f, indent=4, ensure_ascii=False)
        
        if noneStateTrainProcessedData is not None and len(noneStateTrainProcessedData) > 0:
            with open(self.savePathNoneStateTrain, "w") as f:
                print(f"Save {len(noneStateTrainProcessedData)} records to {self.savePathNoneStateTrain}")
                json.dump(noneStateTrainProcessedData, f, indent=4, ensure_ascii=False)
        
    
class AndroidControlPreprocessor(AgenticDataPreprocessor):
    def __init__(self, args):
        super().__init__(args)
        self.jsonPath = args.acjsonPath if (hasattr(args, "acjsonPath") and args.acjsonPath is not None) else "GUIData/android_control/jsons"
        self.imagePath = args.acimagePath if (hasattr(args, "acimagePath") and args.acimagePath is not None) else "GUIData/android_control/images"
        self.layoutPath = args.aclayoutPath if (hasattr(args, "aclayoutPath") and args.aclayoutPath is not None) else "GUIData/android_control/layouts"
        
        
        
        with open('android_control_splits.json', 'r') as f:
            self.splits = json.load(f)
        
        self.savePathTrain = f"./data/GUIAgentic/android_control/{self.model}/android_control_{self.settings}_action_predict_train.json"
        self.savePathTest = f"./data/GUIAgentic/android_control/{self.model}/android_control_{self.settings}_action_predict_test.json"
        
        self.savePathStateTrain = f"./data/GUIAgentic/android_control/{self.model}/android_control_{self.settings}_action_predict_state_train.json"
        
        self.savePathNoneStateTrain = f"./data/GUIAgentic/android_control/{self.model}/android_control_{self.settings}_action_predict_none_state_train.json"
        
        os.makedirs(os.path.dirname(self.savePathTrain), exist_ok=True)
        
        self.trainData = self.splits["train"] + self.splits["validation"]
        self.testData = self.splits["test"]
        
    def preprocessEpisode(self, episode_identifier:int):
        preprocessData = []
        metaDataPath = os.path.join(self.jsonPath, f"metadata_episode_{episode_identifier}.json")
        with open(metaDataPath, "r") as f:
            metaData = json.load(f)
            
            
        pattern = r"(?:\b(turn|switch|power|flip|activate|deactivate|start|shut|enable|disable|engage|disengage|cut|kill|press|push|hit)\b\s+(?:the\s+)?\b\w*\b\s*(on|off|up|down|onwards|back|engaged|disengaged)|activate|deactivate|start\s+up|shut\s+down|enable|disable|engage|disengage|cut\s+the\s+power|kill\s+the\s+power)"
        
        stateEpisode = bool(re.search(pattern, metaData["goal"], re.IGNORECASE))
        metaData["stateEpisode"] = stateEpisode
        if self.args.state and not stateEpisode:
            return []
            
        heights, widths = metaData["screenshot_heights"], metaData["screenshot_widths"]
        actions, screenshots, layouts = metaData["actions"], metaData["screenshots"], metaData["accessibility_trees"]
        step_instructions = metaData["step_instructions"]
        step_instructions.append("The task is completed.")
        screenshots = [os.path.join(self.imagePath, f"episode_{episode_identifier}_{os.path.basename(screenshot)}") for screenshot in screenshots]
        layouts = [os.path.join(self.layoutPath, f"episode_{episode_identifier}_{os.path.basename(layout)}") for layout in layouts]
        assert (len(actions) == len(screenshots) - 1) and (len(screenshots) == len(layouts))
        try:
            actionStrs = [convertAndroidControlAction(actions[i], layouts[i], widths[i], heights[i]) for i in range(len(actions))]
        except NotImplementedError as e:
            logger.warning(f"Convert Action Error {e}")
        
        stateAnnotations = []    
        history, historyImages = [], []
        for i in range(len(actions) + 1):
            sample = copy.deepcopy(SAMPLET)            
            sample["stateEpisode"] = stateEpisode

            if stateEpisode:
                if metaData.get("stateAnnotations", []) and len(metaData["stateAnnotations"]) > 0:
                    frameJudge = metaData["stateAnnotations"][i]
                else:
                    frameJudge = stateFrameJudge(self.client, metaData["goal"], step_instructions[i], screenshots[i])
                    stateAnnotations.append(frameJudge)

                state_info = ""
                state_reasoning = ""
                feature = ""
                desiredState = ""
                currentState = ""

                normalized_instruction = step_instructions[i].strip().lower().rstrip('.')

                if isinstance(frameJudge, dict) and frameJudge.get("state_step", False):
                    feature = frameJudge.get("target_feature", "")
                    desiredState = frameJudge.get("desired_state", "On" if "on" in metaData['goal'].lower() else "Off")
                    currentState = frameJudge.get("current_state", "On" if "on" in step_instructions[i].lower() else "Off")
                    
                    # if desiredState not in ["On", "Off"]:
                    #     desiredState = "On" if "on" in metaData['goal'].lower() else "Off"
                    # if currentState not in ["On", "Off"]:
                    #     currentState = "Off" if "on" in step_instructions[i].lower() else "On"

                    state_info = f"According to the screenshot, the switch `{feature}` is currently `{currentState}`, while according to the goal, it should be `{desiredState}`."

                    if currentState != desiredState and normalized_instruction != "the task is completed":
                        state_reasoning = "Switch needs to be toggled."
                    elif currentState != desiredState and normalized_instruction == "the task is completed":
                        state_info = ""
                        state_reasoning = "The switch is already in the correct state. Marking task as complete."
                    elif currentState == desiredState and normalized_instruction == "the task is completed":
                        state_reasoning = "The switch is already in the correct state. Marking task as complete."
                    else:
                        state_reasoning = "No toggle needed."

                elif isinstance(frameJudge, dict) and normalized_instruction == "the task is completed":
                    state_reasoning = f"Target switch not found on this screen. Marking task as complete."
                else:
                    state_reasoning = f"Target switch not found on this screen."

                step_instruction = "set the task as completed" if normalized_instruction == "the task is completed" else normalized_instruction
                thought = " ".join(filter(None, [
                    state_info,
                    state_reasoning,
                    f"Executing: `{step_instruction}`."
                ]))
            else:
                normalized_instruction = step_instructions[i].strip().lower().rstrip('.')
                if normalized_instruction == "the task is completed":
                    thought = "Task already satisfied. Marking as complete."
                else:
                    thought = f"{normalized_instruction}."
            
            thought = re.sub(r'\.\.$', '.', thought) 
            step_instruction = step_instructions[i].strip() if i != len(actions) else "The task is completed."
            
            labelThought = thought if self.cot_trained else step_instruction
                    
            
            sample["images"].append(screenshots[i])
            sample["layouts"].append(layouts[i])
            sample["episodeID"] = str(metaData["episode_id"])
            sample["stepID"] = i + 1
            sample["width"] = widths[i]
            sample["height"] = heights[i]
            
            labelAction = actionStrs[i] if i != len(actions) else "COMPLETE"
            if self.model == "uitars":
                prompt = copy.deepcopy(ANDORIDCONTROL_UITARS_ACTION_PREDICTION_PROMPT)
                inputs = prompt.format_map({"instruction":metaData["goal"]})
                action = translateAtlas2Uitars(labelAction)
                outputs = f"Thought: {labelThought}\n Action: {action}"
            elif self.model == "agentcpmgui":
                if self.cot_trained:
                    prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_CPM_WITH_THOUGHT)
                    safe_prompt = prompt.replace("{", "{{").replace("}", "}}")
                    safe_prompt = safe_prompt.replace("{{instruction}}", "{instruction}")  # 保留要替换的变量
                    if self.args.low_level:
                        inputs = safe_prompt.format_map({"instruction":step_instructions[i]})
                    else:
                        inputs = safe_prompt.format_map({"instruction":metaData["goal"]})

                else:
                    if self.args.low_level:
                        inputs = step_instructions[i]
                    else:
                        inputs = metaData['goal']


                
                action = translateAtlas2ACG(labelAction)
                if not action:
                    continue
                acg_output = {"thought": labelThought}
                acg_output.update(action)
                outputs = json.dumps(acg_output, ensure_ascii=False, separators=(',', ':'))   
            elif self.model == "atlas":
                prompt = copy.deepcopy(ANDROIDCONTROL_ATLAS_ACTION_PREDICTION_PROMPT)
                previousActions = step_instructions[:i]
                inputs = prompt.format_map({"finalGoal":metaData['goal'], "previousActions":previousActions})
                action = labelAction
                outputs = f"Thought: {labelThought}\n Action: {action}"
                if self.args.low_level:
                    inputs = inputs.replace("__LOW_LEVEL_PLACEHOLDER__", f"Current step instruction: {step_instructions[i]}")
                else:
                    inputs = inputs.replace("__LOW_LEVEL_PLACEHOLDER__", "")
            else:
                raise NotImplementedError(f"Model {self.model} not supported.")
            
            
            sample["label"] = outputs
            if self.model == "uitars":
                if self.llamafactory:
                    sample["messages"][0]["content"] = inputs
                    sample["messages"][1]["content"] = outputs
                    sample["messages"][0]["content"] = "".join([sample["messages"][0]["content"] + "<|im_end|>"] + history[-8:])
                    if self.args.low_level:
                        sample["messages"][0]["content"] += f"\n<|im_start|>assistant\n Thought: {thought} <|im_end|>\n"
                    sample["messages"][0]["content"] += "\n<|im_start|>user\n## ScreenShot\n<image>"
                    sample["images"] = historyImages[-4:] + [screenshots[i]]
                    stepInfo = [
                        "\n<|im_start|>user\n <image> <|im_end|>\n",
                        f"\n<|im_start|>assistant\n{outputs} <|im_end|>\n"
                    ]
                    
                    history.extend(stepInfo)
                    historyImages.append(screenshots[i])
                else:
                    sample["messages"][0]["content"] = inputs
                    messages:list = sample["messages"]
                    messages.insert(0, sys_prompt)
                    messages = messages[:2]
                    messages.extend(history[-8:])
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": screenshots[i]}
                        ]
                    })
                    if self.args.low_level:
                        messages.append({
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": f"Thought: {labelThought}\n"}
                            ]
                        })
                    sample["messages"] = messages
                    stepInfo = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": screenshots[i]
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": outputs}
                            ]
                        }
                    ]
                    
                    history.extend(stepInfo)
            
            else:
                sample["messages"][0]["content"] = inputs
                sample["messages"][1]["content"] = outputs
                if not self.llamafactory:
                    sample["messages"].insert(0, sys_prompt) 
                    sample["messages"] = sample["messages"][:2]
                    sample["messages"].append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": screenshots[i]}
                        ]
                    })
            # if self.llamafactory:
            #     sample =  {
            #         key: sample[key] 
            #         for key in ["messages", "images", "label", "stateEpisode", "episodeID", "stepID", "width", "height"] 
            #         if key in sample
            #     }
            
            
            preprocessData.append(sample)
            if self.model == "agentcpmgui" and labelAction == "COMPLETE" and self.cot_trained:
                preprocessData.append(sample)
                preprocessData.append(sample)
                preprocessData.append(sample)
        metaData["stateAnnotations"] = stateAnnotations
        if metaData["stateAnnotations"] != []:
            with open(metaDataPath, "w") as f:
                json.dump(metaData, f, indent=4, ensure_ascii=False) 
        return preprocessData
    
    
    
class AITZPreprocessor(AgenticDataPreprocessor):
    def __init__(self, args):
        super().__init__(args)
        self.trainRootPath = args.aitzTrainRootPath if hasattr(args, "aitzTrainRootPath") and args.aitzTrainRootPath is not None else "./GUIData/android_in_the_zoo/train/jsons"
        self.testRootPath = args.aitzTestRootPath if hasattr(args, "aitzTestRootPath") and args.aitzTestRootPath is not None else "./GUIData/android_in_the_zoo/test/jsons"
        self.imageRootPath = args.aitzImagePath if hasattr(args, "aitzImagePath") and args.aitzImagePath is not None else "./GUIData/android_in_the_zoo/{mode}/images"
        
        self.savePathTrain = f"./data/GUIAgentic/aitz/{self.model}/aitz_{self.settings}_action_predict_train.json"
        self.savePathTest = f"./data/GUIAgentic/aitz/{self.model}/aitz_{self.settings}_action_predict_test.json"
        
        self.savePathStateTrain = f"./data/GUIAgentic/aitz/{self.model}/aitz_{self.settings}_action_predict_state_train.json"
        
        self.savePathNoneStateTrain = f"./data/GUIAgentic/aitz/{self.model}/aitz_{self.settings}_action_predict_none_state_train.json"
        
        os.makedirs(os.path.dirname(self.savePathTrain), exist_ok=True)
        
        self.trainData = [os.path.join(self.trainRootPath, jsonFile) for jsonFile in os.listdir(self.trainRootPath)]
        
        self.testData = [os.path.join(self.testRootPath, jsonFile) for jsonFile in os.listdir(self.testRootPath)]
        
    def preprocessEpisode(self, episode_identifier:Union[str, Path]):
        jsonPath = Path(episode_identifier)
        mode = "train" if "train" in jsonPath.parts else "test"
        preprocessedData = []
        newAnnotations = []
        with open(episode_identifier, "r") as f:
            data = json.load(f)
        pattern = r"(?:\b(turn|power|flip|activate|deactivate|start|shut|enable|disable|engage|disengage|cut|kill|press|push|hit)\b\s+(?:the\s+)?\b\w*\b\s*(on|off|up|down|onwards|back|engaged|disengaged)|activate|deactivate|start\s+up|shut\s+down|enable|disable|engage|disengage|cut\s+the\s+power|kill\s+the\s+power)"
        
        stateEpisode = bool(re.search(pattern, data[0]["instruction"], re.IGNORECASE))
        
        annotationPath = jsonPath.parent.parent / "annotations" / f"{jsonPath.stem}_annotation.json"
        
        if os.path.exists(annotationPath):
            with open(annotationPath, "r") as f:
                annotations = json.load(f)
        else:
            annotations = []
        
        wds = [Image.open(os.path.join(self.imageRootPath.format(mode=mode), os.path.basename(item["image_path"]))).size for item in data]
        
        if self.args.state and not stateEpisode:
            return []
        try:
            actionStr = [convertAITZAction(item, *wd) for item, wd in zip(data, wds)]
        except NotImplementedError as e:
            logger.warning(f"Convert Action Error: {e}")
            return []
        actionDescs = [item.get("coat_action_desc","") for item in data]
        actionResults = [item.get("coat_action_result","") for item in data]
        
        history, historyImages = [], []
        
        for i, item in enumerate(data):
            imagePath = os.path.join(self.imageRootPath.format(mode=mode), os.path.basename(item["image_path"]))
            image = Image.open(imagePath)
            width, height = image.size
            bboxes = eval(item["ui_positions"])
            convertBboxes = [[x / width * 1000, y / height * 1000, (x + w) / width * 1000, (y + h) / height * 1000] for y, x, h, w in bboxes]
            
            screenDesc = item.get("coat_screen_desc","")
            actionThink = item.get("coat_action_think","")
            actionDesc = actionDescs[i]
            action = actionStr[i]
            actionResult = actionResults[i]
            previousActionDescs = actionDescs[:i]
            previousActionResult = actionResults[i - 1] if i > 0 else ""
            
            
            sample = copy.deepcopy(SAMPLET)
            sample["episodeID"] = str(item["episode_id"])
            sample["stepID"] = item["step_id"]
            sample["width"], sample["height"] = width, height
            sample["images"].append(imagePath)
            sample["bboxes"] = convertBboxes
            sample["screenDesc"] = screenDesc
            sample["previousActionDescs"] = previousActionDescs
            sample["previousActionResult"] = previousActionResult
            sample["actionThink"] = actionThink
            sample["actionThink"] = actionThink
            sample["actionDesc"] = actionDesc
            sample["action"] = action
            sample["actionResult"] = actionResult
            sample["stateEpisode"] = stateEpisode
            
            if stateEpisode:    
                if annotations != [] and len(annotations) > 0:
                    frameJudge = annotations[i]
                else:
                    frameJudge = stateFrameJudge(self.client, item['instruction'], actionDesc, imagePath)
                    newAnnotations.append(frameJudge)
                state_info = ""
                state_reasoning = ""
                feature = ""
                desiredState = ""
                currentState = ""

                normalized_instruction = actionDesc.strip().lower().rstrip('.')
                
                if isinstance(frameJudge, dict) and frameJudge.get("state_step", False) and item['result_action_type'] in [4, 10]:
                    feature = frameJudge.get("target_feature", "")
                    desiredState = frameJudge.get("desired_state", "On" if "on" in item['instruction'].lower() else "Off")
                    currentState = frameJudge.get("current_state", "On" if "on" in actionDesc.lower() else "Off")
                    # if desiredState not in ["On", "Off"]:
                    #     desiredState = "On" if "on" in item['instruction'].lower() else "Off"
                    # if currentState not in ["On", "Off"]:
                    #     currentState = "Off" if "on" in actionDesc.lower() else "On"
                    
                    state_info = f"According to the screenshot, the switch `{feature}` is currently `{currentState}`, while according to the goal, it should be `{desiredState}`."
                    
                    if currentState != desiredState and normalized_instruction != "the task is completed":
                        state_reasoning = "Switch needs to be toggled."
                    # elif normalized_instruction == "the task is completed":
                    #     state_reasoning = "The switch is already in the correct state. Marking task as complete."
                    elif currentState != desiredState and normalized_instruction == "the task is completed":
                        state_info = ""
                        state_reasoning = "The switch is already in the correct state. Marking task as complete."
                    elif currentState == desiredState and normalized_instruction == "the task is completed":
                        state_reasoning = "The switch is already in the correct state. Marking task as complete."
                    else:
                        state_reasoning = "No toggle needed."
                elif isinstance(frameJudge, dict) and normalized_instruction == "the task is completed":
                    state_reasoning = f"Target switch not found on this screen. Marking task as complete."
                else:
                    state_reasoning = f"Target switch not found on this screen."
                
                step_instruction = "set the task as completed" if normalized_instruction == "the task is completed" else normalized_instruction
                
                thought = " ".join(filter(None, [
                    state_info,
                    state_reasoning,
                    f"Executing: `{step_instruction}`."
                ]))
            
            else:
                normalized_instruction = actionDesc.strip().lower().rstrip('.')
                if normalized_instruction == "the task is completed":
                    thought = "Task already satisfied. Marking as complete."
                    
                else:
                    thought = f"{normalized_instruction}."
            thought = re.sub(r'\.\.$', '.', thought)
            labelAction = action
            
            step_instruction = actionDesc.strip()
            
            labelThought = thought if self.args.cot_trained else step_instruction 
            
            if self.model == "uitars":
                prompt = copy.deepcopy(AITZ_UITARS_ACTION_PREDICTION_PROMPT)
                inputs = prompt.format_map({"instruction":item["instruction"]})
                # inputs = inputs + f"\n##Action History\n{previousActionDescs}"
                try:
                    uitarsAction = translateAtlas2Uitars(labelAction)
                except NotImplementedError as e:
                    continue
                outputs = f"Thought: {labelThought}\n Action: {uitarsAction}"
            elif self.model == "agentcpmgui":
                if self.cot_trained:
                    prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_CPM_WITH_THOUGHT)
                    safe_prompt = prompt.replace("{", "{{").replace("}", "}}")
                    safe_prompt = safe_prompt.replace("{{instruction}}", "{instruction}")  # 保留要替换的变量
                    inputs = safe_prompt.format_map({"instruction":item["instruction"]})
                else:
                    inputs = item["instruction"]
                action = translateAtlas2ACG1(labelAction)
                if not action:
                    continue
                acg_output = {"thought": labelThought}
                acg_output.update(action)
                outputs = json.dumps(acg_output, ensure_ascii=False, separators=(',', ':')) 
            elif self.model == "atlas":
                prompt = copy.deepcopy(AITZ_ATLAS_ACTION_PREDICTION_PROMPT)
                inputs = prompt.format_map({"finalGoal":item["instruction"], "PAD":previousActionDescs})
                if self.args.low_level:
                    inputs = inputs.replace("__LOW_LEVEL_PLACEHOLDER__", f"Current step instruction: {actionDesc}")
                else:
                    inputs = inputs.replace("__LOW_LEVEL_PLACEHOLDER__", "")
                
                outputs = f"Thought: {labelThought}\n Action: {labelAction}"
                
            else:
                raise NotImplementedError(f"Model {self.model} not supported.")
            
            sample["label"] = outputs
            if self.model == "uitars":
                if self.llamafactory:
                    sample["messages"][0]["content"] = inputs
                    sample["messages"][1]["content"] = outputs
                    sample["messages"][0]["content"] = "".join([sample["messages"][0]["content"] + "<|im_end|>"] + history[-8:])
                    if self.args.low_level:
                        sample["messages"][0]["content"] += f"\n<|im_start|>assistant\n Thought: {thought} <|im_end|>\n"
                    sample["messages"][0]["content"] += "\n<|im_start|>user\n## ScreenShot\n<image>"
                    sample["images"] = historyImages[-4:] + [imagePath]
                    stepInfo = [
                        "\n<|im_start|>user\n <image> <|im_end|>\n",
                        f"\n<|im_start|>assistant\n{outputs} <|im_end|>\n"
                    ]
                    
                    history.extend(stepInfo)
                    historyImages.append(imagePath)
                else:
                    sample["messages"][0]["content"] = inputs
                    messages:list = sample["messages"]
                    messages.insert(0, sys_prompt)
                    messages = messages[:2]
                    
                    messages.extend(history[-8:])
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": imagePath}
                        ]
                    })
                    if self.args.low_level:
                        messages.append({
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": f"Thought: {labelThought}\n"}
                            ]
                        })
                    sample["messages"] = messages
                    stepInfo = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": imagePath
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": outputs}
                            ]
                        }
                    ]
                    
                    history.extend(stepInfo)
            
            else:
                sample["messages"][0]["content"] = inputs
                sample["messages"][1]["content"] = outputs
                if not self.llamafactory:
                    sample["messages"].insert(0, sys_prompt) 
                    sample["messages"] = sample["messages"][:2]
                    sample["messages"].append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": imagePath}
                        ]
                    })
            # if self.llamafactory:
            #     sample =  {
            #         key: sample[key] 
            #         for key in ["messages", "images", "label", "stateEpisode", "episodeID", "stepID", "width", "height"] 
            #         if key in sample
            #     }
            preprocessedData.append(sample)
            if self.model == "agentcpmgui" and self.cot_trained:
                preprocessedData.append(sample)
                preprocessedData.append(sample)
                preprocessedData.append(sample)
                preprocessedData.append(sample)
            # if self.model == "agentcpmgui" and labelAction == "COMPLETE" and self.cot_trained:
            #     preprocessedData.append(sample)
            #     preprocessedData.append(sample)  
            #     preprocessedData.append(sample)
        if newAnnotations != []:
            with open(annotationPath, "w") as f:
                json.dump(newAnnotations, f, indent=4, ensure_ascii=False)  
        return preprocessedData
            
    
        
class GUIOdysseyPreprocessor(AgenticDataPreprocessor):
    def __init__(self, args):
        super().__init__(args)
        with open("./GUIData/GUIOdyssey/splits/random_split.json", "r") as f:
            self.split = json.load(f)
        self.jsonPath = args.gui_odyssey_json_path if hasattr(args, "gui_odyssey_json_path") else "./GUIData/GUIOdyssey/annotations"
        self.imagePath = args.gui_odyssey_image_path if hasattr(args, "gui_odyssey_image_path") else "./GUIData/GUIOdyssey/screenshots/screenshots"
        self.annotationPath = args.gui_odyssey_annotation_path if hasattr(args, "gui_odyssey_annotation_path") else "./GUIData/GUIOdyssey/state_annotations"
        
        self.savePathTrain = f"./data/GUIAgentic/gui_odyssey/{self.model}/gui_odyssey_{self.settings}_action_predict_train.json"
        self.savePathTest = f"./data/GUIAgentic/gui_odyssey/{self.model}/gui_odyssey_{self.settings}_action_predict_test.json"
        
        self.savePathStateTrain = f"./data/GUIAgentic/gui_odyssey/{self.model}/gui_odyssey_{self.settings}_action_predict_state_train.json"
        
        self.savePathNoneStateTrain = f"./data/GUIAgentic/gui_odyssey/{self.model}/gui_odyssey_{self.settings}_action_predict_none_state_train.json"
        
        os.makedirs(os.path.dirname(self.savePathTrain), exist_ok=True)
        
        self.trainData = [os.path.join(self.jsonPath, episodeJsonPath) for episodeJsonPath in self.split["train"]]
        
        self.testData = [os.path.join(self.jsonPath, episodeJsonPath) for episodeJsonPath in self.split["test"]]
        

    def preprocessEpisode(self, episode_identifier:Union[str, Path]):
        preprocessedData = []
        metaDataPath = Path(episode_identifier)
        with open(metaDataPath, "r") as f:
            metaData = json.load(f)
        
        goal = metaData["task_info"]["instruction"]

        pattern = r"(?:\b(turn|power|flip|activate|deactivate|shut|enable|disable|engage|disengage)\b\s+(?:the\s+)?\b\w*\b\s*(on|off|up|down|onwards|back|engaged|disengaged)\b|activate|deactivate|shut\s+down|enable|disable|engage|disengage|cut\s+the\s+power|kill\s+the\s+power)"
        
        stateEpisode = bool(re.search(pattern, goal, re.IGNORECASE))
        if self.args.state and not bool(re.search(pattern, goal, re.IGNORECASE)):
            return []
        annotationPath = os.path.join(self.annotationPath, f"{metaData['episode_id']}.json")
        
        if os.path.exists(annotationPath):
            with open(annotationPath, "r") as f:
                annotations = json.load(f)
        else:
            annotations = []
        
        newAnnotations = []
        history, historyImages = [], []
        
        actionHistory = []
        
        for stepID, step in enumerate(metaData["steps"]):
            if metaData['episode_id'] == "1712458483298839" and stepID == 8:
                pass
            imagePath = os.path.join(self.imagePath, step["screenshot"])
            width, height = metaData["device_info"]["h"], metaData["device_info"]["w"]

            sample = copy.deepcopy(SAMPLET)
            sample["episodeID"] = str(metaData['episode_id'])
            sample["stepID"] = stepID
            sample["images"].append(imagePath)
            sample["width"], sample["height"] = width, height
            sample["stateEpisode"] = stateEpisode
            sample["bbox"] = step["sam2_bbox"]
            
            low_level_instruction = step['low_level_instruction']
            
            if stateEpisode:    
                if annotations != [] and len(annotations) > 0:
                    frameJudge = annotations[stepID]
                else:
                    frameJudge = stateFrameJudge(self.client, goal, low_level_instruction, imagePath)
                    newAnnotations.append(frameJudge)
                state_info = ""
                state_reasoning = ""
                feature = ""
                desiredState = ""
                currentState = ""

                normalized_instruction = low_level_instruction.strip().lower().rstrip('.')
                
                if isinstance(frameJudge, dict) and frameJudge.get("state_step", False) and (step["action"] in ["COMPLETE"] or (step["action"] == "CLICK" and step["info"] not in ["KEY_HOME", "KEY_BACK", "KEY_APPSELECT"])):
                    feature = frameJudge.get("target_feature", "")
                    desiredState = frameJudge.get("desired_state", "On" if "on" in goal.lower() else "Off")
                    currentState = frameJudge.get("current_state", "Off" if "on" in low_level_instruction.lower() else "On")
                    # if desiredState not in ["On", "Off"]:
                    #     desiredState = "On" if "on" in goal.lower() else "Off"
                    # if currentState not in ["On", "Off"]:
                    #     currentState = "Off" if "on" in low_level_instruction.lower() else "On"
                    
                    state_info = f"According to the screenshot, the switch `{feature}` is currently `{currentState}`, while according to the goal, it should be `{desiredState}`."
                    
                    if currentState != desiredState and normalized_instruction != "task completed":
                        state_reasoning = "Switch needs to be toggled."
                    # elif normalized_instruction == "task completed":
                    #     state_reasoning = "The switch is already in the correct state. Marking task as complete."
                    elif currentState != desiredState and normalized_instruction == "the task is completed":
                        state_info = ""
                        state_reasoning = "The switch is already in the correct state. Marking task as complete."
                    elif currentState == desiredState and normalized_instruction == "the task is completed":
                        state_reasoning = "The switch is already in the correct state. Marking task as complete."
                    else:
                        state_reasoning = "No toggle needed."
                elif isinstance(frameJudge, dict) and normalized_instruction == "task completed":
                    state_reasoning = f"Target switch not found on this screen. Marking task as complete."
                else:
                    state_reasoning = f"Target switch not found on this screen."
                
                step_instruction = "set the task as completed" if normalized_instruction == "task completed" else normalized_instruction
                thought = " ".join(filter(None, [
                    state_info,
                    state_reasoning,
                    f"Executing: `{step_instruction}`."
                ]))
                
            else:
                normalized_instruction = low_level_instruction.strip().lower().rstrip('.')
                if normalized_instruction == "task completed":
                    thought = "Task already satisfied. Marking as complete."
                else:
                    thought = f"{normalized_instruction}."
            thought = re.sub(r'\.\.$', '.', thought)
            
            try:
                action = convertGuiOdysseyAction(step)
            except NotImplementedError as e:
                # print(e)
                continue
            labelAction = action
            
            step_instruction = low_level_instruction.strip()
            
            labelThought = thought if self.args.cot_trained else step_instruction
            
            
            try:
                labelAction = convertGuiOdysseyAction(step)
            except NotImplementedError as e:
                # print(e)
                continue
            if self.model == "uitars":
                prompt = copy.deepcopy(GUIODYSSEY_UITARS_ACTION_PREDICTION_PROMPT)
                inputs = prompt.format_map({"instruction":goal})
                try:
                    uitarsAction = translateAtlas2Uitars(labelAction)
                except NotImplementedError as e:
                    continue
                outputs = f"Thought: {labelThought}\n Action: {uitarsAction}"
            elif self.model == "agentcpmgui":
                if self.cot_trained:
                    prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_CPM_WITH_THOUGHT)
                    safe_prompt = prompt.replace("{", "{{").replace("}", "}}")
                    safe_prompt = safe_prompt.replace("{{instruction}}", "{instruction}")  
                    inputs = safe_prompt.format_map({"instruction":goal})
                else:
                    inputs = goal
                action = translateAtlas2ACG1(labelAction)
                if not action:
                    continue
                acg_output = {"thought": labelThought}
                acg_output.update(action)
                outputs = json.dumps(acg_output, ensure_ascii=False, separators=(',', ':')) 
            elif self.model == "atlas":
                prompt = copy.deepcopy(GUIODYSSEY_ATLAS_ACTION_PREDICTION_PROMPT)
                inputs = prompt.format_map({"finalGoal":goal, "previousActions":actionHistory})
                actionHistory.append(low_level_instruction)
                if self.args.low_level:
                    inputs = inputs.replace("__LOW_LEVEL_PLACEHOLDER__", f"Current step instruction: {low_level_instruction}")
                else:
                    inputs = inputs.replace("__LOW_LEVEL_PLACEHOLDER__", "")
                outputs = f"Thought: {labelThought}\n Action: {labelAction}"
            
            
            sample["label"] = outputs
            if self.model == "uitars":
                if self.llamafactory:
                    sample["messages"][0]["content"] = inputs
                    sample["messages"][1]["content"] = outputs
                    sample["messages"][0]["content"] = "".join([sample["messages"][0]["content"] + "<|im_end|>"] + history[-8:])
                    if self.args.low_level:
                        sample["messages"][0]["content"] += f"\n<|im_start|>assistant\n Thought: {thought} <|im_end|>\n"
                    sample["messages"][0]["content"] += "\n<|im_start|>user\n## ScreenShot\n<image>"
                    sample["images"] = historyImages[-4:] + [imagePath]
                    stepInfo = [
                        "\n<|im_start|>user\n <image> <|im_end|>\n",
                        f"\n<|im_start|>assistant\n{outputs} <|im_end|>\n"
                    ]
                    
                    history.extend(stepInfo)
                    historyImages.append(imagePath)
                else:
                    sample["messages"][0]["content"] = inputs
                    messages:list = sample["messages"]
                    messages.insert(0, sys_prompt)
                    messages = messages[:2]
                    messages.extend(history[-8:])
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": imagePath}
                        ]
                    })
                    if self.args.low_level:
                        messages.append({
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": f"Thought: {labelThought}\n"}
                            ]
                        })
                    sample["messages"] = messages
                    stepInfo = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": imagePath
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": outputs}
                            ]
                        }
                    ]
                    
                    history.extend(stepInfo)
            
            else:
                sample["messages"][0]["content"] = inputs
                sample["messages"][1]["content"] = outputs
                if not self.llamafactory:
                    sample["messages"].insert(0, sys_prompt) 
                    sample["messages"] = sample["messages"][:2]
                    sample["messages"].append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": imagePath}
                        ]
                    })
            # if self.llamafactory:
            #     sample =  {
            #         key: sample[key] 
            #         for key in ["messages", "images", "label", "stateEpisode", "episodeID", "stepID", "width", "height"] 
            #         if key in sample
            #     }
            preprocessedData.append(sample)
            if self.model == "agentcpmgui" and self.cot_trained:
                preprocessedData.append(sample)
            if self.model == "agentcpmgui" and labelAction == "COMPLETE" and self.cot_trained:
                preprocessedData.append(sample)
                preprocessedData.append(sample)            
        if newAnnotations != []:
            with open(annotationPath, "w") as f:
                json.dump(newAnnotations, f, indent=4, ensure_ascii=False)  
                
        return preprocessedData
    
    
class StateDataPreprocessor:
    def __init__(self, args):
        self.args = args
        assert args.model in ["uitars", "agentcpmgui", "atlas", "guir1"], f"{args.model} is not supported!"
        
        self.model = args.model
        self.diversity = bool(args.diversity) if hasattr(args, "diversity") else True
        
        self.apiKey = args.apiKey if hasattr(args, "apiKey") else "Your API Key"
        
        self.client = ZhipuAI(api_key=self.apiKey)
        self.agentCount = args.agentCount if hasattr(args, "agentCount") else 10
        self.llamafactory = args.llamafactory if hasattr(args, "llamafactory") else False
        self.settings = "_".join(["multi_turn" if args.model == "uitars" else "", "diversity" if self.diversity else "", "llamafactory" if self.llamafactory else ""]).strip("_")
        self.trainData, self.testData = [], []
        
        self.savePathTrain, self.savePathTest = None, None
        
        self.stateJsonPathTrain = args.stateJsonPathTrain if (hasattr(args, "stateJsonPathTrain") and args.stateJsonPathTrain is not None) else "./data/state/state_control_benchmark_test.json"
        
        self.stateJsonPathTest = args.stateJsonPathTest if (hasattr(args, "stateJsonPathTest") and args.stateJsonPathTest is not None) else "./data/state/state_control_benchmark_train.json"
        
        with open(self.stateJsonPathTrain, "r") as f:
            self.trainData = json.load(f)
            
        with open(self.stateJsonPathTest, "r") as f:
            self.testData = json.load(f)

    def preprocessRecord(self, record):
        raise NotImplementedError("Please implement this method in the subclass")
    
    def preprocess(self):
        trainProcessedData, testProcessedData = [], []
        
        for record in tqdm(self.trainData, total=len(self.trainData), desc="Preprocessing training records"):
            trainProcessedData.extend(self.preprocessRecord(record))
        
        for record in tqdm(self.testData, total=len(self.testData), desc="Preprocessing test records"):
            testProcessedData.extend(self.preprocessRecord(record))
        
        return trainProcessedData, testProcessedData, None, None
    
    def preprocessMP(self):
        return self.preprocess()
    
    def saveData(self, trainProcessedData=None, testProcessedData=None, stateTrainProcessedData=None, noneStateTrainProcessedData=None):
        if trainProcessedData is not None and len(trainProcessedData) > 0:
            with open(self.savePathTrain, "w") as f:
                print(f"Save {len(trainProcessedData)} records to {self.savePathTrain}")
                json.dump(trainProcessedData, f, indent=4, ensure_ascii=False)

        if testProcessedData is not None and len(testProcessedData) > 0:
            with open(self.savePathTest, "w") as f:
                print(f"Save {len(testProcessedData)} records to {self.savePathTest}")
                json.dump(testProcessedData, f, indent=4, ensure_ascii=False)
            
    
    
    
class StateCoTDataPreprocessor(StateDataPreprocessor):
    def __init__(self, args):
        super().__init__(args)
        assert args.model in ["uitars", "agentcpmgui", "atlas", "guir1"], f"{args.model} is not supported!"
        
        self.model = args.model
        
        self.savePathTrain = args.savePathTrain if hasattr(args, "savePathTrain") else f"./data/GUIState/{self.model}/state_action_prediction_state_cot_{self.settings}_train.json".replace("__", "_")
        self.savePathTest = args.savePathTest if hasattr(args, "savePathTest") else f"./data/GUIState/{self.model}/state_action_prediction_state_cot_{self.settings}_test.json".replace("__", "_")
        
        os.makedirs(os.path.dirname(self.savePathTrain), exist_ok=True)
        
    def preprocessRecord(self, record):
        processedRecords = []
        verbs = [
            ("turn on", "turn off"),
            ("enable", "disable"),
            ("activate", "deactivate")
        ]
        
        weights = [0.5, 0.3, 0.2]
        posRecord, negRecord = copy.deepcopy(record), copy.deepcopy(record)
        posRecord["episodeID"], negRecord["episodeID"] = Path(record["imgPath"]).stem, Path(record["imgPath"]).stem
        posRecord["stepID"], negRecord["stepID"] = 1, 1
        posRecord["images"], negRecord["images"] = [record["imgPath"]], [record["imgPath"]]
        
        posMessage, negMessage = copy.deepcopy(MESSAGET), copy.deepcopy(MESSAGET)
        
        mappings = {"Enabled": "On", "Disabled":"Off"}
        
        feature = record['annotation']['feature']
        before_state = mappings[record['annotation']['state_before_action']]
        after_state = mappings[record['annotation']['state_after_action']]
        
        posInstruction = record["posInstruction"]
        
        if self.diversity:
            verbGroup = random.choices(verbs, weights=weights, k=1)[0]
            idx = 0 if record['annotation']['state_before_action'] == "Disabled" and record['annotation']['state_after_action'] == "Enabled" else 1
            
            verb = verbGroup[idx]
            
            posInstruction = f"{verb} {feature}"
        
        if self.model == "uitars":
            posAction = translateAtlas2Uitars(record["posAction"])
        elif self.model == "agentcpmgui":
            posAction = translateAtlas2ACG(record["posAction"])
        elif self.model == "atlas":
            posAction = record["posAction"]
        elif self.model == "guir1":
            posAction = translateAtlas2GUIR1(record["posAction"], record["image_width"], record["image_height"])
        else:
            raise NotImplementedError(f"model {self.mdoel} is not supported!")     
        
        posThought = (
            f"According to the screenshot, the switch `{feature}` is currently `{before_state}`, while according to the goal, it should be `{after_state}`. "
            f"Switch needs to be toggled. Executing: `click on the {record['annotation']['feature']}`."
        )
        
        if self.model == "uitars":
            prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_UITARS_WITH_THOUGHT)
            if self.llamafactory:
                posMessage[0]["content"] = prompt.format_map({"instruction":posInstruction}) + "<|im_end|>\n<|im_start|>user\n## ScreenShot\n<image>"
                posMessage[1]["content"] = f"Thought: {posThought}\n Action: {posAction}"
            else:
                posMessage[0]["content"] = prompt.format_map({"instruction":posInstruction})
                posMessage[1] = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": record["imgPath"]
                        }
                    ]
                }
                posRecord["label"] = f"Thought: {posThought}\n Action: {posAction}"
        elif self.model == "agentcpmgui":
            if self.llamafactory:
                prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_CPM_WITH_THOUGHT)
                prompt = prompt.replace("{", "{{").replace("}", "}}")
                prompt = prompt.replace("{{instruction}}", "{instruction}") 
                posMessage[0]["content"] = prompt.format_map({"instruction":posInstruction})
            
                posOutput = json.dumps({"thought": posThought, **posAction}, ensure_ascii=False, separators=(',', ':'))
            
                posMessage[1]["content"] = posOutput
            else:
                posMessage[0]["content"] = posInstruction
                posMessage[1] = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": record["imgPath"]
                        }
                    ]
                } 
                posRecord["label"] = json.dumps({"thought": posThought, **posAction}, ensure_ascii=False, separators=(',', ':'))               
        elif self.model == "atlas":
            prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_ATLAS)
            posMessage[0]["content"] = prompt.format_map({"finalGoal":posInstruction, "history":"null"})
            posMessage[1]["content"] = f"Thought: {posThought}\nAction: {posAction}"
            if not self.llamafactory:
                posMessage.insert(0, sys_prompt)
                posMessage = posMessage[:2]
                posMessage.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": record["imgPath"]
                        }
                    ]
                })
                posRecord["label"] = f"Thought: {posThought}\nAction: {posAction}"
        elif self.model == "guir1":
            prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_GUIR1_WITH_THOUGHT)
            posMessage[0]["content"] = prompt.replace("__INSTRUCTION__", posInstruction)
            posMessage[1]["content"] = f"<think>{posThought}</think> <answer>{posAction}</answer>"
            if not self.llamafactory:
                posMessage.insert(0, sys_prompt)
                posMessage = posMessage[:2]
                posMessage.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": record["imgPath"]
                        }
                    ]
                })
                posRecord["label"] = f"<think>{posThought}</think> <answer>{posAction}</answer>"
            
        else:
            raise NotImplementedError(f"model {self.mdoel} is not supported!")        
        
        
        
        negInstruction = record["negInstruction"]
        
        if self.diversity:
            verbGroup = random.choices(verbs, weights=weights, k=1)[0]
            idx = 1 if record['annotation']['state_before_action'] == "Disabled" and record['annotation']['state_after_action'] == "Enabled" else 0
            
            verb = verbGroup[idx]
            
            negInstruction = f"{verb} {feature}"
            
        negThought = (
            f"According to the screenshot, the switch `{feature}` is currently `{before_state}`, while according to the goal, it should be `{before_state}`. "
            f"No toggle needed. Executing: `set the task as completed`."
        ) 
            
        
            
        if self.model == "uitars":
            negAction = translateAtlas2Uitars(record["negAction"])
            prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_UITARS_WITH_THOUGHT)
            if self.llamafactory:
                negMessage[0]["content"] = prompt.format_map({"instruction":negInstruction}) + "<|im_end|>\n<|im_start|>user\n## ScreenShot\n<image>"
                negMessage[1]["content"] = f"Thought: {negThought}\n Action: {negAction}"
            else:
                negMessage[0]["content"] = prompt.format_map({"instruction":negInstruction})
                negMessage[1] = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": record["imgPath"]
                        }
                    ]
                }
                negRecord["label"] = f"Thought: {negThought}\n Action: {negAction}"
        elif self.model == "agentcpmgui":
            negAction = translateAtlas2ACG(record["negAction"])
            if self.llamafactory:
                prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_CPM_WITH_THOUGHT)
                prompt = prompt.replace("{", "{{").replace("}", "}}")
                prompt = prompt.replace("{{instruction}}", "{instruction}")
                negMessage[0]["content"] = prompt.format_map({"instruction":negInstruction})
            
                negOutput = json.dumps({"thought": negThought, **negAction}, ensure_ascii=False, separators=(',', ':'))
            
                negMessage[1]["content"] = negOutput
            else:
                negMessage[0]["content"] = negInstruction
                negMessage[1] = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": record["imgPath"]
                        }
                    ]
                } 
                negRecord["label"] = json.dumps({"thought": negThought, **negAction}, ensure_ascii=False, separators=(',', ':'))                
        elif self.model == "atlas":
            negAction = record["negAction"]
            prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_ATLAS)
            negMessage[0]["content"] = prompt.format_map({"finalGoal":negInstruction, "history":"null"})
            negMessage[1]["content"] = f"Thought: {negThought}\nAction: {negAction}"
            if not self.llamafactory:
                negMessage.insert(0, sys_prompt)
                negMessage = negMessage[:2]
                negMessage.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": record["imgPath"]
                        }
                    ]
                })
                negRecord["label"] = f"Thought: {negThought}\nAction: {negAction}"
        elif self.model == "guir1":
            negAction = translateAtlas2GUIR1(record["negAction"], record["image_width"], record["image_height"])
            prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_GUIR1_WITH_THOUGHT)
            negMessage[0]["content"] = prompt.replace("__INSTRUCTION__", negInstruction)
            negMessage[1]["content"] = f"<think>{negThought}</think> <answer>{negAction}</answer>"
            if not self.llamafactory:
                negMessage.insert(0, sys_prompt)
                negMessage = negMessage[:2]
                negMessage.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": record["imgPath"]
                        }
                    ]
                })
                negRecord["label"] = f"<think>{negThought}</think> <answer>{negAction}</answer>"
                
        else:
            raise NotImplementedError(f"model {self.mdoel} is not supported!")
        
            
        posRecord["messages"] = posMessage
        negRecord["messages"] = negMessage
        
        posRecord["positive"], negRecord["positive"] = True, False
        
        # if self.model != "agentcpmgui":
        processedRecords.extend([posRecord, negRecord])
        # else:
        #     if random.random() >= 0.3:
        #         processedRecords.append(posRecord)
        #     processedRecords.append(negRecord)
        
        return processedRecords
        
        
class StateCoTForAtlasDataPreprocessor(StateDataPreprocessor):
    def __init__(self, args):
        super().__init__(args)
        assert args.model in ["atlas"], f"{args.model} is not supported!"
        
        self.model = args.model
        
        self.savePathTrain = args.savePathTrain if hasattr(args, "savePathTrain") else f"./data/GUIState/{self.model}/state_action_prediction_state_interpretation_{self.settings}_train.json".replace("__", "_")
        self.savePathTest = args.savePathTest if hasattr(args, "savePathTest") else f"./data/GUIState/{self.model}/state_action_prediction_state_interpretation_{self.settings}_test.json".replace("__", "_")
        
        os.makedirs(os.path.dirname(self.savePathTrain), exist_ok=True)
        
    def preprocessRecord(self, record):
        processedRecords = []
        verbs = [
            ("turn on", "turn off"),
            ("enable", "disable"),
            ("activate", "deactivate")
        ]
        
        weights = [0.5, 0.3, 0.2]
        posRecord, negRecord = copy.deepcopy(record), copy.deepcopy(record)
        posRecord["episodeID"], negRecord["episodeID"] = Path(record["imgPath"]).stem, Path(record["imgPath"]).stem
        posRecord["stepID"], negRecord["stepID"] = 1, 1
        posRecord["images"], negRecord["images"] = [record["imgPath"]], [record["imgPath"]]
        
        posMessage, negMessage = copy.deepcopy(MESSAGET), copy.deepcopy(MESSAGET)
        
        mappings = {"Enabled": "On", "Disabled":"Off"}
        
        feature = record['annotation']['feature']
        before_state = mappings[record['annotation']['state_before_action']]
        after_state = mappings[record['annotation']['state_after_action']]
        
        posInstruction = record["posInstruction"]
        
        if self.diversity:
            verbGroup = random.choices(verbs, weights=weights, k=1)[0]
            idx = 0 if record['annotation']['state_before_action'] == "Disabled" and record['annotation']['state_after_action'] == "Enabled" else 1
            
            verb = verbGroup[idx]
            
            posInstruction = f"{verb} {feature}"
        
        if self.model == "uitars":
            posAction = translateAtlas2Uitars(record["posAction"])
        elif self.model == "agentcpmgui":
            posAction = translateAtlas2ACG(record["posAction"])
        elif self.model == "atlas":
            posAction = record["posAction"]
        
        posThought = (
            f"According to the screenshot, the switch `{feature}` is currently `{before_state}`, while according to the goal, it should be `{after_state}`. "
            f"Switch needs to be toggled. Executing: `click on the {record['annotation']['feature']}`."
        )
        
        prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_ATLAS)
        posMessage[0]["content"] = prompt.format_map({"finalGoal":posInstruction, "history":"null"})
        posMessage[1]["content"] = f"Thought: {posThought}\n Action: {posAction}"
        if not self.llamafactory:
            posMessage.insert(0, sys_prompt)
            posMessage = posMessage[:2]
            posRecord["label"] = f"Thought: {posThought}\n Action: {posAction}"      
        
        
        
        negInstruction = record["negInstruction"]
        
        if self.diversity:
            verbGroup = random.choices(verbs, weights=weights, k=1)[0]
            idx = 1 if record['annotation']['state_before_action'] == "Disabled" and record['annotation']['state_after_action'] == "Enabled" else 0
            
            verb = verbGroup[idx]
            
            negInstruction = f"{verb} {feature}"
            
        negThought = (
            f"According to the screenshot, the switch `{feature}` is currently `{before_state}`, while according to the goal, it should be `{before_state}`. "
            f"No toggle needed. Executing: `set the task as completed`."
        ) 
            
        
            
        negAction = record["negAction"]
        prompt = copy.deepcopy(STATE_ACTION_PREDICT_PROMPT_ATLAS)
        negMessage[0]["content"] = prompt.format_map({"finalGoal":negInstruction, "history":"null"})
        negMessage[1]["content"] = f"Thought: {negThought}\n Action: {negAction}"      
        if not self.llamafactory:
            negMessage.insert(0, sys_prompt)
            negMessage = negMessage[:2]
            negRecord["label"] = f"Thought: {negThought}\n Action: {negAction}"      
                
        
            
        posRecord["messages"] = posMessage
        negRecord["messages"] = negMessage
        
        posRecord["positive"], negRecord["positive"] = True, False
        posRecord["episodeID"], negRecord["episodeID"] = Path(record["imgPath"]).stem, Path(record["imgPath"]).stem
        posRecord["stepID"], negRecord["stepID"] = 1, 1
        
        processedRecords.extend([posRecord, negRecord])
        
        return processedRecords
    
    
    
mappings = {
    "android_control":AndroidControlPreprocessor,
    "aitz":AITZPreprocessor,
    "gui_odyssey":GUIOdysseyPreprocessor,
    "state_cot":StateCoTDataPreprocessor,
    "state_cot_for_atlas":StateCoTForAtlasDataPreprocessor
}


def load_processor(config:Namespace) -> Union[AndroidControlPreprocessor, AITZPreprocessor, GUIOdysseyPreprocessor, StateCoTDataPreprocessor]:
    return mappings[config.type](config)
    
    
class DataMerger:
    def __init__(self, args):
        self.args = args
        self.mergeConfigList = args.mergeConfigList
        self.model = args.model
        self.dataConfigs = []
        for mergeConfigPath in self.mergeConfigList:
            with open(mergeConfigPath, "r") as f:
                config = yaml.safe_load(f)
            self.dataConfigs.append(config)
            
        self.dataTypes = [config["type"] for config in self.dataConfigs]
        
        self.levels = ["low" if config.get("low_level", False) else "high" for config in self.dataConfigs]
        
        self.settings = [f"{config['type']}_{level}" for config, level in zip(self.dataConfigs, self.levels)]
        self.settings = "+".join(self.settings)
                
        self.savePathTrain = f"./data/merge/{self.model}/{self.settings}_{self.args.type}_train.json".replace("__", "_").replace("++", "+")
        
        self.dataConfigs = [Namespace(**config) for config in self.dataConfigs]

        
        

        os.makedirs(os.path.dirname(self.savePathTrain), exist_ok=True)
        

    def mergeTrain(self):
        trainDataDict = {}
        for config in self.dataConfigs:
            logger.info(f"Processing {config.type}...")
            processor = load_processor(config)
            trainProcessedData, _, _, _ = processor.preprocessMP()
            
            trainDataDict[f"{config.type}_{'low' if hasattr(config, 'low_level') and config.low_level else 'high'}"] = trainProcessedData

        # merge data
        
        mergeData = []
        for split, data in trainDataDict.items():
            mergeData.extend(data)
        return mergeData
        
    def save(self, mergeData):
        with open(self.savePathTrain, "w") as f:
            print(f"Save {len(mergeData)} records to {self.savePathTrain}")
            json.dump(mergeData, f, indent=4, ensure_ascii=False)


class HalfHighLowDataMerger(DataMerger):
    def __init__(self, args):
        super().__init__(args)
        
        self.dataTypes = [config["type"] for config in self.dataConfigs]
        
        self.levels = ["low" if config["low_level"] else "high" for config in self.dataConfigs]
        
        self.settings = [f"{config['type']}_{level}" for config, level in zip(self.dataConfigs, self.levels)]
        self.settings = "+".join(self.settings)
                
        self.savePathTrain = f"./data/merge/{self.model}/{self.settings}_{self.args.type}_train.json".replace("__", "_").replace("++", "+")

        os.makedirs(os.path.dirname(self.savePathTrain), exist_ok=True)
    
    def mergeTrain(self):
        trainDataDict = {}
        for config in self.dataConfigs:
            logger.info(f"Processing {config.type}...")
            processor = load_processor(config)
            trainProcessedData, _, _, _ = processor.preprocessMP()
            
            trainDataDict[config.type] = trainProcessedData

        # merge data
        
        mergeData = []
        for split, data in trainDataDict.items():
            mergeData.extend(data)
        return mergeData
        
    def save(self, mergeData):
        with open(self.savePathTrain, "w") as f:
            print(f"Save {len(mergeData)} records to {self.savePathTrain}")
            json.dump(mergeData, f, indent=4, ensure_ascii=False)

    
    
    
