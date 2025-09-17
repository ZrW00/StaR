from .utils import *
from .actions import *

import math, time, json, copy
from tqdm import tqdm
from typing import *

from operator import itemgetter
from itertools import groupby
import traceback
import torch.multiprocessing as mp
from pathlib import Path

from argparse import Namespace
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

DEBUG =  False
DEBUGCNT = 10

class AgenticEvaluator:
    def __init__(self, config):
        self.config = config
        self.recordJsonPath = None if not hasattr(config, 'recordJsonPath') else config.recordJsonPath
        assert config.testJsonPath is not None, f"Should provide testJsonPath"
        
        
        self.recordSavePath = "defalut.json" if not hasattr(config, 'recordSavePath') else config.recordSavePath
        self.testJsonPath = config.testJsonPath
        self.cache_dir = "~/.cache"
        
        self.modelPath = None if not hasattr(config, 'modelPath') else config.modelPath
        
        deviceIDs = "[0,0]" if not hasattr(config, 'devicesIDs') else config.devicesIDs
        self.agentCount = 1 if not hasattr(config, 'agentCount') else config.agentCount
        
        try:
            deviceIDs = eval(deviceIDs)
        except TypeError as e:
            logger.warning(f"deviceIDs: {deviceIDs}, type: {type(deviceIDs)}")
            
        agentDeviceCount = math.ceil(len(deviceIDs) / self.agentCount)
    
        self.deviceIDs = [deviceIDs[i * agentDeviceCount: (i + 1) * agentDeviceCount] for i in range(self.agentCount)]
        
        self.agentType = "uitars" if not hasattr(config, 'agentType') else config.agentType
        
        assert self.agentType in ["uitars", "atlas", "agentcpmgui", "gemini", "gpt"], f"agentType should be one of ['uitars', 'atlas', 'agentcpmgui', 'gemini', 'gpt']"

        
        self.max_new_tokens = 512 if not hasattr(config, 'max_new_tokens') else config.max_new_tokens
        
        
        
        self.benchmarkSetting = "high" if not hasattr(config, "benchmarkSetting") else config.benchmarkSetting
        
        self.predictionExtractor = {
            "uitars":extractUITARSPrediction,
            "atlas":extractAtlasPrediction,
            "agentcpmgui":extractCPMPrediction,
            "gemini":extractUITARSPrediction,
            "gpt":extractUITARSPrediction,
        }[self.agentType]
        
        self.actionTranslator = {
            "uitars":translateUITARS2Atlas,
            "atlas":translateAtlas2Atlas,
            "agentcpmgui":translateACG2Atlas,
            "gemini":translateUITARS2Atlas,
            "gpt":translateUITARS2Atlas,
        }[self.agentType]
        
        self.apiKey = None if not hasattr(config, 'apiKey') else config.apiKey
        
    def extractGroundTruth(self, episodeID, step, sample):
        try:
            label = sample["label"]
            if self.agentType != "agentcpmgui":
                gtAction = self.predictionExtractor(label)["action"]
                gtAction = self.actionTranslator(gtAction)
            if self.agentType == "agentcpmgui":
                label = json.loads(label)
                gtAction = self.actionTranslator(label)
            gtActionFormated = qwen_translate_action(gtAction)
            gtActionType = ActionType(gtActionFormated.action_type).name
            
            return gtAction, gtActionFormated, gtActionType
        except Exception as e:
            logger.warning(f"episode ID: {episodeID} step {step} processing error: {e}")
            raise
        
    def preprocessMessage4UITARS(self, episodeID, step, sample, history, benchmarkSetting):
        try:
            record = copy.deepcopy(sample)
            messages:list = record["messages"][:2] # system prompt and user prompt
            
            step_history = history[-8:]
            
            messages.extend(step_history)
            
            
            if benchmarkSetting == "low":
                messages.extend(record["messages"][-2:]) # Current screenshot and low-level instruction
            elif benchmarkSetting == "high":
                messages.extend([record["messages"][-1]])
            else:
                raise NotImplementedError(f"benchmarkSetting {benchmarkSetting} not supported!")
            
            return messages
        except Exception as e:
            logger.warning(f"episode ID: {episodeID} step {step} processing error: {e}")
            raise
        
    def generatePrediction(self, episodeID, step, agent, messages:List[Dict], label:str, benchmarkSetting):
        predText = None
        try:
            predText = agent.generate(messages=messages)
            if isinstance(predText, list) and len(predText) > 0:
                predText = predText[0]
            if self.agentType != "agentcpmgui":
                predDict = self.predictionExtractor(predText)
                predThought, predAction = predDict["thought"], predDict["action"]
            else:
                predThought, predAction = self.predictionExtractor(predText)
            if benchmarkSetting == "low" and predThought is None:
                predThought = self.predictionExtractor(label)["thought"]
            predAtlasAction = self.actionTranslator(predAction)
            return predText, predThought, predAction, predAtlasAction
        except Exception as e:
            logger.warning(f"Predict Error: {e} at episode ID: {episodeID} step {step}! PredText: {predText}")
            return predText, "", "NoneAction", "NONEACTION"
        
    def updateHistory4UITARS(self, episodeID, step, history:List, predThought, predAction, image):
        try:
            updatedHistory = copy.deepcopy(history)
            updatedHistory.extend([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Thought: {predThought}\nAction: {predAction}"
                        }
                    ]
                }
            ])
            return updatedHistory
        except Exception as e:
            logger.warning(f"Update History Error at episode ID: {episodeID} step {step} !")
            return history
        
    def predChunk(self, agentID, deviceIDs, agentType, cache_dir, max_new_tokens, chunk, chunk_start, predResults, listLock, modelPath, progressCounter, progressLock, model_loaded_event, status_queue, benchmarkSetting, apiKey):
        import os
        visible_devices = ",".join(str(i) for i in deviceIDs)
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        
        logger.info(f"Agent {agentID} sees CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        import torch
        torch.cuda.empty_cache()
        torch.cuda.init()
                
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        from .agents import AtlasAgent, UITARSAgent, AgentCPMGUIAgent, APIAgentForState
        
        AgentClass = {
            "uitars": UITARSAgent,
            "atlas": AtlasAgent,
            "agentcpmgui": AgentCPMGUIAgent,
            "gemini":APIAgentForState,
            "gpt":APIAgentForState,
        }[self.agentType]
        
        agent = AgentClass(modelPath, cache_dir=cache_dir, device=device, max_new_tokens=max_new_tokens, benchmarkSetting=benchmarkSetting, apiKey=apiKey)
        
        model_loaded_event.set()
        
        chunk_results = []
        
        chunk.sort(key=itemgetter("episodeID"))
        trajs = {k: list(v) for k, v in groupby(chunk, key=itemgetter("episodeID"))}
        
        try:
            for episodeID, episodeRecords in trajs.items():
                episodeRecords.sort(key=itemgetter("stepID"))
                history = []
                for step, sample in enumerate(episodeRecords):
                    record:Dict = copy.deepcopy(sample)
                    gtAction, gtActionFormated, gtActionType = self.extractGroundTruth(episodeID, step, sample)
                    if record["episodeID"] == "18598":
                        pass
                    if self.agentType == "uitars":
                        messages = self.preprocessMessage4UITARS(episodeID, step, sample, history, benchmarkSetting)
                    else:
                        messages = sample["messages"]
                    record["messages"] = messages
                    if "gpt" in self.agentType:
                        messages[0]["content"] = messages[0]["content"].replace("## Action Space", "## Output Example\n```\nThought: ...\nAction: ...\n```\n## Action Space")
                        messages[0]["content"] = messages[0]["content"].replace("## Note", "## Note\n- All coordinates are unified to [0, 1000]\n")
                        record["messages"] = messages
                    predText, predThought, predAction, predAtlasAction = self.generatePrediction(episodeID, step, agent, messages, sample["label"], benchmarkSetting)
                    
                    if self.agentType == "uitars":
                        history = self.updateHistory4UITARS(episodeID, step, history, predThought, predAction, sample["images"][0])
                    record.update({
                        "ground_truth_action": gtAction,
                        "ground_truth_action_type": gtActionType,
                        "pred_text": predText,
                        "pred_thought":predThought,
                        "pred_action": predAction,
                        "pred_atlas_action":predAtlasAction
                    })

                    chunk_results.append(record)
                    
                    with listLock:
                        predResults.append(record)
                        
                    with progressLock:
                        progressCounter.value += 1        
        
        except Exception as e:
            logger.warning(f"error: {e} occor during predict chunk at agent {agentID}")
            status_queue.put((agentID, "error", f"{e} at {traceback.format_exc()}"))    
            
    def evalMP(self, testData:List[Dict]) -> List[Dict]:
        ctx = mp.get_context("spawn")
        manager = mp.Manager()
        predResults = manager.list([])
        progressCounter = manager.Value('i', 0)
        listLock = manager.Lock()
        progressLock = manager.Lock()
        status_queue = manager.Queue()
        
        testData.sort(key=itemgetter("episodeID"))
        trajs = {k: list(v) for k, v in groupby(testData, key=itemgetter("episodeID"))}
        
        sortedEpisodeIds = sorted(trajs.keys())
        chunkSize = math.ceil(len(trajs) / self.agentCount)
        chunks = [
            sortedEpisodeIds[i:i + chunkSize]
            for i in range(0, len(sortedEpisodeIds), chunkSize)
        ]
        
        chunks = [
            [trajs[episode_id] for episode_id in chunk] for chunk in chunks
        ]
        chunks = [
            [item for sublist in chunk for item in sublist]
            for chunk in chunks
        ]
        
        loadManager = mp.Manager()
        model_loaded_events = [loadManager.Event() for _ in range(self.agentCount)]
        
        processes = []
        for i, chunk in enumerate(chunks):
            chunk_start = i * chunkSize
            p: mp.Process = ctx.Process(
                target=self.predChunk, 
                args=(i, self.deviceIDs[i], self.agentType, self.cache_dir, self.max_new_tokens, chunk, chunk_start, predResults, listLock, self.modelPath, progressCounter, progressLock, model_loaded_events[i], status_queue, self.benchmarkSetting, self.apiKey),
                daemon=False
            )
            p.start()
            processes.append(p)
            
        logger.info("Waiting for all agents to finish loading agents...")
        for event in model_loaded_events:
            event.wait()
            
        logger.info("All agents loaded. Start progress bar.")
        
        with tqdm(total=len(testData), desc=f"evaluate {self.testJsonPath}") as pbar:
            last_progress = 0
            while any(p.is_alive() for p in processes):
                with progressLock:
                    current_progress = progressCounter.value

                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                time.sleep(0.5)
            
            for p in processes:
                p.join(timeout=1080000)
                if p.is_alive():
                    # print(f"Process {p.pid} timed out. Terminating.")
                    logger.warning(f"Process {p.pid} timed out. Terminating.")
                    p.terminate()
                    p.join()
                    
            statuses = []
            while not status_queue.empty():
                statuses.append(status_queue.get())
            for sid, status, info in statuses:
                logger.info(f"[Agent {sid}] Status: {status}, Info: {info}")
            
            failed_agents = [s for s in statuses if s[1] != "success"]
            if failed_agents:
                logger.warning(f"\n {len(failed_agents)} agents failed. You may need to retry or debug.")
        allPredResults = list(predResults)
        allPredResults.sort(key=itemgetter("episodeID"))
        
        return allPredResults
        
    def evaluate(self):
        self.allPredResults = None
        if self.recordJsonPath is not None:
            try:
                with open(self.recordJsonPath, "r") as f:
                    self.allPredResults = json.load(f)
            except FileNotFoundError:
                pass
        if self.allPredResults is None:
            with open(self.testJsonPath, 'r') as f:
                testData = json.load(f)
            if DEBUG:
                testData = testData[:DEBUGCNT]
            try:
                self.allPredResults = self.evalMP(testData)
                self.allPredResults.sort(key=itemgetter("episodeID"))
        
                with open(self.recordSavePath, "w") as f:
                    json.dump(self.allPredResults, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logger.warning(f"pred Error: {e}")
        metrics = self.computeMetrics(self.allPredResults)
        return metrics
    
    def computeMetricsWithJudgement(self, allPredResults:List[Dict], actionTypeJudgement:Callable, actionJudgement:Callable):
        actionSuccess, typeSuccess = 0, 0
        records = []
        typeDetail = {
            "Total":{"type_match": 0, "action_match":0, "sample":0, "type_match_rate": 0, "action_match_rate":0},
            "SCROLL":{"type_match": 0, "action_match":0, "sample":0, "type_match_rate": 0, "action_match_rate":0},
            "PRESS":{"type_match": 0, "action_match":0, "sample":0, "type_match_rate": 0, "action_match_rate":0},
        } 
        
        for i, record in tqdm(enumerate(allPredResults), total=len(allPredResults), desc=f"Analyze prediction"):
            predAction = record["pred_atlas_action"]
            gtAction = record["ground_truth_action"]
            
            try:
                typeFlag = int(actionTypeJudgement(predAction, gtAction))
                # actionFlag = int(actionJudgement(predAction, gtAction, record["layouts"][0], record["width"], record["height"]))
                actionFlag = int(actionJudgement(predAction, gtAction, record))
            except Exception as e:
                logger.warning(f"record{i} Predict Action: {predAction} raises the following exception:\n {e}")
                actionFlag, typeFlag = 0, 0
            record["action_match"] = actionFlag
            record["type_match"] = typeFlag
            
            gtActionType = record["ground_truth_action_type"]
            
            if gtActionType not in typeDetail:
                typeDetail[gtActionType] = {"type_match": typeFlag, "action_match":actionFlag, "sample":1}
            else:
                typeDetail[gtActionType]["type_match"] += typeFlag
                typeDetail[gtActionType]["action_match"] += actionFlag
                typeDetail[gtActionType]["sample"] += 1
            
            typeDetail["Total"]["type_match"] += typeFlag
            typeDetail["Total"]["action_match"] += actionFlag
            typeDetail["Total"]["sample"] += 1
            
            if gtActionType.startswith("PRESS"):
                typeDetail["PRESS"]["type_match"] += typeFlag
                typeDetail["PRESS"]["action_match"] += actionFlag
                typeDetail["PRESS"]["sample"] += 1
            if gtActionType.startswith("SCROLL"):
                typeDetail["SCROLL"]["type_match"] += typeFlag
                typeDetail["SCROLL"]["action_match"] += actionFlag
                typeDetail["SCROLL"]["sample"] += 1
                
            records.append(record)
                   
    
        records.sort(key=itemgetter("episodeID"))
        
        with open(self.recordSavePath, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
            
        # records.sort(key=itemgetter("episodeID"))
        traj = {k: list(v) for k, v in groupby(records, key=itemgetter("episodeID"))}
        
        successTask = 0
        for episode_id, frames in traj.items():
            if all(frame.get("action_match", 0) == 1 for frame in frames):
                successTask += 1
        
        for typename, _ in typeDetail.items():
            try:
                typeDetail[typename]["type_match_rate"] = round(100 * typeDetail[typename]["type_match"] / typeDetail[typename]["sample"], 2)
                typeDetail[typename]["action_match_rate"] = round(100 * typeDetail[typename]["action_match"] / typeDetail[typename]["sample"], 2)
            except ZeroDivisionError:
                continue
            
        metrics = {
            "Task Success Rate": round(successTask * 100 / len(traj), 2),
            **typeDetail
        }
    
        print(json.dumps(metrics, indent=4))
        
        return metrics
    
    def computeMetrics(self, allPredResults:List[Dict]) -> Dict:
        raise NotImplementedError("Please implement `computeMetrics` in subclass")
           
class AndroidControlEvaluator(AgenticEvaluator):
    def __init__(self, config):
        super().__init__(config)
        
        save_dir = Path('./analyses/') / self.agentType / "AndroidControl"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.recordSavePath = save_dir / Path(self.recordSavePath).name
        
        
    def computeMetrics(self, allPredResults):
        actionTypeJudgement = actionTypeJudge
        actionJudgement = androidControlActionJudge
        
        return self.computeMetricsWithJudgement(allPredResults, actionTypeJudgement, actionJudgement)

class AITZEvaluator(AgenticEvaluator):
    def __init__(self, config):
        super().__init__(config)
        
        save_dir = Path('./analyses/') / self.agentType / "AITZ"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.recordSavePath = save_dir / Path(self.recordSavePath).name
        
    def computeMetrics(self, allPredResults):
        actionTypeJudgement = actionTypeJudge
        actionJudgement = aitzActionJudge
        
        return self.computeMetricsWithJudgement(allPredResults, actionTypeJudgement, actionJudgement)
    
class GUIOdysseyEvaluator(AgenticEvaluator):
    def __init__(self, config):
        super().__init__(config)
        
        save_dir = Path('./analyses/') / self.agentType / "GUI-Odyssey"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.recordSavePath = save_dir / Path(self.recordSavePath).name
        
    def computeMetrics(self, allPredResults):
        actionTypeJudgement = actionTypeJudge
        actionJudgement = guiodysseyActionJudge
        
        return self.computeMetricsWithJudgement(allPredResults, actionTypeJudgement, actionJudgement) 
    
class StateActionEvaluator(AgenticEvaluator):
    def __init__(self, config):
        super().__init__(config)
        
        save_dir = Path('./analyses/') / self.agentType / "StateAction"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.recordSavePath = save_dir / Path(self.recordSavePath).name
        
    def updateStateAnalyzeStat(self, typeDetail, gtActionType, typeFlag, actionFlag):
        if gtActionType not in typeDetail:
            typeDetail[gtActionType] = {"type_match": typeFlag, "action_match":actionFlag, "sample":1}
        else:
            typeDetail[gtActionType]["type_match"] += typeFlag
            typeDetail[gtActionType]["action_match"] += actionFlag
            typeDetail[gtActionType]["sample"] += 1
        typeDetail["Total"]["type_match"] += typeFlag
        typeDetail["Total"]["action_match"] += actionFlag
        typeDetail["Total"]["sample"] += 1
        
        if gtActionType.startswith("PRESS") and "PRESS" in typeDetail:
            typeDetail["PRESS"]["type_match"] += typeFlag
            typeDetail["PRESS"]["action_match"] += actionFlag
            typeDetail["PRESS"]["sample"] += 1
        if gtActionType.startswith("SCROLL") and "SCROLL" in typeDetail:
            typeDetail["SCROLL"]["type_match"] += typeFlag
            typeDetail["SCROLL"]["action_match"] += actionFlag
            typeDetail["SCROLL"]["sample"] += 1
            
        return typeDetail
        
    def computeMetricsWithJudgement(self, allPredResults:List[Dict], actionTypeJudgement:Callable, actionJudgement:Callable):
        actionSuccess, typeSuccess = 0, 0
        records = []
        typeDetail = {
            "Total": {"type_match": 0, "action_match":0, "sample":0, "type_match_rate": 0, "action_match_rate":0}
        } 
        posTypeDetail, negTypeDetail = copy.deepcopy(typeDetail), copy.deepcopy(typeDetail)
        posTypeDetail["Erroneous_Toggle"] = {"type_match": 0, "action_match":0, "sample":0, "type_match_rate": 0, "action_match_rate":0}
        negTypeDetail["Erroneous_Toggle"] =  {"type_match": 0, "action_match":0, "sample":0, "type_match_rate": 0, "action_match_rate":0}
        
        for i, record in tqdm(enumerate(allPredResults), total=len(allPredResults), desc=f"Analyze prediction"):
            predAction = record["pred_atlas_action"]
            gtAction = record["ground_truth_action"]
            
            try:
                typeFlag = int(actionTypeJudgement(predAction, gtAction))
                actionFlag = int(actionJudgement(predAction, gtAction, record))
            except Exception as e:
                logger.warning(f"record{i} Predict Action: {predAction} raises the following exception:\n {e}")
                actionFlag, typeFlag = 0, 0
            record["action_match"] = actionFlag
            record["type_match"] = typeFlag
            
            etAction = record["negAction"] if record["positive"] else record["posAction"]
            try:
                etActionFlag = int(actionJudgement(predAction, etAction, record))
                etTypeFlag = int(actionTypeJudgement(predAction, etAction))
            except Exception as e:
                logger.warning(f"record{i} Predict Action: {predAction} raises the following exception:\n {e}")
                actionFlag, typeFlag = 0, 0
            
            record["errorous_toggle_action_match"] = etActionFlag
            record["errorous_toggle_type_match"] = etTypeFlag
            
            gtActionType = record["ground_truth_action_type"]
            
            typeDetail = self.updateStateAnalyzeStat(typeDetail, gtActionType, typeFlag, actionFlag)
                
            if record["positive"]:
                posTypeDetail =  self.updateStateAnalyzeStat(posTypeDetail, gtActionType, typeFlag, actionFlag)
                posTypeDetail["Erroneous_Toggle"]["type_match"] += etTypeFlag
                posTypeDetail["Erroneous_Toggle"]["action_match"] += etActionFlag
                posTypeDetail["Erroneous_Toggle"]["sample"] += 1
                
            else:
                negTypeDetail = self.updateStateAnalyzeStat(negTypeDetail, gtActionType, typeFlag, actionFlag)
                negTypeDetail["Erroneous_Toggle"]["type_match"] += etTypeFlag
                negTypeDetail["Erroneous_Toggle"]["action_match"] += etActionFlag
                negTypeDetail["Erroneous_Toggle"]["sample"] += 1
                
            records.append(record)
                   
    
        records.sort(key=itemgetter("episodeID"))
        
        with open(self.recordSavePath, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
            
        records.sort(key=itemgetter("episodeID"))
        traj = {k: list(v) for k, v in groupby(records, key=itemgetter("episodeID"))}
        
        successTask = 0
        for episode_id, frames in traj.items():
            if all(frame.get("action match", 0) == 1 for frame in frames):
                successTask += 1
        
        for typename, _ in typeDetail.items():
            try:
                typeDetail[typename]["type_match_rate"] = round(100 * typeDetail[typename]["type_match"] / typeDetail[typename]["sample"], 2)
                typeDetail[typename]["action_match_rate"] = round(100 * typeDetail[typename]["action_match"] / typeDetail[typename]["sample"], 2)
                
            except ZeroDivisionError:
                continue
            
        for typename, _ in negTypeDetail.items():
            try:
                negTypeDetail[typename]["type_match_rate"] = round(100 * negTypeDetail[typename]["type_match"] / negTypeDetail[typename]["sample"], 2)
                negTypeDetail[typename]["action_match_rate"] = round(100 * negTypeDetail[typename]["action_match"] / negTypeDetail[typename]["sample"], 2)
                
            except ZeroDivisionError:
                continue
            
        for typename, _ in posTypeDetail.items():
            try:
                posTypeDetail[typename]["type_match_rate"] = round(100 * posTypeDetail[typename]["type_match"] / posTypeDetail[typename]["sample"], 2)
                posTypeDetail[typename]["action_match_rate"] = round(100 * posTypeDetail[typename]["action_match"] / posTypeDetail[typename]["sample"], 2)
                
            except ZeroDivisionError:
                continue
            
        metrics = {
            "Overall":typeDetail,
            "Positive":posTypeDetail,
            "Negative":negTypeDetail,
        }
    
        print(json.dumps(metrics, indent=4))
        
        return metrics
        
    def computeMetrics(self, allPredResults):
        actionTypeJudgement = stateActionTypeJudge
        actionJudgement = stateActionJudge
        
        return self.computeMetricsWithJudgement(allPredResults, actionTypeJudgement, actionJudgement) 


mappings = {
    "android_control":AndroidControlEvaluator,
    "aitz":AITZEvaluator,
    "gui_odyssey":GUIOdysseyEvaluator,
    "state_action":StateActionEvaluator
}

def load_evaluator(config:Namespace) -> Union[AndroidControlEvaluator, AITZEvaluator, GUIOdysseyEvaluator, StateActionEvaluator]:
    return mappings[config.type](config)