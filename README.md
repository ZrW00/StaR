# ⭐ StaR

<div>
 <h2 align="center">See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles</h2>
</div>
</div>

<div>
<br>

<div align="center">

[![Data](https://img.shields.io/badge/Data-4d5eff?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/datasets/ZrW00/StaR_state_control_benchmark)
[![Github](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ZrW00/StaR)
[![arXiv](https://img.shields.io/badge/arXiv-2509.13615-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2509.13615)
</div>
</div>

**This repository is the code implementation of our [paper](https://arxiv.org/abs/2509.13615)**
```
See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles
```
<div align="center">
  <a style="display: inline-block; text-align: center;">
      <img src="./assets/StaR.png">
  </a>
</div>

## 🚀 News

- 2026.2.23 Our paper is accepted by **CVPR 2026**. 
- 2025.9.18 We release the [state control benchmark](https://huggingface.co/datasets/ZrW00/StaR_state_control_benchmark) in our paper.
- 2025.9.17 We release the preprocess and evaluation code of our paper.
- 2025.9.17 We release the video demo of our paper.


##  Video Demo

We provide the video demo corresponding to the Section 5.5 of our paper. The target instruction is `turn wifi on`, with the toggle initially set to `on`, thereby serving as a test for false-positive toggling. The video demo is available at [VideoDemo](./VideoDemo.mp4). 

- OS-Atlas-7B without StaR fails to execute the instruction correctly, resulting in a false-positive toggle.  The agent mistakenly perceives the current toggle state as `off` and incorrectly clicks the toggle, resulting in an unintended state change. It then repeatedly toggles between `on` and `off`, falling into an infinite loop and ultimately failing the task.

- OS-Atlas-7B with StaR, by contrast, executes the instruction successfully. At the critical decision step, the agent adaptively applies the state-aware reasoning chain, correctly perceiving the current toggle state as *on* and appropriately deciding to finish the task, thereby completing the instruction as intended.


## Dependencies
- Install requirements by:
```bash
pip install -r requirements.txt
```
- For training the agents, we adopt the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework. We provide the reference of the source code of 0.9.4.dev0 version. Navigate to the LLaMA-Factory directory and install the dependencies:
```
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```
- For evaluating the agent in the dynamic environment, we adopt the [AndroidWorld](https://github.com/google-research/android_world) framework. Please refer to their repository for deployment instructions.


## State Control Benchmark
We construct a state control benchmark with binary toggle instructions from public datasets to evaluate agent performance on toggle execution. 

Examples are provided in this repository:

- Benchmark samples: [Examples](./data/state/state_control_benchmark_sample.json)
- Corresponding screenshots: [ImagePaths](./GUIData/stateControlBenchmark)

An example record of the state control benchmark are presented as below:

```json
{
    "images": [
        "GUIData/stateControlBenchmark/AITW_episode_8680156250447271550_step_10.jpg"
    ], 
    "img_filename": "episode_8680156250447271550_step_10.jpg", 
    "bbox": [
        814.4,
        360.5,
        914.4,
        460.5
    ], 
    "image_height": 732,
    "image_width": 412,
    "clickCoordinate": [
        864.4,
        410.5
    ], 
    "useBbox": false, 
    "annotation": {
        "is_switch": true, 
        "feature": "picture-in-picture", 
        "state_before_action": "Enabled", 
        "state_after_action": "Disabled",
        "action_effect": "The action turn off picture-in-picture by changing the switch from Enabled to Disabled" 
    },
    "rawClickCoordinate": [
        356,
        300
    ], 
    "posInstruction": "turn off picture-in-picture", 
    "negInstruction": "turn on picture-in-picture", 
    "posAtlasAction": "CLICK <point>[[864.4, 410.5]]</point>", 
    "negAtlasAction": "COMPLETE"
}
```
The description of each field in the state control benchmark is presented below:

| Field Name                   | Description                                                               |
|-----------------------------|----------------------------------------------------------------------------|
| `images`                    | Corresponding GUI screenshot path                                          |
| `img_filename`              | Corresponding GUI screenshot filename, not used                            |
| `bbox`                      | Bounding box of the target element, normalized to `[0, 1000]`              |
| `image_height`, `image_width` | Height and width of the original screenshot (in pixels)                  |
| `clickCoordinate`           | Normalized click coordinate of the target element, normalized to `[0, 1000]`       |
| `useBbox`                   | Whether to use the bounding box to locate the target element (boolean)     |
| `annotation`                | Annotations related to the target element interaction                      |
| └── `is_switch`             | Whether the target element is a toggle switch                              |
| └── `feature`               | Feature name controlled by the toggle (e.g., picture-in-picture)           |
| └── `state_before_action`   | State of the toggle before the click (e.g., Enabled)                       |
| └── `state_after_action`    | State of the toggle after the click (e.g., Disabled)                       |
| └── `action_effect`         | Description of the effect caused by the action (natural language)          |
| `rawClickCoordinate`        | Raw click coordinate (in pixels, not normalized)                           |
| `posInstruction`            | Positive instruction — vary the toggle state                               |
| `negInstruction`            | Negative instruction — maintains the current state                         |
| `posAtlasAction`            | Positive label action (OS-Atlas format)                                    |
| `negAtlasAction`            | Negative label action (OS-Atlas format)                                    |


The full benchmark is available on [huggingface](https://huggingface.co/datasets/ZrW00/StaR_state_control_benchmark).


## Data Preprocessing
Data preprocessing scripts are provided in the [dataPreprocessor](./dataPreprocessor/) directory. Hyperparameters are configured using YAML files.

Example: Preprocess State Control Benchmark for UI-TARS-7B
```yaml
type: state_cot
model: uitars
apiKey: "Your API key for zhipuai"
diversity: true
agentCount: 10
llamafactory: false
stateJsonPathTrain: ./data/state/state_control_benchmark_train.json
stateJsonPathTest: ./data/state/state_control_benchmark_test.json
```

Example: Preprocess AndroidControl Benchmark for UI-TARS-7B
```yaml
type: android_control
model: atlas
apiKey: "Your API key for zhipuai"
state: false
low_level: false
agentCount: 10
llamafactory: true
acjsonPath: GUIData/android_control/jsons
acimagePath: GUIData/android_control/images
aclayoutPath: GUIData/android_control/layouts
cot_trained: true
```

Example: Merge Data for AgentCPM-GUI-8B Training
```yaml
mergeConfigList: 
  - "dataPreprocessorYamls/acg/aitz_cot_trained_llamafactory.yaml"
  - "dataPreprocessorYamls/acg/androidControl_high_cot_trained_llamafactory.yaml"
  - "dataPreprocessorYamls/acg/androidControl_low_cot_trained_llamafactory.yaml"
  - "dataPreprocessorYamls/acg/gui_odyssey_cot_trained_llamafactory.yaml"
  - "dataPreprocessorYamls/acg/state_cot_llama_factory.yaml"
model: agentcpmgui
type: agentic_with_state_cot
```

To preprocess data, run:
```bash
python preprocessData.py --config <path to config yaml> --mergeConfig <path to merge config yaml> 
```

## Train the Agents
The implementation of training the agents are based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). After preprocessing and merging the data, configure the training settings in LLaMA-Factory. (see [Preprocess the Data](#data-preprocessing) for more details). See [README](./LLaMA-Factory/README.md) for more details.


## Evaluate the Agent on Agentic Benchmarks
Evaluation scripts are provided in the [evaluator](./evaluator/) directory. Hyperparameters are configured via YAML files.

Example: Evaluate UI-TARS-7B on State Control Benchmark
```yaml
testJsonPath: "test Json path in data/GUIState/uitars"
modelPath: "path to the agent model"
devicesIDs: "CUDA device IDs for evaluation, such as [0,1,2,3]"
agentCount: 4 # The process number of the evaluation
agentType: uitars
max_new_tokens: 512
benchmarkSetting: high
type: state_action # see (./evaluator/evaluators.py)
recordSavePath: uitars_state_action_predict_test.json # save record file name in ./analyses
```

Example: Evaluate OS-Atlas-7B on AndroidControl-H Benchmark

```yaml
testJsonPath: "test Json path in data/GUIAgentic/android_control/atlas/"
modelPath: "path to the agent model"
devicesIDs: "CUDA device IDs for evaluation, such as [0,1,2,3]"
agentCount: 4 # The process number of the evaluation
agentType: atlas
max_new_tokens: 512
benchmarkSetting: high
type: android_control
recordSavePath: atlas_android_control_high_action_predict_test.json # save record file name in ./analyses
```

To evaluate:
```bash
python evaluate.py --config <path to config yaml>
```

## Evaluate the Agent on Dynamic Environment
To further assess real-world applicability, we construct a dynamic evaluation benchmark consisting of 20 real-world toggle control tasks. This benchmark is implemented on the Android emulator from [AndroidStudio](https://developer.android.com/studio) and built upon the [AndroidWorld](https://github.com/google-research/android_world) framework, enabling evaluation under dynamic and realistic mobile environments. See [README](./android_world/README.md) for more details.


## Acknowledgement
This work can not be done without the help of the following repos:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [AndroidWorld](https://github.com/google-research/android_world)


## Citation
If you find this work useful, please consider citing:
```
@article{wu2025see,
	title={See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles}, 
	author={Zongru Wu and Rui Mao and Zhiyuan Tian and Pengzhou Cheng and Tianjie Ju and Zheng Wu and Lingzhong Dong and Haiyue Sheng and Zhuosheng Zhang and Gongshen Liu},
	year={2025},
	journal={arXiv preprint arXiv:2509.13615},
	url={https://arxiv.org/abs/2509.13615}, 
}
```

    