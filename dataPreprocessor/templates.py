from string import Template



ANDROIDCONTROL_ATLAS_ACTION_PREDICTION_PROMPT = """
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
__LOW_LEVEL_PLACEHOLDER__
Screenshot: <image>

# Instructions for Determining the Next Action
- Carefully analyze the final goal, previous actions, and the current screenshot.
- Identify the most suitable action based on the context and the goal.
- Make sure the action you suggest aligns with the desired outcome, considering the previous steps.
"""



ANDORIDCONTROL_UITARS_ACTION_PREDICTION_PROMPT = """
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


AITZ_UITARS_ACTION_PREDICTION_PROMPT = """
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')
type(content=\'\')
scroll(direction=\'down or up or right or left\')
enter()
press_back()
press_home()
finished() # Submit the task regardless of whether it succeeds or fails.


## Note
- Use English in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

AITZ_ATLAS_ACTION_PREDICTION_PROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs and give a score. Your skill set includes both basic and custom actions:

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
    - Purpose: SCROLL in the specified direction.
    - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - Example Usage: SCROLL [UP]

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

Custom Action 3: ENTER
    - purpose: Press the enter button.
    - format: ENTER
    - example usage: ENTER

Custom Action 4: IMPOSSIBLE
    - purpose: Indicate the task is impossible.
    - format: IMPOSSIBLE
    - example usage: IMPOSSIBLE

Custom Action 5: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thought and Action.
Thought: Clearly outline your reasoning process for current step.
Action: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

### Final Input Details
Your final goal, previous actions, current screen description, and any additional context are provided as follows:
- **Final Goal**: {finalGoal}
- **Previous Action Descriptions**: {PAD}
__LOW_LEVEL_PLACEHOLDER__
- **Screenshot**: <image>
"""

GUIODYSSEY_UITARS_ACTION_PREDICTION_PROMPT = """
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
press_back()
press_home()
finished() # Submit the task regardless of whether it succeeds or fails.


## Note
- Use English in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

GUIODYSSEY_ATLAS_ACTION_PREDICTION_PROMPT = """
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


Custom Action 3: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

Custom Action 4: LONG_CLICK
    - purpose: Long click at the specified position.
    - format: LONG_CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_CLICK <point>[[101, 872]]</point>

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thought and Action.
Thought: Clearly outline your reasoning process for current step.
Action: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

And your final goal, previous actions and associated screenshot are as follows:

Final goal: {finalGoal}
Previous actions: {previousActions}
__LOW_LEVEL_PLACEHOLDER__
Screenshot: <image>

# Instructions for Determining the Next Action
- Carefully analyze the final goal, previous actions, and the current screenshot.
- Identify the most suitable action based on the context and the goal.
- Make sure the action you suggest aligns with the desired outcome, considering the previous steps.
"""

ANDORIDCONTROL_LOW_GUIR1_ACTION_PREDICTION_PROMPT = """
You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '__GOAL__' with the current step instruction '__STEP_INSTRUCT__', with the action history being '__HISTORY__'.
Please provide the action to perform (enumerate from ['wait', 'long_press', 'click', 'press_back', 'press_home', 'type', 'open_app', 'scroll']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.
Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:
<think> ... </think> <answer>[{'action': enum['wait', 'long_press', 'click', 'press_back', 'press_home', 'type', 'open_app', 'scroll'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>
Note:\n specific input text (no default) is necessary for actions enum['type', 'open_app', 'scroll'] \n Example:
[{'action': enum['wait', 'press_back', 'press_home'], 'point': [-100, -100], 'input_text': 'no input text'}]
[{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': 'no input text'}]
[{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]
[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]
"""

ANDORIDCONTROL_HIGH_GUIR1_ACTION_PREDICTION_PROMPT = """
You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '__GOAL__', with the action history being '__HISTORY__'.
Please provide the action to perform (enumerate from ['wait', 'long_press', 'click', 'press_back', 'press_home', 'type', 'open_app', 'scroll']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.
Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:
<think> ... </think> <answer>[{'action': enum['wait', 'long_press', 'click', 'press_back', 'press_home', 'type', 'open_app', 'scroll'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>
Note:\n specific input text (no default) is necessary for actions enum['type', 'open_app', 'scroll'] \n Example:
[{'action': enum['wait', 'press_back', 'press_home'], 'point': [-100, -100], 'input_text': 'no input text'}]
[{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': 'no input text'}]
[{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]
[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]
"""


SWITCHSTATUSBBOXASSESSMENTPROMPT = """
You are in Switch Status Perception mode. Your task is to determine if clicking within the red box (__BBOX__) toggles a switch state (Enabled/Disabled). Analyze ONLY the UI control fully contained within the red box - completely ignore all other elements on screen.

# Absolute Priorities:
1. The red box boundaries are PRIMARY - only elements fully inside should be evaluated
2. Instruction text is SECONDARY - only use it for context when describing the feature
3. DO NOT consider any elements outside or partially overlapping the red box

# Inputs
1. **Screenshot**: <image> (red box highlights the target area)
2. **Click Action**: "CLICK within red box __BBOX__"
3. **Instruction**: __INSTRUCTION__ (only extract the object or feature being acted on; ignore verbs like "turn on/off")

Note: __BBOX__ is normalized to [0, 1000] coordinates.

# Strict Criteria for Switch Identification:
1. MUST be a physical switch metaphor control (slider/lever style toggle) that exhibits clear binary state change (Enabled↔Disabled)
2. MUST have visible ON/OFF or ENABLED/DISABLED state indicators
3. MUST NOT include:
   - Buttons without persistent state
   - Navigation elements
   - Dropdown selectors
   - Radio buttons
   - Non-toggle UI elements
   - Any element that doesn't visually represent a physical switch
   - Any element not fully contained within the red box

# Analysis Protocol:
1. Identify ALL UI elements fully contained within the red box boundaries
2. If no elements exist inside box:
   - Return is_switch=False IMMEDIATELY
   - Do NOT consider nearby elements
3. For contained elements:
   - Evaluate toggle capability
   - Determine current state
   - Assess state change potential

# Steps:
1. Locate and examine the UI element inside the red box.
2. Decide if it's an enable/disable control (e.g., switch, checkbox).
3. Determine if its state changes after the click.
4. Identify the function or target object of the control.
5. Describe the state before and after the click (must be "Enabled" or "Disabled").
6. Assess how the instruction relates to the observed state change.

# Output format
If switch state changes:
{
    "is_switch": True,
    "feature": "Name or function of the switch (in noun form, e.g., 'Wi-Fi', 'Bluetooth', 'Dark Mode')",
    "state_before_action": "Enabled" or "Disabled",
    "state_after_action": "Enabled" or "Disabled",
    "action_effect": "The action [turned on|turned off] [feature] by changing the switch from [state_before_action] to [state_after_action]"
}

If no state change:
{
    "is_switch": False,
    "action_effect": "Observed behavior (navigation/selection/etc)"
}

# Examples

## Click changes switch state
Instruction: Turn on Wi-Fi
{
    "is_switch": True,
    "feature": "Wi-Fi control",
    "state_before_action": "Disabled",
    "state_after_action": "Enabled",
    "action_effect": "The red box action [turned on] [Wi-Fi] by changing the switch from [Disabled] to [Enabled]"
}

## Click does not change switch state
Instruction: Tap the menu icon
{
    "is_switch": False,
    "action_effect": "The red box element is not a switch; it triggered navigation"
}

# Critical Reminders:
- The red box boundaries are LAW
- Instruction text is ONLY for feature naming
- Never infer functionality from instruction verbs
- Never consider elements outside the box
- Never continue searching if no switch is found inside
- Strictly require FULL containment within box boundaries

Output:
"""

SWITCHSTATUSPOINTASSESSMENTPROMPT = """
You are in Switch Status Perception mode. Your task is to decide whether a click at the red dot (__COORDINATE__) changes the state of a nearby switch (Enabled or Disabled).  Analyze ONLY the UI control directly under the red dot - completely ignore all other elements on screen.

# Absolute Priorities:
1. The red dot location is PRIMARY - its exact position determines which UI element to evaluate
2. Instruction text is SECONDARY - only use it for context when describing the feature
3. DO NOT consider any elements outside the red dot's immediate vicinity

# Inputs
1. **Screenshot**: <image>
2. **Click location**: Red dot at __COORDINATE__ (normalized to [0, 1000])
3. **Instruction**: __INSTRUCTION__ (only extract the object or feature being acted on; ignore verbs like "turn on/off")

# Strict Criteria for Switch Identification:
1. MUST be a physical switch metaphor control (slider/lever style toggle) that exhibits clear binary state change (Enabled↔Disabled)
2. MUST have visible ON/OFF or ENABLED/DISABLED state indicators
3. MUST NOT include:
   - Buttons without persistent state
   - Navigation elements
   - Dropdown selectors
   - Radio buttons
   - Non-toggle UI elements
   - Any element that doesn't visually represent a physical switch
   - Any element not directly under the red dot

   

# Steps
1. Locate the UI element nearest the red dot.
2. Check if it's an enable/disable control (e.g., toggle, checkbox). If no toggle exists under dot: 
   - Return is_switch=False IMMEDIATELY
   - Do NOT search nearby
3. Determine whether its state changes due to the click.
4. Identify its function (what it controls).
5. Describe its state before and after the click as "Enabled" or "Disabled".
6. Optionally relate the outcome to the object mentioned in the instruction (ignore action verbs).

# Output format:
If switch state changes:
{
    "is_switch": True,
    "feature": "Name or function of the switch (in noun form, e.g., 'Wi-Fi', 'Bluetooth', 'Dark Mode')",
    "state_before_action": "Enabled" or "Disabled",
    "state_after_action": "Enabled" or "Disabled",
    "action_effect": "The action [turned on|turned off] [feature] by changing the switch from [state_before_action] to [state_after_action]"
}

If no state change:
{
    "is_switch": False,
    "action_effect": "Observed behavior (navigation/selection/etc)"
}

# Examples:

## Click changes switch state
Instruction: Turn on Wi-Fi
{
    "is_switch": True,
    "feature": "Wi-Fi control",
    "state_before_action": "Disabled",
    "state_after_action": "Enabled",
    "action_effect": "The action [turned on] [Wi-Fi] by changing the switch from [Disabled] to [Enabled]"
}

## Click does not change switch state
Instruction: Tap the menu icon
{
    "is_switch": False,
    "action_effect": "The red dot element is not a switch; it triggered navigation"
}

# Critical Reminders:
- The red dot's position is LAW
- Instruction text is ONLY for feature naming
- Never infer functionality from instruction verbs
- Never consider elements beyond the red dot
- Never continue searching if no switch is found

Output:
"""

SWITCH_STATUS_BBOX_ASSESSMENT_STAGE1_PROMPT = """
You are in Switch Detection Mode. Your task is to determine whether the UI element fully enclosed in the red box (__BBOX__) is a switch-type control.

# Inputs:
1. Screenshot: <image> (red box highlights the target)
2. Click Action: "CLICK within red box __BBOX__"
3. Instruction: __INSTRUCTION__ (only extract the object being acted on; ignore action verbs like "enable/disable" and "turn on/off")

# Scope:
- Only analyze the element fully inside the red box.
- Completely ignore surrounding or adjacent elements.
- Do not infer function from layout, instruction, or context.

# Valid Switch Criteria:
An element is a **switch** only if it satisfies **all** of the following:

1. **It is a UI control**, not a label or plain text.
2. **It has a visible binary state** (e.g., ON vs OFF, Enabled vs Disabled, Checked vs Unchecked).
3. **It supports persistent toggle** — clicking must flip state, not trigger one-time action.
4. **It gives visual feedback** that indicates current state before and after the click (e.g., toggle position, color, icon, label change).


# Common Switch Types:
- Checkbox (✓ / empty box)
- Toggle slider
- Dual-state labeled button (e.g., text/icon flips between Enabled / Disabled)

# Not a Switch If:
- The red box contains **plain text**, such as "Vibrate", "Always" or "Choose text color"
- It is a **button** used for **single-use action** (e.g., "Subscribe", "Skip", "Continue")
- It lacks clear binary state indication
- It has no visual change after click
- It is a **static label**, description, or non-interactive UI
- No valid UI element is fully inside the red box

# Rules:
- Do NOT guess based on text like "Subscribe" or "Start"
- Do NOT treat standalone text, button labels, or descriptions as switches
- Do NOT infer behavior from nearby UI elements
- Do NOT treat labels or descriptions as switches
- Only consider visual evidence of **binary toggle capability** inside the red box


# Output Format:
If it is a switch (visual toggle or toggle-button):
{
  "is_switch": true
}
If not:
{
  "is_switch": false
}

Output:
"""


SWITCH_STATUS_POINT_ASSESSMENT_STAGE1_PROMPT = """
You are in Switch Detection Mode. Your task is to determine whether the UI element directly under the red dot (__COORDINATE__) is a switch-type control.

# Inputs:
1. Screenshot: <image>
2. Click Location: Red dot at __COORDINATE__ ([0, 1000])
3. Instruction: __INSTRUCTION__ (only extract the object being acted on; ignore action verbs like "enable/disable" and "turn on/off")

# Scope:
- Only evaluate the element directly under the red dot.
- Completely ignore surrounding elements, layout, or text context.
- Ignore action verbs from instruction text.

# Switch Criteria:
An element is a **switch** only if it meets **all** of the following:

1. **It is a UI control**, not a label or plain text.
2. **It has a visible binary state** (e.g., ON vs OFF, Enabled vs Disabled, Checked vs Unchecked).
3. **It responds to click by toggling state**, not by navigation or launching a new screen.
4. **It shows localized visual feedback** before and after the click (e.g., a handle moving, checkbox marking, background or icon flipping).
5. **Its switchable area is visibly self-contained**, not part of a larger list item or heading.

# Common Switch Types:
- Checkbox (✓ / empty box)
- Toggle slider
- Dual-state labeled button (e.g., text/icon flips between Enabled / Disabled)

# Not a Switch If:
- It is **plain text** (e.g. "Vibrate", "Always", "Choose text color", "Languages", "Inbox", "Flashlight")
- It is **part of a navigation list**, not a togglable control
- It **launches a new page or action, not a state toggle**
- It has **no visual cue for two toggle states**
- There is no visual change under the red dot after clicking
- It is a **button** used for **single-use action** (e.g., "Subscribe", "Skip", "Continue")
- It lacks clear **binary state indication**
- It is a **static label**, description, or non-interactive UI
- No identifiable, bounded, interactive UI element exists under the red dot


# Rules:
- Do NOT guess based on text like "Subscribe" or "Start"
- Do NOT treat **standalone text, button labels, descriptions, rows, list items, or feature labels** as switches
- Do NOT infer behavior from nearby UI elements
- Do NOT treat labels or descriptions as switches
- Only consider visual evidence of **binary toggle capability** under the red dot
- Only treat a component as a switch if its own appearance shows binary toggle behavior

# Output Format:
If it is a switch:
{
  "is_switch": true
}
If not:
{
  "is_switch": false
}

Output:
"""


SWITCH_STATUS_BBOX_ASSESSMENT_STAGE2_PROMPT = """
You are in Switch Status Perception Mode. Your task is to determine whether the UI element fully enclosed in the red box (__BBOX__) is a binary switch control that changes state (Enabled or Disabled) when clicked.

# Inputs:
1. Screenshot: <image> (the red box highlights the clickable element)
2. Click Action: "CLICK within red box __BBOX__" (coordinates normalized to [0, 1000])
3. Instruction: __INSTRUCTION__ (used only to name the feature; ignore action verbs like "enable/disable" and "turn on/off")

# Evaluation Scope:
- Analyze ONLY the element fully contained within the red box.
- Completely IGNORE any content outside or partially inside the box.
- Use the instruction ONLY to identify the target feature (noun only, e.g., "Wi-Fi", "Notifications").
- Do NOT infer behavior from instruction verbs or surrounding layout.

# Valid Switch Criteria:
An element is a **switch** only if it satisfies **all** of the following:

1. **It is an interactive UI control**, not a label, icon, or plain text.
2. **It has a visible binary state** (e.g., ON vs OFF, Enabled vs Disabled, Checked vs Unchecked).
3. **It supports persistent toggle** — clicking must flip state reliably, not just trigger an one-time action.
4. **It gives immediate visual feedback** that indicates current state before and after the click (e.g., toggle position, color, icon, label change).

# Common Switch Types:
- Checkbox (✓ / empty box)
- Toggle slider
- Dual-state labeled button (e.g., text/icon flips between Enabled / Disabled)

# Not a Switch If:
- The red box contains **plain text**, such as "Vibrate", "Always" or "Choose text color"
- It is a **button** used for **single-use action** (e.g., "Subscribe", "Skip", "Continue")
- It lacks clear binary state indication
- It has no visual change after click
- It is a **static label**, description, or non-interactive UI
- No valid UI element is fully inside the red box


# Decision Procedure:
1. Locate the UI element fully within the red box.
2. Decide whether it is a switch with binary, persistent states.
3. Extract the controlled feature name from __INSTRUCTION__ (use noun only).
4. Determine the current state **before** the click.
5. Determine the expected state **after** the click. 
6. Output structured result as specified.

# Output Format:

If it is a switch and clicking it changes state:
{
  "is_switch": true,
  "feature": "[Feature name in noun form]",
  "state_before_action": "Enabled" or "Disabled",
  "state_after_action": "Enabled" or "Disabled",
  "action_effect": "The action [turn on|turn off] [feature] by changing the switch from [state_before_action] to [state_after_action]"
}

If it is not a switch or does not toggle state:
{
  "is_switch": false,
  "action_effect": "The red box element is not a switch; it triggered [e.g., navigation, selection, or no state change]"
}

# Examples

Instruction: Turn on Bluetooth  
{
  "is_switch": true,
  "feature": "Bluetooth",
  "state_before_action": "Disabled",
  "state_after_action": "Enabled",
  "action_effect": "The action turn on Bluetooth by changing the switch from Disabled to Enabled"
}

Instruction: Tap profile icon  
{
  "is_switch": false,
  "action_effect": "The red box element is not a switch; it triggered navigation"
}

# Important:
- The red box defines the evaluation boundary — never go outside.
- Do NOT infer meaning from verbs in the instruction.
- Only report a switch if the element shows two visual states and persistent toggle behavior.
- Precision is critical: misclassifying labels or buttons as switches is unacceptable.

Outputs:
"""


SWITCH_STATUS_POINT_ASSESSMENT_STAGE2_PROMPT = """
You are in Switch Status Perception Mode. Your task is to determine whether the UI element directly under the red dot (__COORDINATE__) is a binary switch control, and whether clicking it changes its state (Enabled or Disabled).

# Inputs:
1. Screenshot: <image>
2. Click: Red dot at __COORDINATE__ (normalized to [0, 1000])
3. Instruction: __INSTRUCTION__ (used only to name the feature; ignore action verbs like "enable/disable" and "turn on/off")


# Evaluation Scope:
- Only analyze the UI element directly under the red dot.
- Completely ignore surrounding elements or context.
- Do not infer behavior based on the instruction’s wording.

# Valid Switch Criteria:
An element is a **switch** only if it satisfies **all** of the following:

1. **It is an interactive UI control**, not a label, icon, or plain text.
2. **It has a visible binary state** (e.g., ON vs OFF, Enabled vs Disabled, Checked vs Unchecked).
3. **It supports persistent toggle** — clicking must flip state reliably, not just trigger an one-time action.
4. **It gives immediate visual feedback** that indicates current state before and after the click (e.g., toggle position, color, icon, label change).

# Common Switch Types:
- Checkbox (✓ / empty box)
- Toggle slider
- Dual-state labeled button (e.g., text/icon flips between Enabled / Disabled)

# Not a Switch If:
- It is **plain text**, such as "Vibrate", "Always" or "Choose text color"
- It is a **button** used for **single-use action** (e.g., "Subscribe", "Skip", "Continue")
- It lacks clear binary state indication
- It has no visual change after click
- It is a **static label**, description, or non-interactive UI
- There is no identifiable interactive control under the red dot

# Decision Steps:
1. Find the element under the red dot.
2. If no toggle control exists: return `"is_switch": false` immediately.
3. If a toggle exists:
   - Use the instruction to extract the **controlled feature** (noun only).
   - Determine its state **before** and **after** the click.
   - Report the toggle effect clearly.

# Output Format

If switch state changes:
{
  "is_switch": true,
  "feature": "[Feature name, noun only]",
  "state_before_action": "Enabled" or "Disabled",
  "state_after_action": "Enabled" or "Disabled",
  "action_effect": "The action [turn on|turn off] [feature] by changing the switch from [state_before_action] to [state_after_action]"
}

If no switch or no state change:
{
  "is_switch": false,
  "action_effect": "The red dot element is not a switch; it triggered [navigation/selection/no effect/etc]"
}

# Examples

Instruction: Turn off Location
{
  "is_switch": true,
  "feature": "Location",
  "state_before_action": "Enabled",
  "state_after_action": "Disabled",
  "action_effect": "The action turn off Location by changing the switch from Enabled to Disabled"
}

Instruction: Tap profile icon
{
  "is_switch": false,
  "action_effect": "The red dot element is not a switch; it triggered navigation"
}

# Rules:
- The red dot location is final — only use that UI element
- Do NOT treat static text or labels as switches
- Do NOT infer from instruction semantics
- Switches must show two visual states and toggle when clicked

Output:
"""


STATE_ACTION_PREDICT_PROMPT_QWEN = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
        
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning, follow current step instruction to determine the most appropriate next action. 

And your final goal and associated screenshot are as follows:

Final goal: {finalGoal}
Screenshot: <image>

# Instructions for Determining the Next Action
- Carefully analyze the final goal and the current screenshot.
- Identify the most suitable action based on the context and the goal.
- Make sure the action you suggest aligns with the desired outcome, considering the previous steps.

# Output Format
Your output must strictly follow the above action format, and especially avoid using unnecessary quotation marks or other punctuation marks. (where osatlas action must be one of the action formats I provided).

Output:
"""



STATE_ACTION_PREDICT_PROMPT_ATLAS = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
        
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thought and Action.
Thought: Clearly outline your reasoning process for current step.
Action: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

Your current task instruction, action history, and associated screenshot are as follows:

Final goal: {finalGoal}
History: {history}
Screenshot: <image>

# Instructions for Determining the Next Action
- Carefully analyze the final goal and the current screenshot.
- Identify the most suitable action based on the context and the goal.
- Make sure the action you suggest aligns with the desired outcome, considering the previous steps.

# Output Format
Your output must strictly follow the above action format, and especially avoid using unnecessary quotation marks or other punctuation marks. (where osatlas action must be one of the action formats I provided).

Output:
"""


STATE_ACTION_PREDICT_PROMPT_UITARS = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: click(start_box='<|box_start|>(x,y)<|box_end|>')
    - example usage: click(start_box='<|box_start|>(880,569)<|box_end|>')
        
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: Finished
    - purpose: Indicate the task is finished.
    - format: finished()
    - example usage: finished()

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning, follow current step instruction to determine the most appropriate next action. 

And your final goal and associated screenshot are as follows:

Final goal: {finalGoal}
Screenshot: <image>

# Instructions for Determining the Next Action
- Carefully analyze the final goal and the current screenshot.
- Identify the most suitable action based on the context and the goal.
- Make sure the action you suggest aligns with the desired outcome, considering the previous steps.

# Output Format
Your output must strictly follow the above action format, and especially avoid using unnecessary quotation marks or other punctuation marks. 

Output:
"""

STATE_ACTION_PREDICT_PROMPT_UITARS_WITH_THOUGHT_BACKUP = """
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')
finished() # Submit the task regardless of whether it succeeds or fails.


## Note
- Use English in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}

## ScreenShot
<image>
"""

STATE_ACTION_PREDICT_PROMPT_UITARS_WITH_THOUGHT = """
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



STATE_ACTION_PREDICT_PROMPT_GUIR1_WITH_THOUGHT = """
You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '__INSTRUCTION__', with the action history being None.
Please provide the action to perform (enumerate from ['complete', 'click', 'wait', 'long_press', 'press_back', 'press_home', 'type', 'open_app', 'scroll']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.
Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:
<think> ... </think> <answer>[{'action': enum['complete', 'click', 'wait', 'long_press', 'press_back', 'press_home', 'type', 'open_app', 'scroll'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>
Example:
[{'action': enum['complete'], 'point': [-100, -100], 'input_text': 'no input text'}]
[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]
[{'action': enum['long_press'], 'point': [123, 300], 'input_text': 'no input text'}]
[{'action': enum['wait', 'press_back', 'press_home'], 'point': [-100, -100], 'input_text': 'no input text'}]
[{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]
[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]
"""



STATE_ACTION_PREDICT_PROMPT_CPM_WITH_THOUGHT = """
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

<Question>{instruction}</Question>
当前屏幕截图：
<image>
"""


STATE_INTERPRETATION_WITH_MIDDLE_REASONING_REFINED_POINT_PROMPT = """
You are in State Interpretation Mode. Your task is to analyze the UI element located at __COORDINATE__, determine its current state based on the screenshot, interpret the user's instruction, and decide the appropriate action to fulfill the user's intent.

# Inputs:
1. Screenshot: <image>
2. UI element position: __COORDINATE__ (a point [x, y] normalized to [0, 1000])
3. User instruction: __INSTRUCTION__ (e.g., "Turn off Bluetooth", "Disable Wi-Fi", "Enable Location")

# Assumptions:
- The UI element in the bounding box is a binary switch: either "On" or "Off".
- Clicking toggles the switch's state.
- The instruction indicates the desired **final state**.
- You must visually determine the current state from the screenshot.
- Do not infer the current state from the instruction text.
- Always compare the **visual state** with the **instruction state**.
- Reasoning and decision must be grounded in the **screenshot**, not just language.

# Your Tasks:
1. Identify the **feature name** (noun-only, e.g., "Wi-Fi").
2. Determine the current **state_before_action**: "On" or "Off".
3. Determine the **desired_final_state** from instruction: "On" or "Off".
4. Provide a **reasoning trace** using this template:
   - "The switch is currently [On/Off]. The instruction requests [On/Off]. The states [match/differ], so the action is [COMPLETE/CLICK]."
5. Compare the two states to decide the appropriate **action**:
   - If they match, action = "COMPLETE".
   - If they differ, action = "CLICK".
6. Predict the state the switch would be in after **a simulated click**, regardless of whether the action is "CLICK" or "COMPLETE". This represents the toggled state.


# Output Format:

{
  "feature": "[Feature name, noun only]",
  "state_before_action": "On" or "Off",
  "desired_final_state": "On" or "Off",
  "reasoning": "[Concise explanation based on screenshot and instruction]",
  "action": "CLICK" or "COMPLETE",
  "state_after_action": "On" or "Off"
}

# Example

Inputs:
- Screenshot: A settings UI showing a switch for "Location", currently ON
- Coordinate: <point>[[542, 168]]</point>
- User instruction: "Turn off Location"

Output:
{
  "feature": "Location",
  "state_before_action": "On",
  "desired_final_state": "Off",
  "reasoning": "The switch is currently On. The instruction requests Off. The states differ, so the action is CLICK.",
  "action": "CLICK",
  "state_after_action": "Off"
}
"""


STATE_INTERPRETATION_WITH_MIDDLE_REASONING_REFINED_BBOX_PROMPT = """
You are in State Interpretation Mode. Your task is to analyze the UI element inside the bounding box __BBOX__, determine its current state based on the screenshot, interpret the user's instruction, and decide the appropriate action to fulfill the user's intent.

# Inputs:
1. Screenshot: <image>
2. Bounding box: __BBOX__ (<|box_start|>[x1, y1, x2, y2]<|box_end|>)
3. User instruction: __INSTRUCTION__ (e.g., "Turn off Bluetooth", "Disable Wi-Fi", "Enable Location")

# Assumptions:
- The UI element in the bounding box is a binary switch: either "On" or "Off".
- Clicking toggles the switch's state.
- The instruction indicates the desired **final state**.
- You must visually determine the current state from the screenshot.
- Do not infer the current state from the instruction text.
- Always compare the **visual state** with the **instruction state**.
- Reasoning and decision must be grounded in the **screenshot**, not just language.

# Your Tasks:
1. Identify the **feature name** (noun-only, e.g., "Wi-Fi").
2. Determine the current **state_before_action**: "On" or "Off".
3. Determine the **desired_final_state** from instruction: "On" or "Off".
4. Provide a **reasoning trace** using this template:
   - "The switch is currently [On/Off]. The instruction requests [On/Off]. The states [match/differ], so the action is [COMPLETE/CLICK]."
5. Compare the two states to decide the appropriate **action**:
   - If they match, action = "COMPLETE".
   - If they differ, action = "CLICK".
6. Predict the state the switch would be in after **a simulated click**, regardless of whether the action is "CLICK" or "COMPLETE". This represents the toggled state.


# Output Format:

{
  "feature": "[Feature name, noun only]",
  "state_before_action": "On" or "Off",
  "desired_final_state": "On" or "Off",
  "reasoning": "[Concise explanation based on screenshot and instruction]",
  "action": "CLICK" or "COMPLETE",
  "state_after_action": "On" or "Off"
}

# Example

Inputs:
- Screenshot: A settings UI showing a switch for "Location", currently ON
- Bounding box: <|box_start|>[510, 160, 580, 200]<|box_end|>
- User instruction: "Turn off Location"

Output:
{
  "feature": "Location",
  "state_before_action": "On",
  "desired_final_state": "Off",
  "reasoning": "The switch is currently On. The instruction requests Off. The states differ, so the action is CLICK.",
  "action": "CLICK",
  "state_after_action": "Off"
}
"""
