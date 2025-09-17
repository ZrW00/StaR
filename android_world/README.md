# Additional codes for Androidworld testing

Three files of agent definition, they shall be in `android_world/agents/`
- `atlas.py` for OS-ATLAS
- `uitars.py` for UI-TARS
- `cpm_agent.py` for AgentCPM-GUI

Then add following lines in `run.py`:
```Python
# Add these codes to the top of the file:
from android_world.agents import uitars
from android_world.agents import atlas
from android_world.agents import cpm_agent

# Modify the `_get_agent` function as follows
def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent."""
  print('Initializing agent...')
  agent = None
  if _AGENT_NAME.value == 'human_agent':
    agent = human_agent.HumanAgent(env)
  elif _AGENT_NAME.value == 'random_agent':
    agent = random_agent.RandomAgent(env)
  # Gemini.
  elif _AGENT_NAME.value == 'm3a_gemini_gcp':
    agent = m3a.M3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  elif _AGENT_NAME.value == 't3a_gemini_gcp':
    agent = t3a.T3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  # GPT.
  elif _AGENT_NAME.value == 't3a_gpt4':
    agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  elif _AGENT_NAME.value == 'm3a_gpt4v':
    agent = m3a.M3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  # SeeAct.
  elif _AGENT_NAME.value == 'seeact':
    agent = seeact.SeeAct(env)
  # Betula added here: for cuntom agents
  elif _AGENT_NAME.value == 'uitars':
    agent = uitars.UITARS(env, modelPath="your/model/path")
  elif _AGENT_NAME.value == 'atlas':
    agent = atlas.Atlas(env, modelPath="your/model/path")
  elif _AGENT_NAME.value == 'cpm':
    agent = cpm_agent.CPM(env, modelPath="your/model/path")

  if not agent:
    raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')

  if (
      agent.name in ['M3A', 'T3A', 'SeeAct']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
  agent.name = _AGENT_NAME.value

  return agent
```


One file of custom tasks: `button_diy.py`, it shall be in `android_world/task_evals/single/`

Then modify `android_world/registry.py`:
```Python
# Add this import statement at the top of the file:
from android_world.task_evals.single import button_diy

_TASKS = (
      ...
      # These lines shall be added in the definition of `_TASKS`
      button_diy.TurnOnAlarm9AM, 
      button_diy.TurnOffAlarm9AM,
      button_diy.TurnOnDoNotDisturb,
      button_diy.TurnOffDoNotDisturb,
      button_diy.TurnOnSaveAndFillPaymentMethodsChrome,
      button_diy.TurnOffSaveAndFillPaymentMethodsChrome,
      button_diy.TurnOnAlwaysSecureConnChrome,
      button_diy.TurnOffAlwaysSecureConnChrome,
      button_diy.TurnOnCaptionYoutube,
      button_diy.TurnOffCaptionYoutube,
      ...
)
```

This way our custom task is added into Androidworld's own task collection. After that you can reference them in your startup scripts, like `run.py`.
