## Betula added this file for diy button tasks
#  
# Tasks:
# SystemBluetoothTurnOff
# SystemBluetoothTurnOffVerify
# SystemBluetoothTurnOn
# SystemBluetoothTurnOnVerify
# SystemWifiTurnOff
# SystemWifiTurnOffVerify
# SystemWifiTurnOn
# SystemWifiTurnOnVerify
# TurnOffWifiAndTurnOnBluetooth
# TurnOnWifiAndOpenApp
# TurnOnAlarm9AM
# TurnOffAlarm9AM
# TurnOnCaptionYoutube
# TurnOffCaptionYoutube
# TurnOnDoNotDisturb
# TurnOffDoNotDisturb
# TurnOnSaveAndFillPaymentMethodsChrome
# TurnOffSaveAndFillPaymentMethodsChrome
# TurnOnAlwaysSecureConnChrome
# TurnOffAlwaysSecureConnChrome



from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import representation_utils
from android_world.task_evals import task_eval
from android_world.task_evals.single import clock

import random


class NoTargetButtonFoundException(ValueError):
    pass


def _close_chrome(env: interface.AsyncEnv):
    adb_utils.clear_app_data(
        adb_utils.extract_package_name(adb_utils.get_adb_activity("chrome")),
        env.controller,
    )
    return


def _is_fill_payment_on(
    ui_elements: list[representation_utils.UIElement], current_activity: str
) -> bool:
    if "chrome" not in current_activity.lower():
        return False

    just_find = False

    for element in ui_elements:
        if (
            element.package_name == "com.android.chrome"
            and element.text == "Save and fill payment methods"
        ):
            just_find = True
        if just_find and element.resource_name == "com.android.chrome:id/switchWidget":
            return element.is_checked
    return False


def _is_fill_payment_off(
    ui_elements: list[representation_utils.UIElement], current_activity: str
) -> bool:
    if "chrome" not in current_activity.lower():
        return False

    just_find = False

    for element in ui_elements:
        if (
            element.package_name == "com.android.chrome"
            and element.text == "Save and fill payment methods"
        ):
            just_find = True
        if just_find and element.resource_name == "com.android.chrome:id/switchWidget":
            return not element.is_checked
    return False


def _is_secure_conn_on(
    ui_elements: list[representation_utils.UIElement], current_activity: str
) -> bool:
    if "chrome" not in current_activity.lower():
        return False

    just_find = False

    for element in ui_elements:
        if (
            element.package_name == "com.android.chrome"
            and element.text == "Always use secure connections"
        ):
            just_find = True
        if just_find and element.resource_name == "com.android.chrome:id/switchWidget":
            return element.is_checked
    return False


def _is_secure_conn_off(
    ui_elements: list[representation_utils.UIElement], current_activity: str
) -> bool:
    if "chrome" not in current_activity.lower():
        return False

    just_find = False

    for element in ui_elements:
        if (
            element.package_name == "com.android.chrome"
            and element.text == "Always use secure connections"
        ):
            just_find = True
        if just_find and element.resource_name == "com.android.chrome:id/switchWidget":
            return not element.is_checked
    return False


class _ChromeButtonTask(task_eval.TaskEval):
    app_names = ("chrome",)

    def initialize_task(self, env: interface.AsyncEnv):
        super().initialize_task(env)
        _close_chrome(env)

    def tear_down(self, env: interface.AsyncEnv):
        super().tear_down(env)
        _close_chrome(env)

    @classmethod
    def generate_random_params(cls) -> dict[str, str]:
        return {}


class TurnOnSaveAndFillPaymentMethodsChrome(_ChromeButtonTask):
    complexity = 1
    schema = {
        "type": "object",
        "properties": {},
    }
    template = "Turn on save and fill payment methods in Chrome's settings."

    def is_successful(
        self,
        env: interface.AsyncEnv,
    ) -> float:
        super().is_successful(env)
        ui_elements = env.get_state().ui_elements
        current_activity = adb_utils.get_current_activity(env.controller)[0]
        return (
            1.0
            if _is_fill_payment_on(
                ui_elements=ui_elements,
                current_activity=current_activity,
            )
            else 0.0
        )


class TurnOffSaveAndFillPaymentMethodsChrome(_ChromeButtonTask):
    complexity = 1
    schema = {
        "type": "object",
        "properties": {},
    }
    template = "Turn off save and fill payment methods in Chrome's settings."

    def is_successful(
        self,
        env: interface.AsyncEnv,
    ) -> float:
        super().is_successful(env)
        ui_elements = env.get_state().ui_elements
        current_activity = adb_utils.get_current_activity(env.controller)[0]
        return (
            1.0
            if _is_fill_payment_off(
                ui_elements=ui_elements,
                current_activity=current_activity,
            )
            else 0.0
        )


class TurnOnAlwaysSecureConnChrome(_ChromeButtonTask):
    complexity = 1
    schema = {
        "type": "object",
        "properties": {},
    }
    template = "Turn on Always use the secure connections in Chrome's settings."

    def is_successful(
        self,
        env: interface.AsyncEnv,
    ) -> float:
        super().is_successful(env)
        ui_elements = env.get_state().ui_elements
        current_activity = adb_utils.get_current_activity(env.controller)[0]
        return (
            1.0
            if _is_secure_conn_on(
                ui_elements=ui_elements,
                current_activity=current_activity,
            )
            else 0.0
        )


class TurnOffAlwaysSecureConnChrome(_ChromeButtonTask):
    complexity = 1
    schema = {
        "type": "object",
        "properties": {},
    }
    template = "Turn off Always use the secure connections in Chrome's settings."

    def is_successful(
        self,
        env: interface.AsyncEnv,
    ) -> float:
        super().is_successful(env)
        ui_elements = env.get_state().ui_elements
        current_activity = adb_utils.get_current_activity(env.controller)[0]
        return (
            1.0
            if _is_secure_conn_off(
                ui_elements=ui_elements,
                current_activity=current_activity,
            )
            else 0.0
        )


def _close_youtube(env: interface.AsyncEnv):
    adb_utils.clear_app_data(
        adb_utils.extract_package_name(adb_utils.get_adb_activity("youtube")),
        env.controller,
    )
    return


def _is_caption_on(
    ui_elements: list[representation_utils.UIElement], current_activity: str
) -> bool:
    if "setting" not in current_activity.lower():
        return False

    just_find = False

    for element in ui_elements:
        if element.text == "Show captions":
            just_find = True
        if just_find and element.resource_name == "android:id/switch_widget":
            return element.is_checked
    return False


def _is_caption_off(
    ui_elements: list[representation_utils.UIElement], current_activity: str
) -> bool:
    if "setting" not in current_activity.lower():
        return False

    just_find = False

    for element in ui_elements:
        if element.text == "Show captions":
            just_find = True
        if just_find and element.resource_name == "android:id/switch_widget":
            return not element.is_checked
    return False


class _YoutubeButtonTask(task_eval.TaskEval):
    app_names = ("youtube",)

    def initialize_task(self, env: interface.AsyncEnv):
        super().initialize_task(env)
        _close_youtube(env)

    def tear_down(self, env: interface.AsyncEnv):
        super().tear_down(env)
        _close_youtube(env)

    @classmethod
    def generate_random_params(cls) -> dict[str, str]:
        return {}


class TurnOnCaptionYoutube(_YoutubeButtonTask):
    complexity = 1
    schema = {
        "type": "object",
        "properties": {},
    }
    template = "Turn on captions in Youtube's settings."

    def is_successful(
        self,
        env: interface.AsyncEnv,
    ) -> float:
        super().is_successful(env)
        ui_elements = env.get_state().ui_elements
        current_activity = adb_utils.get_current_activity(env.controller)[0]
        return (
            1.0
            if _is_caption_on(
                ui_elements=ui_elements,
                current_activity=current_activity,
            )
            else 0.0
        )


class TurnOffCaptionYoutube(_YoutubeButtonTask):
    complexity = 1
    schema = {
        "type": "object",
        "properties": {},
    }
    template = "Turn off captions in Youtube's settings."

    def is_successful(
        self,
        env: interface.AsyncEnv,
    ) -> float:
        super().is_successful(env)
        ui_elements = env.get_state().ui_elements
        current_activity = adb_utils.get_current_activity(env.controller)[0]
        return (
            1.0
            if _is_caption_off(
                ui_elements=ui_elements,
                current_activity=current_activity,
            )
            else 0.0
        )


def _is_nine_alarm_on(
    ui_elements: list[representation_utils.UIElement], current_activity: str
) -> bool:
    """Checks if 9:00 alarm is on"""
    if "DeskClock" not in current_activity:
        return False

    just_find_9alarm = False

    for element in ui_elements:
        if (
            element.package_name == "com.google.android.deskclock"
            and element.text == "09:00"
        ):
            just_find_9alarm = True
        if (
            just_find_9alarm
            and element.resource_name == "com.google.android.deskclock:id/onoff"
        ):
            return element.is_checked
    raise ValueError("Can't find target button on the screen.")


class TurnOnAlarm9AM(clock._ClockEval):
    complexity = 1
    schema = {
        "type": "object",
        "properties": {},
    }
    template = "Trun on alarm at 9:00 AM."

    def is_successful(
        self,
        env: interface.AsyncEnv,
    ) -> float:
        super().is_successful(env)
        ui_elements = env.get_state().ui_elements
        current_activity = adb_utils.get_current_activity(env.controller)[0]
        return (
            1.0
            if _is_nine_alarm_on(
                ui_elements=ui_elements,
                current_activity=current_activity,
            )
            else 0.0
        )

    @classmethod
    def generate_random_params(cls) -> dict[str, str]:
        return {}


class TurnOffAlarm9AM(clock._ClockEval):
    complexity = 1
    schema = {
        "type": "object",
        "properties": {},
    }
    template = "Trun off alarm at 9:00 AM."

    def is_successful(
        self,
        env: interface.AsyncEnv,
    ) -> float:
        super().is_successful(env)
        ui_elements = env.get_state().ui_elements
        current_activity = adb_utils.get_current_activity(env.controller)[0]
        return (
            0.0
            if _is_nine_alarm_on(
                ui_elements=ui_elements,
                current_activity=current_activity,
            )
            else 1.0
        )

    @classmethod
    def generate_random_params(cls) -> dict[str, str]:
        return {}


class _SystemDnDToggle(task_eval.TaskEval):
  """Task for checking that Do not Disturb has been turned {on_or_off}."""
  app_names = ('settings',)
  complexity = 1
  schema = {
      'type': 'object',
      'properties': {'on_or_off': {'type': 'string', 'enum': ['on', 'off']}},
      'required': ['on_or_off'],
  }
  template = 'Turn {on_or_off} Do not Disturb.'

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'get', 'global', 'zen_mode'], env.controller
    )
    dnd_status = res.generic.output.decode().strip()

    if self.params['on_or_off'] == 'on':
      return 1.0 if dnd_status in ['1', '2', '3'] else 0.0
    else:
      return 1.0 if dnd_status == '0' else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on' if random.choice([True, False]) else 'off'}


class TurnOnDoNotDisturb(_SystemDnDToggle):
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'put', 'global', 'zen_mode', '0'], env.controller
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'on'}


class TurnOffDoNotDisturb(_SystemDnDToggle):
  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    res = adb_utils.issue_generic_request(
        ['shell', 'settings', 'put', 'global', 'zen_mode', '1'], env.controller
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, str]:
    return {'on_or_off': 'off'}

