"""AstrBot API doubles keep logic tests independent of IM adapters and live providers."""

import copy
import importlib.util
import json
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


@dataclass
class Plain:
    text: str


@dataclass
class At:
    qq: str


class Event:
    def __init__(self, owner="alice", group="100", platform="qq1", text="周五几点见？", message_id="1", admin=False):
        self.owner, self.group, self.platform = owner, group, platform
        self.message_str = text
        self.parts = [Plain(text)]
        self.message_obj = SimpleNamespace(message_id=message_id)
        self.unified_msg_origin = f"{platform}:GroupMessage:{group}"
        self.extra, self.sent = {}, []
        self.admin, self.stopped = admin, False

    def get_group_id(self):
        return self.group

    def get_platform_id(self):
        return self.platform

    def get_sender_id(self):
        return self.owner

    def get_sender_name(self):
        return self.owner

    def get_self_id(self):
        return "bot"

    def get_messages(self):
        return self.parts

    def is_admin(self):
        return self.admin

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key, value):
        self.extra[key] = value

    def stop_event(self):
        self.stopped = True

    def plain_result(self, text):
        return text

    async def send(self, result):
        self.sent.append(result)


class Conversations:
    def __init__(self):
        self.histories = {
            "base": [
                {"role": "user", "content": "原话题：周五聚餐。"},
                {"role": "assistant", "content": "原人格的回复。"},
            ]
        }
        self.writes = []

    async def get_conversation(self, origin, cid):
        if cid not in self.histories:
            return None
        return SimpleNamespace(
            cid=cid, history=json.dumps(self.histories[cid], ensure_ascii=False), persona_id="original"
        )

    async def update_conversation(self, origin, cid, history):
        self.histories[cid] = copy.deepcopy(history)
        self.writes.append((origin, cid, copy.deepcopy(history)))


class Personas:
    def __init__(self):
        self.items = {}

    async def get_persona(self, persona_id):
        if persona_id not in self.items:
            raise ValueError("missing")  # The real API raises for nonexistent personas.
        return self.items[persona_id]

    async def create_persona(self, persona_id, system_prompt, begin_dialogs):
        self.items[persona_id] = system_prompt

    async def update_persona(self, persona_id, system_prompt, begin_dialogs):
        self.items[persona_id] = system_prompt

    async def delete_persona(self, persona_id):
        del self.items[persona_id]


class Context:
    def __init__(self):
        self.conversation_manager = Conversations()
        self.persona_manager = Personas()
        self.calls = []
        self.fail = False

    async def get_current_chat_provider_id(self, origin):
        return "test-provider"

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("provider unavailable")
        text = (
            "说话简短，偏好自然口语。"
            if "语言风格分析员" in kwargs["system_prompt"]
            else "用户与AI讨论聚餐，决定周五19点见面；地点尚未确定。"
        )
        return SimpleNamespace(role="assistant", completion_text=text)


@pytest.fixture(scope="session")
def plugin_class():
    patch = pytest.MonkeyPatch()

    def decorator(*args, **kwargs):
        def wrap(function):
            function.command = decorator
            return function

        return wrap

    class Star:
        def __init__(self, context):
            self.context = context

    class StarTools:
        data_dir = None

        @classmethod
        def get_data_dir(cls, name):
            return cls.data_dir

    modules = {
        "astrbot": {},
        "astrbot.api": {"AstrBotConfig": dict, "logger": logging.getLogger("astrbot-test")},
        "astrbot.api.event": {
            "AstrMessageEvent": Event,
            "filter": SimpleNamespace(
                command_group=decorator,
                event_message_type=decorator,
                on_llm_request=decorator,
                on_llm_response=decorator,
                llm_tool=decorator,
                EventMessageType=SimpleNamespace(GROUP_MESSAGE=1),
            ),
        },
        "astrbot.api.message_components": {"At": At, "Plain": Plain},
        "astrbot.api.provider": {"LLMResponse": SimpleNamespace, "ProviderRequest": SimpleNamespace},
        "astrbot.api.star": {"Context": Context, "Star": Star, "StarTools": StarTools, "register": decorator},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        patch.setitem(sys.modules, name, module)
    root = Path(__file__).parents[1]
    package = types.ModuleType("mockspeaker_test_plugin")
    package.__path__ = [str(root)]
    patch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location("mockspeaker_test_plugin.main", root / "main.py")
    module = importlib.util.module_from_spec(spec)
    patch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    yield module.MockSpeakerPlugin, StarTools
    patch.undo()


@pytest.fixture
def plugin(plugin_class, tmp_path):
    cls, star_tools = plugin_class
    star_tools.data_dir = tmp_path
    instance = cls(Context(), {"allowed_groups": ["100"], "min_messages": 2, "update_every": 2})
    yield instance
    instance.store.close()


async def request_for(plugin, event, cid="base"):
    conv = await plugin.context.conversation_manager.get_conversation(event.unified_msg_origin, cid)
    return SimpleNamespace(
        conversation=conv,
        contexts=[{"role": "assistant", "content": "原人格预设对话"}] + json.loads(conv.history),
        system_prompt="原人格系统提示词",
        prompt=event.message_str,
        tool_calls_result=None,
        session_id=event.unified_msg_origin,
    )


def add_profile(plugin, target="bob", group="100", platform="qq1", prompt="模仿专属风格", count=2):
    from mockspeaker.store import scope_key

    scope = scope_key(platform, group)
    for i in range(count):
        plugin.store.collect(scope, target, target, f"{target}-{i}", "群友原始发言", 300)
    plugin.store.save_profile(scope, target, prompt, count)
    return scope
