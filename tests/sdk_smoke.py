"""Run separately in an AstrBot environment: ASTRBOT_SOURCE=/path/to/AstrBot python tests/sdk_smoke.py.

Uses actual SDK registration, message objects, SQLite conversation/persona managers,
runner assembly, and core history saving. Only the model is replaced; no IM connection.
"""

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


async def verify():
    import astrbot
    from astrbot.api.event import AstrMessageEvent, MessageChain
    from astrbot.api.message_components import Plain
    from astrbot.api.provider import LLMResponse, ProviderRequest
    from astrbot.core import db_helper
    from astrbot.core.agent.hooks import BaseAgentRunHooks
    from astrbot.core.agent.message import Message
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    from astrbot.core.conversation_mgr import ConversationManager
    from astrbot.core.persona_mgr import PersonaManager
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import InternalAgentSubStage
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    from astrbot.core.platform_message_history_mgr import PlatformMessageHistoryManager
    from astrbot.core.provider.register import llm_tools

    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("_mockspeaker_sdk_test")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + ".main", root / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    names = {tool.name for tool in llm_tools.func_list}
    assert {"mockspeaker_list", "mockspeaker_get_prompt", "mockspeaker_select", "mockspeaker_pause"} <= names

    def event(group="100", sender="alice", platform="qq1"):
        message = AstrBotMessage()
        message.type, message.self_id, message.group_id = MessageType.GROUP_MESSAGE, "bot", group
        message.sender = MessageMember(sender, sender.title())
        message.message, message.message_id = [Plain("周五见面")], "test-message"
        message.message_str = "周五见面"
        return AstrMessageEvent(message.message_str, message, PlatformMetadata("test", "test", platform), group)

    await db_helper.initialize()
    cm = ConversationManager(db_helper)
    personas = PersonaManager(db_helper, SimpleNamespace(default_conf={}))
    await personas.initialize()

    async def generate(**kwargs):
        return LLMResponse(role="assistant", completion_text="用户和AI约定周五19点见面。")

    context = SimpleNamespace(
        conversation_manager=cm,
        persona_manager=personas,
        llm_generate=generate,
        message_history_manager=PlatformMessageHistoryManager(db_helper),
    )
    plugin = module.MockSpeakerPlugin(context, {"allowed_groups": ["100"], "min_messages": 1, "provider_id": "fake"})
    try:
        first = event()
        cid = await cm.new_conversation(first.unified_msg_origin, content=[])
        scope = plugin._scope(first)
        plugin.store.collect(scope, "bob", "Bob", "sample", "群友发言", 10)
        plugin.store.save_profile(scope, "bob", "专属群友人格", 1)
        await plugin._sync_persona(scope, "bob", "专属群友人格")
        await plugin._sync_persona(scope, "bob", "更新群友人格")
        await plugin._select(first, "bob")
        req = ProviderRequest(
            prompt=first.message_str,
            conversation=await cm.get_conversation(first.unified_msg_origin, cid),
            contexts=[{"role": "assistant", "content": "原人格预设"}],
            system_prompt="原人格",
        )
        await plugin.prepare_request(first, req)
        assert not first.is_stopped() and req.conversation is None
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider=SimpleNamespace(provider_config={"id": "fake"}),
            request=req,
            run_context=ContextWrapper(context=None),
            tool_executor=None,
            agent_hooks=BaseAgentRunHooks(),
        )
        assert "专属群友人格" in runner.run_context.messages[0].content
        assert "原人格预设" not in str(runner.run_context.messages)
        response = LLMResponse(role="assistant", completion_text="老铁周五19点见！")
        runner.run_context.messages.append(Message(role="assistant", content=response.completion_text))
        await plugin.capture_response(first, response)
        saver = object.__new__(InternalAgentSubStage)
        saver.conv_manager = cm
        await saver._save_to_history(first, req, response, runner.run_context.messages, None)
        assert json.loads((await cm.get_conversation(first.unified_msg_origin, cid)).history) == []
        await plugin._stop(first)
        restored = event()
        original = ProviderRequest(
            prompt="地点呢？",
            conversation=await cm.get_conversation(restored.unified_msg_origin, cid),
            system_prompt="原人格",
        )
        await plugin.prepare_request(restored, original)
        assert original.system_prompt == "原人格" and original.conversation
        history = json.dumps(
            json.loads((await cm.get_conversation(restored.unified_msg_origin, cid)).history), ensure_ascii=False
        )
        assert "周五19点" in history and "老铁" not in history

        # Reload against existing collection: the second group completes the shared threshold.
        await plugin.terminate()
        plugin = module.MockSpeakerPlugin(
            context,
            {
                "allowed_groups": ["100", "200"],
                "cross_group_profiles": True,
                "min_messages": 2,
                "provider_id": "fake",
                "sync_personas": True,
            },
        )
        await plugin.collect_messages(event(group="200", sender="bob"))
        await asyncio.gather(*plugin.service.jobs.values())
        second = event(group="200")
        second_scope = plugin._scope(second)
        merged = plugin.service.ready_profile(scope, "bob")
        assert merged["total"] == merged["generated_at_count"] == 2
        assert plugin.service.ready_profile(second_scope, "bob")["prompt"] == merged["prompt"]
        assert plugin.store.member(scope, "bob")["total"] == 1
        assert json.loads(await plugin.tool_list(second))[0]["messages"] == 2
        shared_key = plugin.service.profile_key(scope)
        persona_id = "mockspeaker_" + hashlib.sha256(f"{shared_key}:bob".encode()).hexdigest()[:24]
        assert merged["prompt"] in (await personas.get_persona(persona_id)).system_prompt
        cid_second = await cm.new_conversation(second.unified_msg_origin, content=[])
        await plugin._select(second, "bob")
        merged_request = ProviderRequest(
            prompt=second.message_str,
            conversation=await cm.get_conversation(second.unified_msg_origin, cid_second),
            system_prompt="原人格",
        )
        await plugin.prepare_request(second, merged_request)
        assert not second.is_stopped() and merged_request.conversation is None
        assert merged["prompt"] in merged_request.system_prompt
        assert plugin.store.selected(scope, "alice") is None
        assert [reply async for reply in plugin.forget_command(event(group="200", sender="bob"))]
        assert plugin.store.shared_profile(shared_key, "bob") is None
        assert plugin.store.member(scope, "bob")["total"] == 1
        try:
            await personas.get_persona(persona_id)
        except ValueError:
            pass
        else:
            raise AssertionError("Deleting a source group must remove the exported shared persona")

        # Use the SDK's registered command filter, including default arguments and alias.
        command = next(f for f in plugin.mimic.parent_group.sub_command_filters if f.command_name == "导入")
        historical = event()
        historical.role, historical.is_at_or_wake_command = "admin", True
        historical.message_str = "模仿 导入"
        assert command.filter(historical, {})
        assert historical.get_extra("parsed_params") == {"count": 1000, "target": ""}
        historical.message_str = "mockspeaker import 2 bob"
        assert command.filter(historical, {})
        assert historical.get_extra("parsed_params") == {"count": 2, "target": "bob"}

        async def history_action(action, **params):
            assert action == "get_group_msg_history" and params["group_id"] == 100
            return {
                "messages": [
                    {
                        "message_id": f"history-{i}",
                        "time": i,
                        "group_id": 100,
                        "sender": {"user_id": "bob", "nickname": "Bob"},
                        "message": [{"type": "text", "data": {"text": f"历史群聊发言{i}"}}],
                    }
                    for i in (1, 2)
                ]
            }

        historical.bot = SimpleNamespace(api=SimpleNamespace(call_action=history_action))
        replies = [r async for r in plugin.import_command(historical, **historical.get_extra("parsed_params"))]
        assert "新增 2 条" in "".join(part.text for reply in replies for part in reply.chain if isinstance(part, Plain))
        await asyncio.gather(*plugin.service.jobs.values())
        assert plugin.service.ready_profile(scope, "bob")["generated_at_count"] == 3

        # Official QQ uses nonnumeric OpenIDs and the core's saved platform history.
        saved_event = event(group="group-openid", platform="official-instance")
        saved_event.role = "admin"
        plugin.settings = replace(plugin.settings, allowed_groups=(*plugin.settings.allowed_groups, "group-openid"))
        plugin.service.settings = plugin.settings
        first_row = None
        for index, role in enumerate(("user", "user", "bot"), 1):
            row = await context.message_history_manager.insert_message_chain(
                platform_id=saved_event.get_platform_id(),
                user_id=saved_event.unified_msg_origin,
                message_chain=MessageChain([Plain(f"本地群聊发言{index}")]),
                role=role,
                sender_id="member-openid",
                sender_name="群友",
            )
            first_row = first_row or row
        live = event(group="group-openid", sender="member-openid", platform="official-instance")
        live.set_extra("_current_platform_message_history_id", first_row.id)
        await plugin.collect_messages(live)
        replies = [r async for r in plugin.import_command(saved_event)]
        saved_reply = "".join(part.text for reply in replies for part in reply.chain if isinstance(part, Plain))
        assert "AstrBot 本地群历史" in saved_reply and "读取 3 条，新增 1 条" in saved_reply
        await asyncio.gather(*plugin.service.jobs.values())
        assert plugin.service.ready_profile(plugin._scope(saved_event), "member-openid")["generated_at_count"] == 2
        print(
            f"AstrBot {astrbot.__version__}: registration, persona sync, runner assembly, history isolation, "
            "summary restoration, cross-group profiles, OneBot and saved group history import passed"
        )
    finally:
        await plugin.terminate()
        await db_helper.engine.dispose()


if __name__ == "__main__":
    if source := os.environ.get("ASTRBOT_SOURCE"):
        sys.path.insert(0, str(Path(source).resolve()))
    with tempfile.TemporaryDirectory(prefix="mockspeaker-sdk-") as runtime:
        os.environ["ASTRBOT_ROOT"] = runtime
        asyncio.run(verify())
