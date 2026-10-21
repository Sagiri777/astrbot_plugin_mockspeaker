import asyncio
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

from conftest import At, Event, Plain, add_profile, request_for


async def results(generator):
    return [item async for item in generator]


async def test_collect_only_enabled_human_plain_messages(plugin):
    ignored = [
        Event(group="200"),
        Event(owner="bot"),
        Event(text="/命令 参数"),
        Event(text=" "),
        Event(text="模仿 开始 bob"),
        Event(owner="excluded"),
    ]
    plugin.settings = replace(plugin.settings, excluded_users=("excluded",))
    for event in ignored:
        await plugin.collect_messages(event)
    event = Event(owner="bob", text="真实文本", message_id="ok")
    event.parts = [At("bot"), Plain("真实文本"), SimpleNamespace(text="引用的假文字")]
    await plugin.collect_messages(event)
    await plugin.collect_messages(event)
    member = plugin.store.member(plugin._scope(event), "bob")
    assert member["total"] == 1
    assert plugin.store.samples(plugin._scope(event), "bob") == ["真实文本"]
    assert len(plugin.store.members(plugin._scope(event))) == 1


async def test_persona_replacement_isolation_pause_and_fact_delivery(plugin):
    scope = add_profile(plugin)
    event = Event()
    base = copy.deepcopy(plugin.context.conversation_manager.histories["base"])
    await plugin._select(event, "bob")
    req = await request_for(plugin, event)
    await plugin.prepare_request(event, req)
    assert req.conversation is None  # The real AstrBot history saver exits on this condition.
    assert "模仿专属风格" in req.system_prompt and "原人格" not in req.system_prompt
    assert "原人格预设对话" not in json.dumps(req.contexts, ensure_ascii=False)
    assert plugin.context.conversation_manager.histories["base"] == base
    await plugin.capture_response(event, SimpleNamespace(role="assistant", completion_text="老铁，周五19点见！"))
    assert plugin.context.conversation_manager.histories["base"] == base
    assert not plugin.store.notes(event.unified_msg_origin, "base")
    await plugin._stop(Event())
    assert plugin.store.selected(scope, "alice") is None
    restored = Event()
    original_req = await request_for(plugin, restored)
    await plugin.prepare_request(restored, original_req)
    assert original_req.conversation is not None and original_req.system_prompt == "原人格系统提示词"
    history = json.dumps(plugin.context.conversation_manager.histories["base"], ensure_ascii=False)
    assert "周五19点" in history and "老铁" not in history
    assert not plugin.store.notes(event.unified_msg_origin, "base")


async def test_different_people_use_different_personas_in_same_group(plugin):
    add_profile(plugin, "bob", prompt="风格B")
    add_profile(plugin, "dave", prompt="风格D")
    await plugin._select(Event(owner="alice"), "bob")
    await plugin._select(Event(owner="carol"), "dave")
    alice, carol = Event(owner="alice"), Event(owner="carol")
    req_a, req_c = await request_for(plugin, alice), await request_for(plugin, carol)
    await plugin.prepare_request(alice, req_a)
    await plugin.prepare_request(carol, req_c)
    assert "风格B" in req_a.system_prompt and "风格D" in req_c.system_prompt
    assert req_a.session_id != req_c.session_id
    regular = Event(owner="eve")
    req_e = await request_for(plugin, regular)
    await plugin.prepare_request(regular, req_e)
    assert req_e.system_prompt == "原人格系统提示词" and req_e.conversation


async def test_switch_does_not_reuse_other_target_raw_history(plugin):
    scope = add_profile(plugin, "bob")
    add_profile(plugin, "dave")
    first = Event()
    await plugin._select(first, "bob")
    req = await request_for(plugin, first)
    await plugin.prepare_request(first, req)
    sid = first.get_extra("mockspeaker_turn")[0]
    await plugin.capture_response(first, SimpleNamespace(role="assistant", completion_text="B的特殊口头禅"))
    await plugin._select(Event(), "dave")
    next_event = Event()
    next_req = await request_for(plugin, next_event)
    await plugin.prepare_request(next_event, next_req)
    assert plugin.store.session(sid)["closed"]
    assert plugin.store.selected(scope, "alice") == "dave"
    assert "B的特殊口头禅" not in json.dumps(next_req.contexts, ensure_ascii=False)


async def test_inflight_reply_after_pause_is_summarized_without_reactivating(plugin):
    scope = add_profile(plugin)
    event = Event()
    await plugin._select(event, "bob")
    req = await request_for(plugin, event)
    await plugin.prepare_request(event, req)
    await plugin._stop(Event())
    await plugin.capture_response(event, SimpleNamespace(role="assistant", completion_text="延迟返回的模仿原文"))
    assert plugin.store.selected(scope, "alice") is None
    assert len(plugin.store.notes(event.unified_msg_origin, "base")) == 1
    assert req.conversation is None


async def test_failed_seed_stops_request_without_original_persona_fallback(plugin):
    add_profile(plugin)
    event = Event()
    await plugin._select(event, "bob")
    plugin.context.fail = True
    req = await request_for(plugin, event)
    await plugin.prepare_request(event, req)
    assert event.stopped and req.conversation is None and req.contexts == []
    assert not plugin.context.conversation_manager.writes


async def test_reset_conversation_starts_fresh_segment(plugin):
    add_profile(plugin)
    event = Event()
    await plugin._select(event, "bob")
    req = await request_for(plugin, event)
    await plugin.prepare_request(event, req)
    old_sid = event.get_extra("mockspeaker_turn")[0]
    await plugin.capture_response(event, SimpleNamespace(role="assistant", completion_text="旧对话的原文"))
    plugin.context.conversation_manager.histories["new"] = []
    next_event = Event()
    new_req = await request_for(plugin, next_event, "new")
    await plugin.prepare_request(next_event, new_req)
    assert plugin.store.session(old_sid)["closed"]
    assert next_event.get_extra("mockspeaker_turn")[0] != old_sid
    assert new_req.contexts == []
    assert not plugin.store.notes(event.unified_msg_origin, "new")


async def test_tool_prompt_isolated_and_summarized_without_persistent_selection(plugin):
    scope = add_profile(plugin, prompt="绝不能污染原人格的模仿规则")
    event = Event()
    req = await request_for(plugin, event)
    await plugin.prepare_request(event, req)
    assert await plugin.tool_prompt(event, "bob") == "绝不能污染原人格的模仿规则"
    assert req.conversation is None
    assert plugin.store.selected(scope, "alice") is None
    await plugin.capture_response(event, SimpleNamespace(role="assistant", completion_text="使用提示词后的风格化回复"))
    normal = Event()
    original = await request_for(plugin, normal)
    await plugin.prepare_request(normal, original)
    data = json.dumps(plugin.context.conversation_manager.histories, ensure_ascii=False)
    assert "绝不能污染" not in data and "风格化回复" not in data
    assert original.system_prompt == "原人格系统提示词"


async def test_tool_selection_and_pause_only_affect_caller(plugin):
    scope = add_profile(plugin)
    plugin.store.select(scope, "carol", "bob")
    event = Event(owner="alice", text="后面一直模仿bob")
    req = await request_for(plugin, event)
    await plugin.prepare_request(event, req)
    assert "下一条" in await plugin.tool_select(event, "bob")
    assert req.conversation is None
    assert plugin.store.selected(scope, "alice") == "bob"
    pause = Event(owner="alice")
    pause_req = await request_for(plugin, pause)
    await plugin.prepare_request(pause, pause_req)
    await plugin.tool_pause(pause)
    assert plugin.store.selected(scope, "alice") is None
    assert plugin.store.selected(scope, "carol") == "bob"
    assert "未启用" in await plugin.tool_select(Event(group="200"), "bob")


async def test_deletion_cancels_related_choices_and_inflight_reply_cannot_resurrect(plugin):
    scope = add_profile(plugin)
    event = Event()
    await plugin._select(event, "bob")
    req = await request_for(plugin, event)
    await plugin.prepare_request(event, req)
    rejected = await results(plugin.forget_command(Event(owner="alice"), "bob"))
    assert "只能删除自己" in rejected[0]
    await results(plugin.forget_command(Event(owner="bob")))
    await plugin.capture_response(event, SimpleNamespace(role="assistant", completion_text="late reply"))
    assert plugin.store.selected(scope, "alice") is None
    assert not plugin.store.member(scope, "bob")
    assert not plugin.store.pending_sessions()


async def test_notes_are_idempotent_after_write_then_ack_failure(plugin, monkeypatch):
    add_profile(plugin)
    event = Event()
    session = plugin.store.create_session(
        plugin._scope(event), "alice", "bob", event.unified_msg_origin, "base", closed=True
    )
    await plugin.service.record(session["id"], "hi", "原始模仿回复")
    req = await request_for(plugin, event)
    original_ack = plugin.store.acknowledge

    def crash(ids):
        raise RuntimeError("simulated crash after successful conversation write")

    monkeypatch.setattr(plugin.store, "acknowledge", crash)
    await plugin.prepare_request(event, req)
    assert len(plugin.store.notes(event.unified_msg_origin, "base")) == 1
    monkeypatch.setattr(plugin.store, "acknowledge", original_ack)
    retry = Event()
    retry_req = await request_for(plugin, retry)
    await plugin.prepare_request(retry, retry_req)
    history = json.dumps(plugin.context.conversation_manager.histories["base"], ensure_ascii=False)
    assert history.count("[MockSpeaker 事实摘要") == 1
    assert not plugin.store.notes(event.unified_msg_origin, "base")


async def test_persona_sync_creates_then_updates(plugin):
    scope = add_profile(plugin)
    await plugin._sync_persona(scope, "bob", "第一版")
    await plugin._sync_persona(scope, "bob", "第二版")
    assert len(plugin.context.persona_manager.items) == 1
    assert "第二版" in next(iter(plugin.context.persona_manager.items.values()))


async def test_generate_permission_and_at_target(plugin):
    add_profile(plugin)
    denied = await results(plugin.generate_command(Event(), "bob"))
    assert "管理员" in denied[0]
    admin = Event(admin=True)
    admin.parts = [Plain("模仿 开始"), At("bot"), At("bob")]
    assert plugin._target(admin, "") == "bob"
    accepted = await results(plugin.generate_command(admin, "bob"))
    assert "已提交" in accepted[0]
    await asyncio.gather(*plugin.service.jobs.values())


async def test_disabling_group_restores_persona_and_flushes_remaining_facts(plugin):
    scope = add_profile(plugin)
    event = Event()
    await plugin._select(event, "bob")
    req = await request_for(plugin, event)
    await plugin.prepare_request(event, req)
    await plugin.capture_response(event, SimpleNamespace(role="assistant", completion_text="模仿原文"))
    plugin.settings = replace(plugin.settings, allowed_groups=())
    normal = Event()
    restored = await request_for(plugin, normal)
    await plugin.prepare_request(normal, restored)
    assert plugin.store.selected(scope, "alice") is None
    assert restored.conversation and restored.system_prompt == "原人格系统提示词"
    assert "周五19点" in json.dumps(restored.contexts, ensure_ascii=False)


async def test_duplicate_response_hook_does_not_duplicate_turn(plugin):
    add_profile(plugin)
    event = Event()
    await plugin._select(event, "bob")
    req = await request_for(plugin, event)
    await plugin.prepare_request(event, req)
    response = SimpleNamespace(role="assistant", completion_text="单次回复")
    await plugin.capture_response(event, response)
    await plugin.capture_response(event, response)
    assert len(plugin.store.turns(event.get_extra("mockspeaker_turn")[0])) == 1
