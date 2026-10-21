import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from conftest import Event, add_profile, request_for

from mockspeaker.config import Settings
from mockspeaker.service import MockSpeakerService
from mockspeaker.store import Store, scope_key

A = scope_key("qq1", "100")
B = scope_key("qq1", "200")
C = scope_key("qq1", "300")


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "cross-group.sqlite3")
    yield result
    result.close()


def collect(store, scope, count=1, start=0, user="bob", text=None):
    for i in range(start, start + count):
        store.collect(scope, user, user, str(i), text or f"{json.loads(scope)[1]}的发言{i}", 300)


def settings(**kwargs):
    return Settings(
        cross_group_profiles=True, allowed_groups=("100", "qq1:200"), min_messages=2, update_every=2, **kwargs
    )


async def persona(*args):
    return "跨群共同风格"


def test_cross_group_merging_is_disabled_by_default(store):
    collect(store, A)
    collect(store, B)
    service = MockSpeakerService(store, Settings(min_messages=2), persona)
    assert not service.settings.cross_group_profiles
    assert service.profile_member(A, "bob")["total"] == 1
    assert service.profile_member(B, "bob")["total"] == 1
    assert service.profile_key(A) != service.profile_key(B)
    assert not service.schedule_profile(A, "bob", "origin")


async def test_combined_threshold_uses_only_enabled_groups_on_same_platform(store):
    calls = []

    async def generate(origin, prompt, system):
        calls.append(json.loads(prompt))
        return "合并人格"

    collect(store, A)
    collect(store, B)
    collect(store, C, 10, text="已关闭群的语料")
    collect(store, scope_key("qq2", "200"), 10, text="其他平台实例的语料")
    collect(store, A, user="eve", start=100, text="其他用户的语料")
    service = MockSpeakerService(store, settings(), generate)
    assert service.profile_member(A, "bob")["total"] == 2
    assert service.schedule_profile(A, "bob", "origin")
    await asyncio.gather(*service.jobs.values())
    assert {sample["group_id"] for sample in calls[0]["messages"]} == {"100", "200"}
    serialized = json.dumps(calls, ensure_ascii=False)
    assert "已关闭群" not in serialized and "其他平台" not in serialized and "其他用户" not in serialized
    assert service.ready_profile(A, "bob")["prompt"] == service.ready_profile(B, "bob")["prompt"] == "合并人格"
    assert store.member(A, "bob")["total"] == store.member(B, "bob")["total"] == 1


async def test_cross_group_update_watermark_and_inflight_deduplication(store):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def generate(origin, prompt, system):
        calls.append(json.loads(prompt))
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return f"共同风格{len(calls)}"

    collect(store, A)
    collect(store, B)
    service = MockSpeakerService(store, settings(), generate)
    service.schedule_profile(A, "bob", "origin-a")
    await entered.wait()
    collect(store, B, 2, start=1)
    assert not service.schedule_profile(B, "bob", "origin-b")
    release.set()
    await asyncio.gather(*service.jobs.values())
    assert len(calls) == 2
    assert service.ready_profile(A, "bob")["generated_at_count"] == 4
    assert service.ready_profile(B, "bob")["generated_at_count"] == 4
    collect(store, A, start=1)
    assert not service.schedule_profile(A, "bob", "origin-a")
    collect(store, B, start=3)
    assert service.schedule_profile(B, "bob", "origin-b")
    await asyncio.gather(*service.jobs.values())
    assert service.ready_profile(A, "bob")["generated_at_count"] == 6


def test_cross_group_sampling_considers_quiet_groups_within_budget(store):
    collect(store, A, 20, text="甲" * 500)
    collect(store, B, text="乙" * 500)
    service = MockSpeakerService(store, settings(analysis_chars=2000), persona)
    samples = service.profile_samples(A, "bob")
    assert {sample["group_id"] for sample in samples} == {"100", "200"}
    assert sum(len(sample["text"]) for sample in samples) <= 2000


async def test_changing_whitelist_does_not_reuse_persona_from_removed_group(store):
    for scope in (A, B, C):
        collect(store, scope)
    service = MockSpeakerService(store, replace(settings(), allowed_groups=("100", "200", "300")), persona)
    await service.refresh_profile(A, "bob", "origin")
    previous_key = service.profile_key(A)
    service.settings = settings()
    assert service.profile_key(A) != previous_key
    assert service.profile_member(A, "bob")["total"] == 2
    with pytest.raises(ValueError, match="尚未生成"):
        service.ready_profile(A, "bob")
    assert service.profile_member(C, "bob") is None
    assert {sample["group_id"] for sample in service.profile_samples(A, "bob")} == {"100", "200"}
    # Reordering or repeating the same sources preserves the shared persona identity.
    key = service.profile_key(A)
    service.settings = replace(settings(), allowed_groups=("qq1:200", "100", "100", "qq2:999"))
    assert service.profile_key(B) == key


async def test_switching_off_restores_local_profiles_and_reuses_collected_data(store):
    collect(store, A, 2)
    collect(store, B, 2)
    store.save_profile(A, "bob", "本群原人格A", 2)
    store.save_profile(B, "bob", "本群原人格B", 2)
    service = MockSpeakerService(store, settings(), persona)
    with pytest.raises(ValueError, match="尚未生成"):
        service.ready_profile(A, "bob")
    await service.refresh_profile(A, "bob", "origin")
    assert service.ready_profile(A, "bob")["total"] == 4
    service.settings = replace(settings(), cross_group_profiles=False)
    assert service.ready_profile(A, "bob")["prompt"] == "本群原人格A"
    assert service.ready_profile(B, "bob")["prompt"] == "本群原人格B"
    service.settings = settings()
    assert service.ready_profile(B, "bob")["prompt"] == "跨群共同风格"


async def test_shared_generation_failure_and_retry_are_shared_across_groups(store):
    collect(store, A)
    collect(store, B)
    service = MockSpeakerService(store, settings(), persona)
    await service.refresh_profile(A, "bob", "origin")
    collect(store, B, 2, start=1)

    async def fail(*args):
        raise RuntimeError("offline")

    service.generate = fail
    with pytest.raises(RuntimeError):
        await service.refresh_profile(B, "bob", "origin")
    assert service.ready_profile(A, "bob")["prompt"] == "跨群共同风格"
    assert service.profile_member(A, "bob")["error"] == "RuntimeError"
    assert not service.schedule_profile(A, "bob", "origin")
    service.generate = persona
    assert service.schedule_profile(A, "bob", "origin", force=True)
    await asyncio.gather(*service.jobs.values())
    assert service.ready_profile(B, "bob")["generated_at_count"] == 4


async def test_forgetting_one_group_invalidates_shared_profiles_but_retains_other_samples(store):
    collect(store, A)
    collect(store, B, 2)
    store.save_profile(B, "bob", "保留的本群人格B", 2)
    service = MockSpeakerService(store, settings(), persona)
    await service.refresh_profile(A, "bob", "origin")
    key = service.profile_key(A)
    invalidated = store.forget(A, "bob")
    assert key in invalidated and store.shared_profile(key, "bob") is None
    assert store.member(A, "bob") is None
    assert len(store.samples(B, "bob")) == 2
    assert store.member(B, "bob")["prompt"] == "保留的本群人格B"
    assert service.schedule_profile(B, "bob", "origin")
    await asyncio.gather(*service.jobs.values())
    assert service.ready_profile(B, "bob")["total"] == 2
    assert service.profile_member(A, "bob") is None


async def test_shared_profiles_survive_restart_and_database_upgrade(tmp_path):
    path = tmp_path / "upgrade.sqlite3"
    store = Store(path)
    collect(store, A)
    collect(store, B)
    store.select(A, "alice", "bob")
    with store.db:
        store.db.execute("DROP TABLE shared_profiles")
        store.db.execute("PRAGMA user_version=1")
    store.close()
    store = Store(path)
    service = MockSpeakerService(store, settings(), persona)
    await service.refresh_profile(A, "bob", "origin")
    store.close()
    store = Store(path)
    service = MockSpeakerService(store, settings(), persona)
    assert service.ready_profile(B, "bob")["generated_at_count"] == 2
    assert store.selected(A, "alice") == "bob"
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == 2
    store.close()


def enable_plugin_merging(plugin, sync=False):
    plugin.settings = replace(
        plugin.settings, allowed_groups=("100", "200"), cross_group_profiles=True, sync_personas=sync
    )
    plugin.service.settings = plugin.settings


async def test_commands_tools_and_chat_use_shared_profile_but_selections_and_history_stay_separate(plugin):
    add_profile(plugin, count=1)
    add_profile(plugin, group="200", count=1)
    enable_plugin_merging(plugin)
    plugin.service.generate = persona
    await plugin.service.refresh_profile(A, "bob", "origin")
    first, second = Event(group="100"), Event(group="200")
    await plugin._select(first, "bob")
    await plugin._select(second, "bob")
    req_a, req_b = await request_for(plugin, first), await request_for(plugin, second)
    await plugin.prepare_request(first, req_a)
    await plugin.prepare_request(second, req_b)
    assert req_a.conversation is None and req_b.conversation is None
    assert req_a.system_prompt == req_b.system_prompt
    assert req_a.session_id != req_b.session_id
    assert "跨群共同风格" in req_a.system_prompt
    assert await plugin.tool_prompt(second, "bob") == "跨群共同风格"
    assert json.loads(await plugin.tool_list(first))[0]["messages"] == 2
    lines = [line async for line in plugin.list_command(first)]
    assert "跨群合并" in lines[0] and "2 条，可模仿" in lines[0]
    status = [line async for line in plugin.status_command(second, "bob")]
    assert "有效消息 2 条" in status[0]
    await plugin.capture_response(first, SimpleNamespace(role="assistant", completion_text="仅群100的原文"))
    assert "仅群100的原文" not in json.dumps(req_b.contexts, ensure_ascii=False)
    await plugin._stop(first)
    assert plugin.store.selected(A, "alice") is None
    assert plugin.store.selected(B, "alice") == "bob"


async def test_merged_profile_does_not_list_or_select_users_never_observed_in_current_group(plugin):
    add_profile(plugin, group="200")
    enable_plugin_merging(plugin)
    await plugin.service.refresh_profile(B, "bob", "origin")
    assert json.loads(await plugin.tool_list(Event(group="100"))) == []
    with pytest.raises(ValueError, match="有效消息数为 0"):
        await plugin._select(Event(group="100"), "bob")


async def test_cross_group_persona_export_is_shared_and_invalidated_when_source_is_deleted(plugin):
    add_profile(plugin, count=1)
    add_profile(plugin, group="200", count=1)
    enable_plugin_merging(plugin, sync=True)
    await plugin.service.refresh_profile(A, "bob", "origin")
    await plugin.service.refresh_profile(B, "bob", "origin", force=True)
    assert len(plugin.context.persona_manager.items) == 1
    deleted = [line async for line in plugin.forget_command(Event(owner="bob", group="100"))]
    assert "合并人格已失效" in deleted[0]
    assert not plugin.context.persona_manager.items
    assert plugin.store.member(B, "bob") is not None


async def test_manual_refresh_accepts_cross_group_total_below_individual_threshold(plugin):
    add_profile(plugin, count=1)
    add_profile(plugin, group="200", count=1)
    enable_plugin_merging(plugin)
    response = [line async for line in plugin.generate_command(Event(admin=True), "bob")]
    assert "已提交" in response[0]
    await asyncio.gather(*plugin.service.jobs.values())
    assert plugin.service.ready_profile(B, "bob")["generated_at_count"] == 2
