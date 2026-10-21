import asyncio
import copy
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from conftest import At, Event, Plain, add_profile

from mockspeaker.history import fetch_history, plain_text
from mockspeaker.store import Store, scope_key


def message(number, user="bob", text=None, **overrides):
    return {
        "message_id": str(number),
        "message_seq": number + 100,
        "time": number,
        "group_id": 100,
        "sender": {"user_id": user, "card": "群名片", "nickname": "昵称"},
        "message": [{"type": "text", "data": {"text": text or f"历史发言{number}"}}],
        **overrides,
    }


class History:
    """Chronological pages with an inclusive cursor and an adapter page-size cap."""

    def __init__(self, messages, page_size=100, wrapped=False):
        self.messages, self.page_size, self.wrapped = messages, page_size, wrapped
        self.calls = []
        self.api = SimpleNamespace(call_action=self.call_action)

    async def call_action(self, action, **params):
        self.calls.append((action, params))
        end = len(self.messages)
        if "message_seq" in params:
            end = next(
                i + 1
                for i, item in enumerate(self.messages)
                if str(item.get("message_seq", item["message_id"])) == str(params["message_seq"])
            )
        count = min(self.page_size, params["count"])
        data = {"messages": copy.deepcopy(self.messages[max(0, end - count) : end])}
        return {"status": "ok", "retcode": 0, "data": data} if self.wrapped else data


async def imported(plugin, event, count=1000, target=""):
    replies = [reply async for reply in plugin.import_command(event, count, target)]
    await asyncio.gather(*plugin.service.jobs.values())
    return "\n".join(replies)


@pytest.mark.parametrize(
    ("event", "count", "expected"),
    [
        (Event(), 10, "管理员"),
        (Event(group="200", admin=True), 10, "未启用"),
        (Event(group="", admin=True), 10, "未启用"),
        (Event(admin=True), 0, "1 到 10000"),
        (Event(admin=True), 10001, "1 到 10000"),
        (Event(admin=True), 10, "不支持群历史导入"),
    ],
)
async def test_history_requires_enabled_group_admin_valid_count_and_supported_adapter(plugin, event, count, expected):
    assert expected in await imported(plugin, event, count)
    assert not plugin.store.members(scope_key("qq1", "100"))
    assert not plugin.history_imports


async def test_history_filters_and_generates_once_after_all_pages(plugin):
    plugin.settings = replace(plugin.settings, excluded_users=("excluded",))
    plugin.service.settings = plugin.settings
    messages = [
        message(1),
        message(2, text="/命令 参数"),
        message(3, text="模仿 开始 bob"),
        message(4, text="x"),
        message(5, user="bot"),
        message(6, user="excluded"),
        message(7, group_id=200),
        message(8, message=[{"type": "forward", "data": {"text": "转发内部文字"}}]),
        message(9, user="carol"),
        message(
            10,
            message=[
                {"type": "reply", "data": {"text": "引用内部文字"}},
                {"type": "text", "data": {"text": "真实"}},
                {"type": "image", "data": {"text": "图片说明"}},
                {"type": "text", "data": {"text": "文字"}},
            ],
        ),
    ]
    event = Event(admin=True)
    event.bot = History(messages, page_size=4, wrapped=True)
    reply = await imported(plugin, event, count=10)
    scope = plugin._scope(event)
    assert "读取 10 条，新增 3 条有效语料，涉及 2 位群友" in reply
    assert plugin.store.samples(scope, "bob") == ["历史发言1", "真实文字"]
    assert plugin.service.ready_profile(scope, "bob")["generated_at_count"] == 2
    assert len(plugin.context.calls) == 1
    assert {m["user_id"] for m in plugin.store.members(scope)} == {"bob", "carol"}
    assert len(event.bot.calls) > 1
    assert all(action == "get_group_msg_history" for action, _ in event.bot.calls)
    assert event.bot.calls[1][1]["message_seq"] == messages[6]["message_seq"]
    assert "新增 0 条" in await imported(plugin, event, count=10)
    assert len(plugin.context.calls) == 1


async def test_history_deduplicates_live_messages_and_keeps_latest_samples_across_restart(plugin):
    plugin.settings = replace(plugin.settings, max_samples=10)
    plugin.service.settings = plugin.settings
    live = Event(owner="bob", message_id="31", text="最新实时消息")
    live.message_obj.timestamp = 31
    await plugin.collect_messages(live)
    event = Event(admin=True)
    event.bot = History([message(i) for i in range(1, 32)], page_size=7)
    assert "新增 30 条" in await imported(plugin, event, count=31)
    scope = plugin._scope(event)
    assert plugin.store.member(scope, "bob")["total"] == 31
    assert plugin.store.samples(scope, "bob") == [f"历史发言{i}" for i in range(22, 31)] + ["最新实时消息"]
    path = plugin.store.db.execute("PRAGMA database_list").fetchone()[2]
    plugin.store.close()
    plugin.store = Store(path)
    plugin.service.store = plugin.store
    assert "新增 0 条" in await imported(plugin, event, count=31)
    await plugin.collect_messages(live)
    assert plugin.store.member(scope, "bob")["total"] == 31
    # Backfilled old records increase totals without replacing newer raw samples.
    event.bot = History([message(0)])
    assert "新增 1 条" in await imported(plugin, event, count=1)
    assert plugin.store.samples(scope, "bob")[-1] == "最新实时消息"
    assert plugin.store.member(scope, "bob")["total"] == 32


@pytest.mark.parametrize("at_target", [False, True])
async def test_history_can_limit_collection_to_one_member(plugin, at_target):
    event = Event(admin=True)
    event.bot = History([message(1), message(2, user="carol"), message(3)])
    if at_target:
        event.parts = [Plain("模仿 导入 3"), At("bot"), At("bob")]
    reply = await imported(plugin, event, count=3, target="" if at_target else "bob")
    assert "新增 2 条" in reply
    assert plugin.store.member(plugin._scope(event), "carol") is None
    assert plugin.service.ready_profile(plugin._scope(event), "bob")["total"] == 2


async def test_history_respects_exclusion_before_reading_and_text_limit(plugin):
    plugin.settings = replace(plugin.settings, excluded_users=("bob",), max_message_chars=20)
    event = Event(admin=True)
    event.bot = History([message(1, user="carol", text="字" * 100)])
    assert "已被排除" in await imported(plugin, event, target="bob")
    assert not event.bot.calls
    await imported(plugin, event, count=1)
    assert plugin.store.samples(plugin._scope(event), "carol") == ["字" * 20]


async def test_history_counts_toward_cross_group_persona_threshold(plugin):
    add_profile(plugin, group="200", count=1)
    plugin.settings = replace(plugin.settings, allowed_groups=("100", "200"), cross_group_profiles=True, min_messages=3)
    plugin.service.settings = plugin.settings
    event = Event(admin=True)
    event.bot = History([message(1), message(2)])
    await imported(plugin, event, count=2)
    assert plugin.store.member(plugin._scope(event), "bob")["total"] == 2
    assert plugin.service.ready_profile(plugin._scope(event), "bob")["total"] == 3
    assert plugin.service.ready_profile(scope_key("qq1", "200"), "bob")["generated_at_count"] == 3


async def test_history_handles_shared_timestamps_and_adapters_ignoring_page_size():
    event = Event(admin=True)
    event.bot = History([message(i, time=1) for i in range(1, 10)], page_size=3)
    messages, notice = await fetch_history(event, 7, 1)
    assert [m["message_id"] for m in messages] == [str(i) for i in range(3, 10)]
    assert not notice
    assert event.bot.calls[1][1]["message_seq"] == 107
    assert event.bot.calls[2][1]["message_seq"] == 105


async def test_history_stops_when_adapter_repeats_same_page(plugin):
    calls = []

    async def action(*args, **kwargs):
        calls.append(kwargs)
        return {"messages": [message(1), message(2)]}

    event = Event(admin=True)
    event.bot = SimpleNamespace(call_action=action)
    reply = await imported(plugin, event)
    assert "新增 2 条" in reply and "未返回更早" in reply
    assert len(calls) == 2
    assert plugin.store.member(plugin._scope(event), "bob")["total"] == 2


@pytest.mark.parametrize("response", [None, {}, {"retcode": 1}, {"messages": "invalid"}])
async def test_history_rejects_invalid_or_failed_responses_without_writes(plugin, response):
    async def action(*args, **kwargs):
        return response

    event = Event(admin=True)
    event.bot = SimpleNamespace(call_action=action)
    assert "历史接口读取失败" in await imported(plugin, event)
    assert not plugin.store.members(plugin._scope(event))
    assert not plugin.history_imports


async def test_history_keeps_retrieved_pages_on_later_failure_without_leaking_adapter_error(plugin):
    calls = 0

    async def action(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("credential-not-for-users")
        return {"messages": [message(1), message(2)]}

    event = Event(admin=True)
    event.bot = SimpleNamespace(call_action=action)
    reply = await imported(plugin, event)
    assert "新增 2 条" in reply and "历史接口读取失败（RuntimeError）" in reply
    assert "credential-not-for-users" not in reply
    assert plugin.service.ready_profile(plugin._scope(event), "bob")["total"] == 2


async def test_history_times_out_and_releases_group_import_guard(plugin):
    async def action(*args, **kwargs):
        await asyncio.Future()

    plugin.settings = replace(plugin.settings, timeout=0.01)
    event = Event(admin=True)
    event.bot = SimpleNamespace(call_action=action)
    assert "TimeoutError" in await imported(plugin, event)
    assert not plugin.history_imports


async def test_only_one_import_per_group_and_cancellation_releases_guard(plugin):
    event = Event(admin=True)
    event.bot = History([message(1)])
    running = plugin.import_command(event)
    assert "正在读取" in await anext(running)
    assert "已有历史导入" in await imported(plugin, event)
    await running.aclose()
    assert not plugin.history_imports
    assert "新增 1 条" in await imported(plugin, event, count=1)


def test_cq_text_excludes_components_and_preserves_escaped_literals():
    text = "[CQ:reply,id=1]文字[CQ:image,file=x] &amp; &#91;CQ:face&#44;id=1&#93;"
    assert plain_text(text) == "文字 & [CQ:face,id=1]"
    assert plain_text(None) == ""


def test_legacy_database_upgrade_backfills_dedup_and_forget_allows_recollection(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, user_id TEXT NOT NULL,
            message_id TEXT, text TEXT NOT NULL, UNIQUE(scope,message_id));
        INSERT INTO samples(scope,user_id,message_id,text) VALUES ('group','bob','one','旧语料1');
        INSERT INTO samples(scope,user_id,message_id,text) VALUES ('group','bob','two','旧语料2');
        PRAGMA user_version=2;
        """)
    store = Store(path)
    try:
        assert store.samples("group", "bob") == ["旧语料1", "旧语料2"]
        assert not store.collect("group", "bob", "Bob", "one", "重复", 10)
        assert store.db.execute("PRAGMA user_version").fetchone()[0] == 3
        store.forget("group", "bob")
        assert store.collect("group", "bob", "Bob", "one", "重新采集", 10)
        assert store.member("group", "bob")["total"] == 1
    finally:
        store.close()


def official_event(plugin):
    event = Event(group="group-openid", platform="qq_official_instance", admin=True)
    event.bot = SimpleNamespace(api=SimpleNamespace())
    plugin.settings = replace(plugin.settings, allowed_groups=(event.group,))
    plugin.service.settings = plugin.settings
    return event


def saved_message(event, number, user="member-openid", **overrides):
    return SimpleNamespace(
        **{
            "id": number,
            "platform_id": event.get_platform_id(),
            "user_id": event.unified_msg_origin,
            "sender_id": user,
            "sender_name": "群友",
            "created_at": datetime.fromtimestamp(number, timezone.utc).replace(tzinfo=None),
            "content": {"type": "user", "message": [{"type": "plain", "text": f"已保存的发言{number}"}]},
            **overrides,
        }
    )


class SavedHistory:
    def __init__(self, records):
        self.records, self.calls = records, []

    async def get(self, **params):
        self.calls.append(params)
        return self.records[-params["page_size"] :]


async def test_qq_official_imports_saved_group_messages_and_filters_bot_media_and_quotes(plugin):
    event = official_event(plugin)
    records = [
        saved_message(event, 1),
        saved_message(
            event,
            2,
            content={
                "type": "user",
                "message": [
                    {"type": "reply", "text": "引用中的文字不能采集"},
                    {"type": "plain", "text": "[Image]"},
                    {"type": "plain", "text": "有效发言"},
                    {"type": "at", "user_id": "bot"},
                ],
            },
        ),
        saved_message(event, 3, content={"type": "bot", "message": [{"type": "plain", "text": "AI回复"}]}),
        saved_message(event, 4, content={"type": "user", "message": [{"type": "plain", "text": "[Record]"}]}),
        saved_message(event, 5, sender_id=None),
    ]
    plugin.context.message_history_manager = SavedHistory(records)
    reply = await imported(plugin, event)
    assert "AstrBot 本地群历史" in reply
    assert "读取 5 条，新增 2 条" in reply
    assert "无法拉取此前未接收、未保存或已清理" in reply
    assert plugin.store.samples(plugin._scope(event), "member-openid") == ["已保存的发言1", "有效发言"]
    assert plugin.service.ready_profile(plugin._scope(event), "member-openid")["generated_at_count"] == 2
    assert plugin.context.message_history_manager.calls == [
        {"platform_id": event.platform, "user_id": event.unified_msg_origin, "page_size": 1000}
    ]
    assert "新增 0 条" in await imported(plugin, event)


async def test_saved_history_stays_in_current_platform_and_group_and_supports_target_filter(plugin):
    event = official_event(plugin)
    plugin.context.message_history_manager = SavedHistory(
        [
            saved_message(event, 1, platform_id="other-platform"),
            saved_message(event, 2, user_id="qq_official_instance:GroupMessage:other-group"),
            saved_message(event, 3, user="another-member"),
            saved_message(event, 4),
        ]
    )
    reply = await imported(plugin, event, target="member-openid")
    assert "新增 1 条" in reply
    assert plugin.store.samples(plugin._scope(event), "member-openid") == ["已保存的发言4"]
    assert len(plugin.store.members(plugin._scope(event))) == 1


async def test_live_and_saved_history_share_dedup_ids_in_both_directions(plugin):
    event = official_event(plugin)
    live = Event(group=event.group, platform=event.platform, owner="member-openid", message_id="qq-native-id")
    live.message_obj.timestamp = 100
    live.set_extra("_current_platform_message_history_id", 1)
    await plugin.collect_messages(live)
    plugin.context.message_history_manager = SavedHistory([saved_message(event, 1), saved_message(event, 2)])
    assert "新增 1 条" in await imported(plugin, event)
    assert plugin.store.member(plugin._scope(event), "member-openid")["total"] == 2
    replay = Event(group=event.group, platform=event.platform, owner="member-openid", message_id="second-native-id")
    replay.set_extra("_current_platform_message_history_id", 2)
    await plugin.collect_messages(replay)
    assert plugin.store.member(plugin._scope(event), "member-openid")["total"] == 2
    # Learning a native ID for an imported row prevents a later replay without the local alias.
    replay.extra.clear()
    await plugin.collect_messages(replay)
    assert plugin.store.member(plugin._scope(event), "member-openid")["total"] == 2


async def test_empty_saved_history_explains_required_setting(plugin):
    event = official_event(plugin)
    plugin.context.message_history_manager = SavedHistory([])
    reply = await imported(plugin, event)
    assert "新增 0 条" in reply
    assert "group_message_history_enable" in reply
    assert not plugin.history_imports


async def test_saved_history_read_failure_is_reported_without_leaking_error(plugin):
    async def fail(**kwargs):
        raise RuntimeError("database-credential-not-for-users")

    event = official_event(plugin)
    plugin.context.message_history_manager = SimpleNamespace(get=fail)
    reply = await imported(plugin, event)
    assert "本地群历史读取失败（RuntimeError）" in reply
    assert "database-credential-not-for-users" not in reply
    assert not plugin.history_imports
