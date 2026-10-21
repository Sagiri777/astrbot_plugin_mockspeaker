import asyncio
import json

import pytest

from mockspeaker.config import Settings
from mockspeaker.service import MockSpeakerService
from mockspeaker.store import Store, scope_key


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "test.sqlite3")
    yield result
    result.close()


def collect(store, count, start=0, scope="group", user="bob", keep=10):
    for i in range(start, start + count):
        store.collect(scope, user, "群友", str(i), f"message {i}", keep)


def test_scopes_and_bounded_samples_do_not_reset_total(store):
    collect(store, 25)
    assert len(store.samples("group", "bob")) == 10
    assert store.member("group", "bob")["total"] == 25
    assert not store.collect("group", "bob", "群友", "24", "重复", 10)
    assert store.member("group", "bob")["total"] == 25
    collect(store, 1, scope="other")
    assert store.member("other", "bob")["total"] == 1
    assert scope_key("qq1", "100") != scope_key("qq2", "100")


async def test_initial_and_incremental_generation_watermarks(store):
    calls = []

    async def generate(*args):
        calls.append(args)
        return f"人格版本{len(calls)}"

    service = MockSpeakerService(store, Settings(min_messages=3, update_every=2), generate)
    collect(store, 2)
    assert not service.schedule_profile("group", "bob", "origin")
    collect(store, 1, start=2)
    assert service.schedule_profile("group", "bob", "origin")
    await asyncio.gather(*service.jobs.values())
    assert store.member("group", "bob")["generated_at_count"] == 3
    collect(store, 1, start=3)
    assert not service.schedule_profile("group", "bob", "origin")
    collect(store, 1, start=4)
    service.schedule_profile("group", "bob", "origin")
    await asyncio.gather(*service.jobs.values())
    assert len(calls) == 2
    assert store.member("group", "bob")["generated_at_count"] == 5


async def test_messages_during_generation_are_not_marked_as_analyzed(store):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def generate(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        return f"版本{calls}"

    service = MockSpeakerService(store, Settings(min_messages=2, update_every=2), generate)
    collect(store, 2)
    service.schedule_profile("group", "bob", "origin")
    await entered.wait()
    collect(store, 2, start=2)
    assert not service.schedule_profile("group", "bob", "origin")
    release.set()
    await asyncio.gather(*service.jobs.values())
    assert calls == 2
    assert store.member("group", "bob")["generated_at_count"] == 4


async def test_generation_failure_keeps_old_version_and_manual_retry_works(store):
    async def fail(*args):
        raise RuntimeError("failure")

    service = MockSpeakerService(store, Settings(min_messages=2, update_every=2), fail)
    collect(store, 2)
    store.save_profile("group", "bob", "old", 2)
    collect(store, 2, start=2)
    with pytest.raises(RuntimeError):
        await service.refresh_profile("group", "bob", "origin")
    member = store.member("group", "bob")
    assert member["prompt"] == "old" and member["generated_at_count"] == 2
    assert not service.schedule_profile("group", "bob", "origin")

    async def recover(*args):
        return "new"

    service.generate = recover
    assert service.schedule_profile("group", "bob", "origin", force=True)
    await asyncio.gather(*service.jobs.values())
    assert store.member("group", "bob")["prompt"] == "new"


async def test_failed_summary_survives_restart_and_never_becomes_raw_note(tmp_path):
    path = tmp_path / "durable.sqlite3"
    store = Store(path)

    async def fail(*args):
        raise RuntimeError("offline")

    settings = Settings(summary_every=1)
    service = MockSpeakerService(store, settings, fail)
    session = store.create_session("g", "alice", "bob", "origin", "cid")
    with pytest.raises(RuntimeError):
        await service.record(session["id"], "周五见？", "原始口头禅——好嘞老铁19点见！")
    assert not store.notes("origin", "cid")
    store.select("g", "alice", None)
    store.close()
    store = Store(path)

    async def recover(*args):
        return "双方决定周五19点见面。"

    service = MockSpeakerService(store, settings, recover)
    assert await service.flush_pending(origin="origin", cid="cid", closed_only=True)
    notes = store.notes("origin", "cid")
    assert len(notes) == 1 and "老铁" not in notes[0]["text"]
    await service.flush_pending(origin="origin", cid="cid")
    assert len(store.notes("origin", "cid")) == 1
    store.close()


async def test_summary_interval_compaction_and_fact_memory(store):
    async def generate(*args):
        return "此前已约定见面时间。"

    service = MockSpeakerService(store, Settings(summary_every=2, context_turns=2), generate)
    session = store.create_session("g", "a", "b", "o", "c")
    await service.record(session["id"], "one", "first")
    assert not store.notes("o", "c")
    await service.record(session["id"], "two", "second")
    assert len(store.notes("o", "c")) == 1
    await service.record(session["id"], "three", "third")
    await service.record(session["id"], "four", "fourth")
    assert len(store.turns(session["id"])) == 2
    context = service.context(store.session(session["id"]))
    assert "one" not in json.dumps(context)
    assert "约定" in json.dumps(context, ensure_ascii=False)


def test_selection_is_independent_and_persistent(tmp_path):
    path = tmp_path / "choices.sqlite3"
    store = Store(path)
    store.select("g", "a", "b")
    store.select("g", "c", "d")
    store.close()
    store = Store(path)
    assert store.selected("g", "a") == "b" and store.selected("g", "c") == "d"
    assert store.selected("another", "a") is None
    store.select("g", "a", None)
    assert store.selected("g", "c") == "d"
    store.close()


def test_config_rejects_bad_bounds_and_default_whitelist_is_empty():
    assert not Settings().allows("p", "100")
    assert Settings(allowed_groups=("p:100",)).allows("p", "100")
    assert not Settings(allowed_groups=("p:100",)).allows("q", "100")
    for config in ({"min_messages": 0}, {"update_every": -1}, {"summary_every": 6, "context_turns": 2}):
        with pytest.raises(ValueError):
            Settings.from_config(config)


def test_schema_defaults_match_runtime_settings():
    from pathlib import Path

    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    assert Settings.from_config({k: v["default"] for k, v in schema.items()}) == Settings()


async def test_profile_character_budget_prioritizes_latest_complete_samples(store):
    calls = []

    async def generate(origin, prompt, system):
        calls.append(json.loads(prompt))
        return "persona"

    for i in range(5):
        store.collect("g", "b", "name", str(i), str(i) * 1000, 10)
    service = MockSpeakerService(store, Settings(min_messages=1, analysis_chars=2000), generate)
    await service.refresh_profile("g", "b", "origin")
    assert calls[0]["messages"] == ["3" * 1000, "4" * 1000]
    assert store.member("g", "b")["generated_at_count"] == 5


async def test_profile_generation_concurrency_is_bounded(store):
    running, peak = 0, 0

    async def generate(*args):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0)
        running -= 1
        return "persona"

    service = MockSpeakerService(store, Settings(min_messages=1), generate)
    for i in range(5):
        collect(store, 1, start=i, user=str(i))
        service.schedule_profile("group", str(i), "origin")
    await asyncio.gather(*service.jobs.values())
    assert peak == 2
