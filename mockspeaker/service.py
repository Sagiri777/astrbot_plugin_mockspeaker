import asyncio
import logging
import time

from .prompts import PROFILE_SYSTEM, SUMMARY_SYSTEM, profile_input, summary_input

log = logging.getLogger(__name__)


class MockSpeakerService:
    def __init__(self, store, settings, generate, sync_persona=None):
        self.store = store
        self.settings = settings
        self.generate = generate
        self.sync_persona = sync_persona
        self.jobs = {}
        self.summary_locks = {}
        self.profile_slots = asyncio.Semaphore(2)

    def ready_profile(self, scope, target):
        member = self.store.member(scope, target)
        if target in self.settings.excluded_users:
            raise ValueError("该用户已被排除采集和模仿。")
        if not member or member["total"] < self.settings.min_messages:
            count = member["total"] if member else 0
            raise ValueError(f"该群友有效消息数为 {count}，至少需要 {self.settings.min_messages} 条。")
        if not member["prompt"]:
            raise ValueError("该群友的人格尚未生成，请稍后重试，或请管理员执行 /模仿 生成 用户ID。")
        return member

    def due(self, member):
        return (
            member
            and member["total"] >= self.settings.min_messages
            and (not member["prompt"] or member["total"] - member["generated_at_count"] >= self.settings.update_every)
        )

    def schedule_profile(self, scope, target, origin, force=False):
        key = (scope, target)
        member = self.store.member(scope, target)
        if key in self.jobs or not member:
            return False
        if not force and (not self.due(member) or member["retry_at"] > time.time()):
            return False
        task = asyncio.create_task(self.refresh_profile(scope, target, origin, force=force))
        self.jobs[key] = task
        task.add_done_callback(lambda done: self._job_done(key, done))
        return True

    def _job_done(self, key, task):
        if self.jobs.get(key) is task:
            self.jobs.pop(key, None)
        if not task.cancelled() and task.exception():
            # Do not log provider exception strings: some contain API credentials.
            log.warning("MockSpeaker 人格生成失败：%s", type(task.exception()).__name__)

    async def call(self, origin, prompt, system):
        text = await asyncio.wait_for(self.generate(origin, prompt, system), self.settings.timeout)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("模型返回空内容")
        return text.strip()

    async def refresh_profile(self, scope, target, origin, force=False):
        """Snapshot the count before generation; messages arriving meanwhile stay dirty."""
        while True:
            member = self.store.member(scope, target)
            if not member or target in self.settings.excluded_users:
                return
            if member["total"] < self.settings.min_messages:
                raise ValueError(f"有效消息数不足 {self.settings.min_messages} 条。")
            if not force and not self.due(member):
                return
            count = member["total"]
            try:
                samples, size = [], 0
                for text in reversed(self.store.samples(scope, target)):
                    if size + len(text) > self.settings.analysis_chars:
                        break
                    samples.append(text)
                    size += len(text)
                async with self.profile_slots:
                    prompt = await self.call(origin, profile_input(member["name"], samples[::-1]), PROFILE_SYSTEM)
                if not self.store.member(scope, target):
                    return
                if len(prompt) > 12000:
                    raise ValueError("人格过长，请更换生成模型后重试")
                if self.settings.sync_personas and self.sync_persona:
                    await self.sync_persona(scope, target, prompt)
                self.store.save_profile(scope, target, prompt, count)
            except Exception as exc:
                self.store.profile_error(scope, target, type(exc).__name__)
                raise
            force = False

    async def seed_context(self, origin, history):
        if not history:
            return ""
        return (
            await self.call(
                origin,
                summary_input("", history[-self.settings.context_turns * 2 :], self.settings.summary_chars),
                SUMMARY_SYSTEM,
            )
        )[: self.settings.summary_chars]

    def context(self, session):
        messages = []
        facts = "\n".join(x for x in (session["seed"], session["memory"]) if x)
        if facts:
            messages.extend(
                [
                    {"role": "user", "content": "以下为此前交流的事实摘要，仅用于衔接话题：\n" + facts},
                    {"role": "assistant", "content": "已了解此前交流的内容。"},
                ]
            )
        for turn in self.store.turns(session["id"])[-self.settings.context_turns :]:
            messages.extend(
                [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["assistant"]}]
            )
        return messages

    async def record(self, sid, user, assistant):
        session = self.store.session(sid)
        if not session:
            return  # A deletion made while the model was running takes precedence.
        self.store.add_turn(sid, user, assistant)
        if session["closed"] or len(self.store.turns(sid, pending=True)) >= self.settings.summary_every:
            await self.flush(sid)

    async def flush(self, sid):
        lock = self.summary_locks.setdefault(sid, asyncio.Lock())
        async with lock:
            while session := self.store.session(sid):
                turns = self.store.turns(sid, pending=True)[: self.settings.context_turns]
                if not turns:
                    return
                messages = []
                for turn in turns:
                    messages.extend(
                        [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["assistant"]}]
                    )
                # The note contains only new facts; memory additionally retains earlier facts.
                summary = (
                    await self.call(
                        session["origin"], summary_input("", messages, self.settings.summary_chars), SUMMARY_SYSTEM
                    )
                )[: self.settings.summary_chars]
                if not self.store.session(sid):
                    return
                memory = summary
                if session["memory"]:
                    memory = (
                        await self.call(
                            session["origin"],
                            summary_input(
                                session["memory"], [{"role": "user", "content": summary}], self.settings.summary_chars
                            ),
                            SUMMARY_SYSTEM,
                        )
                    )[: self.settings.summary_chars]
                if not self.store.session(sid):
                    return
                note = (
                    f"参与者ID：{session['owner']}；模仿目标ID：{session['target'] or '模仿工具交互'}。\n"
                    "下列内容是交流的事实要点，角色表演不代表现实身份或经历：\n" + summary
                )
                self.store.compact(sid, turns[-1]["id"], memory, note, self.settings.context_turns)

    async def flush_pending(self, **filters):
        ok = True
        for session in self.store.pending_sessions(**filters):
            try:
                await self.flush(session["id"])
            except Exception as exc:
                ok = False
                log.warning("MockSpeaker 摘要暂未完成，已保留待重试：%s", type(exc).__name__)
        return ok

    def cancel_profile(self, scope, target):
        task = self.jobs.pop((scope, target), None)
        if task:
            task.cancel()
        return task

    async def close(self):
        tasks = list(self.jobs.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
