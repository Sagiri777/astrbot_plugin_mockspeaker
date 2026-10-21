import asyncio
import hashlib
import json

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools

from .mockspeaker.config import Settings
from .mockspeaker.prompts import mimic_system
from .mockspeaker.service import MockSpeakerService
from .mockspeaker.store import Store, scope_key

HELP = """群友发言模仿（仅限配置中启用的群）
/模仿 列表 [页码]：查看采集数和可用人格
/模仿 开始 用户ID（或 @群友）：为你选择模仿对象
/模仿 暂停：恢复原人格
/模仿 状态 [用户ID]：查看自己选择或某群友的采集进度
/模仿 提示词 用户ID：查看生成的人格
/模仿 生成 用户ID：管理员手动生成/更新
/模仿 删除 [用户ID]：删除本人数据；管理员可删除他人数据
英文：/mockspeaker list|start|stop|status|prompt|generate|forget
每人的选择独立。只在机器人本来会回复你的消息时模仿。"""


class MockSpeakerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.settings = Settings.from_config(config)
        self.store = Store(StarTools.get_data_dir("astrbot_plugin_mockspeaker") / "mockspeaker.sqlite3")
        self.service = MockSpeakerService(self.store, self.settings, self._generate, self._sync_persona)

    def _scope(self, event):
        return scope_key(str(event.get_platform_id()), str(event.get_group_id()))

    def _enabled(self, event):
        return self.settings.allows(str(event.get_platform_id()), str(event.get_group_id()))

    def _require_group(self, event):
        if not self._enabled(event):
            raise ValueError("当前群未启用 MockSpeaker；请在插件配置的 allowed_groups 中添加群号。")

    def _target(self, event, target):
        for component in event.get_messages():
            if isinstance(component, At) and str(component.qq) not in (str(event.get_self_id()), "all", "0"):
                return str(component.qq)
        target = str(target).strip()
        if not target:
            raise ValueError("请提供群友用户ID，或 @目标群友。")
        return target

    async def _generate(self, origin, prompt, system):
        provider_id = self.settings.provider_id or await self.context.get_current_chat_provider_id(origin)
        if not provider_id:
            raise ValueError("请先配置可用的聊天模型提供商。")
        response = await self.context.llm_generate(
            chat_provider_id=provider_id, prompt=prompt, system_prompt=system, contexts=[]
        )
        if response.role != "assistant":
            raise ValueError("模型未返回有效回复")
        return response.completion_text

    async def _sync_persona(self, scope, target, prompt):
        persona_id = "mockspeaker_" + hashlib.sha256(f"{scope}:{target}".encode()).hexdigest()[:24]
        manager = self.context.persona_manager
        try:
            await manager.get_persona(persona_id)
        except ValueError:
            await manager.create_persona(persona_id=persona_id, system_prompt=mimic_system(prompt), begin_dialogs=[])
        else:
            await manager.update_persona(persona_id=persona_id, system_prompt=mimic_system(prompt), begin_dialogs=[])

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=100)
    async def collect_messages(self, event: AstrMessageEvent):
        if not self._enabled(event):
            return
        user = str(event.get_sender_id())
        if user == str(event.get_self_id()) or user in self.settings.excluded_users:
            return
        text = "".join(part.text for part in event.get_messages() if isinstance(part, Plain)).strip()
        if len(text) < self.settings.min_message_chars or any(
            text.startswith(p) for p in self.settings.command_prefixes
        ):
            return
        # Never learn the plugin's own command arguments when wake prefixes omit '/'.
        if text.lstrip("/").startswith(("模仿", "mockspeaker")):
            return
        scope = self._scope(event)
        self.store.collect(
            scope,
            user,
            str(event.get_sender_name()),
            str(event.message_obj.message_id or ""),
            text[: self.settings.max_message_chars],
            self.settings.max_samples,
        )
        self.service.schedule_profile(scope, user, event.unified_msg_origin)

    @filter.command_group("模仿", alias={"mockspeaker"})
    async def mimic(self, event: AstrMessageEvent):
        """群友发言风格模仿。"""
        yield event.plain_result(HELP)

    @mimic.command("帮助", alias={"help"})
    async def help_command(self, event: AstrMessageEvent):
        yield event.plain_result(HELP)

    @mimic.command("列表", alias={"list"})
    async def list_command(self, event: AstrMessageEvent, page: int = 1):
        try:
            self._require_group(event)
            if page < 1:
                raise ValueError("页码必须大于零。")
            members = [
                m
                for m in self.service.profile_members(self._scope(event))
                if m["user_id"] not in self.settings.excluded_users
            ]
            mode = "跨群合并" if self.settings.cross_group_profiles else "本群"
            lines = [f"群友人格（第 {page} 页；每页20人；{mode}门槛 {self.settings.min_messages} 条）"]
            for member in members[(page - 1) * 20 : page * 20]:
                state = (
                    "可模仿" if member["prompt"] and member["total"] >= self.settings.min_messages else "积累中/生成中"
                )
                lines.append(f"{member['name']} ({member['user_id']})：{member['total']} 条，{state}")
            yield event.plain_result("\n".join(lines) if len(lines) > 1 else "本页暂无采集记录。")
        except ValueError as exc:
            yield event.plain_result(str(exc))

    async def _select(self, event, target):
        self._require_group(event)
        scope, owner = self._scope(event), str(event.get_sender_id())
        member = self.service.ready_profile(scope, target)
        if self.store.selected(scope, owner) == target:
            return f"你已在模仿 {member['name']} ({target})。"
        self.store.select(scope, owner, target)
        complete = await self.service.flush_pending(scope=scope, owner=owner)
        return f"已为你选择 {member['name']} ({target})，从下一条触发 AI 的消息起生效。" + (
            " 旧会话摘要暂未完成，已保留待重试。" if not complete else ""
        )

    @mimic.command("开始", alias={"start"})
    async def start_command(self, event: AstrMessageEvent, target: str = ""):
        try:
            yield event.plain_result(await self._select(event, self._target(event, target)))
        except ValueError as exc:
            yield event.plain_result(str(exc))

    async def _stop(self, event):
        if not event.get_group_id():
            raise ValueError("请在原群聊中暂停模仿。")
        scope, owner = self._scope(event), str(event.get_sender_id())
        self.store.select(scope, owner, None)
        complete = await self.service.flush_pending(scope=scope, owner=owner)
        return "已暂停模仿，下一次回复恢复原人格；事实摘要将在原会话下次请求时回填。" + (
            " 摘要模型暂不可用，原文已隔离保存，将重试提取要点。" if not complete else ""
        )

    @mimic.command("暂停", alias={"stop", "停止"})
    async def stop_command(self, event: AstrMessageEvent):
        try:
            yield event.plain_result(await self._stop(event))
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @mimic.command("状态", alias={"status"})
    async def status_command(self, event: AstrMessageEvent, target: str = ""):
        try:
            self._require_group(event)
            scope = self._scope(event)
            if not target:
                selected = self.store.selected(scope, str(event.get_sender_id()))
                yield event.plain_result(f"你当前的模仿对象：{selected or '未开启，使用原人格'}。")
                return
            member = self.service.profile_member(scope, target)
            if not member or target in self.settings.excluded_users:
                raise ValueError("该群友暂无可用采集记录。")
            remaining = max(
                0,
                (
                    member["generated_at_count"] + self.settings.update_every
                    if member["prompt"]
                    else self.settings.min_messages
                )
                - member["total"],
            )
            yield event.plain_result(
                f"{member['name']} ({target})：有效消息 {member['total']} 条；"
                f"人格已覆盖 {member['generated_at_count']} 条；距更新 {remaining} 条。"
                + (" 计数范围：同一平台实例的已启用群合计。" if self.settings.cross_group_profiles else "")
                + (f" 上次生成失败：{member['error']}；五分钟后收到新消息重试。" if member["error"] else "")
            )
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @mimic.command("提示词", alias={"prompt"})
    async def prompt_command(self, event: AstrMessageEvent, target: str = ""):
        try:
            self._require_group(event)
            member = self.service.ready_profile(self._scope(event), self._target(event, target))
            yield event.plain_result(member["prompt"])
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @mimic.command("生成", alias={"generate"})
    async def generate_command(self, event: AstrMessageEvent, target: str = ""):
        try:
            self._require_group(event)
            if not event.is_admin():
                raise ValueError("此操作需要 AstrBot 管理员权限。")
            target = self._target(event, target)
            scope = self._scope(event)
            if target in self.settings.excluded_users:
                raise ValueError("该用户已被排除采集。")
            if self.service.profile_job_key(scope, target) in self.service.jobs:
                raise ValueError("该人格正在生成，请稍后查看状态。")
            member = self.service.profile_member(scope, target)
            if not member or member["total"] < self.settings.min_messages:
                raise ValueError(f"有效消息数不足 {self.settings.min_messages} 条。")
            self.service.schedule_profile(scope, target, event.unified_msg_origin, force=True)
            yield event.plain_result("已提交人格生成任务，可用 /模仿 状态 用户ID 查看结果。")
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @mimic.command("删除", alias={"forget"})
    async def forget_command(self, event: AstrMessageEvent, target: str = ""):
        if not event.get_group_id():
            yield event.plain_result("请在需要删除记录的群聊中操作。")
            return
        target = str(target or event.get_sender_id()).strip()
        if target != str(event.get_sender_id()) and not event.is_admin():
            yield event.plain_result("只能删除自己的数据；删除其他群友数据需要 AstrBot 管理员权限。")
            return
        scope = self._scope(event)
        task = self.service.cancel_profile(scope, target)
        if task:
            await asyncio.gather(task, return_exceptions=True)
        shared_keys = self.store.forget(scope, target)
        if self.settings.sync_personas:
            for profile_scope in [scope, *shared_keys]:
                pid = "mockspeaker_" + hashlib.sha256(f"{profile_scope}:{target}".encode()).hexdigest()[:24]
                try:
                    await self.context.persona_manager.get_persona(pid)
                except ValueError:
                    pass
                else:
                    await self.context.persona_manager.delete_persona(pid)
        yield event.plain_result(
            "已删除该用户在本群的采集、人格和相关隔离记录，并取消本群相关选择；使用这些语料的合并人格已失效。已有事实摘要不回溯删除。后续消息会重新采集；要停采请配置 excluded_users。"
        )

    async def _deliver_notes(self, event, req):
        """Runs inside AstrBot's session lock. Ack only after the DB write succeeds."""
        if not req.conversation:
            return
        cid, origin = req.conversation.cid, event.unified_msg_origin
        notes = self.store.notes(origin, cid)
        if not notes:
            return
        conv = await self.context.conversation_manager.get_conversation(origin, cid)
        if not conv:
            return
        history = json.loads(conv.history or "[]")
        serialized = json.dumps(history, ensure_ascii=False)
        additions = []
        for note in notes:
            marker = f"[MockSpeaker 事实摘要 {note['id']}]"
            if marker not in serialized:
                additions.extend(
                    [
                        {"role": "user", "content": marker + "\n" + note["text"]},
                        {"role": "assistant", "content": "已记录这些交流要点。"},
                    ]
                )
        if additions:
            history.extend(additions)
            await self.context.conversation_manager.update_conversation(origin, cid, history=history)
            req.contexts = list(req.contexts or []) + additions
            req.conversation.history = json.dumps(history, ensure_ascii=False)
        self.store.acknowledge([note["id"] for note in notes])

    @filter.on_llm_request(priority=-10000)
    async def prepare_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not event.get_group_id():
            return
        event.set_extra("mockspeaker_request", req)
        event.set_extra("mockspeaker_original_conversation", req.conversation)
        scope, owner = self._scope(event), str(event.get_sender_id())
        target = self.store.selected(scope, owner)
        if target and (not self._enabled(event) or target in self.settings.excluded_users):
            self.store.select(scope, owner, None)
            target = None
        try:
            if req.conversation:
                await self.service.flush_pending(
                    origin=event.unified_msg_origin, cid=req.conversation.cid, closed_only=True
                )
                await self._deliver_notes(event, req)
            if not target:
                return
            member = self.service.ready_profile(scope, target)
            original = req.conversation
            # Detach BEFORE any model call. A failed preparation cannot save raw imitation.
            req.conversation = None
            req.system_prompt = mimic_system(member["prompt"])
            req.contexts = []
            req.tool_calls_result = None
            if not original:
                raise ValueError("当前请求没有可绑定的 AstrBot 会话，请重新发起聊天。")
            session = self.store.active_session(scope, owner, target, event.unified_msg_origin, original.cid)
            if not session:
                seed = await self.service.seed_context(event.unified_msg_origin, json.loads(original.history or "[]"))
                # Commands may change the choice while a seed is being generated.
                if self.store.selected(scope, owner) != target:
                    raise ValueError("模仿选择已改变，请重新发送消息。")
                session = self.store.create_session(scope, owner, target, event.unified_msg_origin, original.cid, seed)
            req.contexts = self.service.context(session)
            req.session_id = "mockspeaker:" + session["id"]
            event.set_extra("mockspeaker_turn", (session["id"], req.prompt or event.message_str))
        except Exception as exc:
            logger.warning("MockSpeaker 请求准备失败：%s", type(exc).__name__)
            if target:
                req.conversation = None
                event.stop_event()
                await event.send(
                    event.plain_result("模仿请求准备失败，已阻止本次回复。可重试，或 /模仿 暂停 恢复原人格。")
                )

    @filter.on_llm_response(priority=-10000)
    async def capture_response(self, event: AstrMessageEvent, response: LLMResponse):
        turn = event.get_extra("mockspeaker_turn")
        if not turn or event.get_extra("mockspeaker_captured") or not response or response.role != "assistant":
            return
        if not response.completion_text:
            return
        event.set_extra("mockspeaker_captured", True)
        try:
            await self.service.record(turn[0], turn[1], response.completion_text)
        except Exception as exc:
            logger.warning("MockSpeaker 摘要暂未完成，隔离记录已保留：%s", type(exc).__name__)

    def _isolate_tool_turn(self, event):
        """Neither persona rules nor persistent imitation commands belong in original history."""
        if event.get_extra("mockspeaker_turn"):
            return
        req = event.get_extra("mockspeaker_request")
        original = event.get_extra("mockspeaker_original_conversation")
        if req is None or original is None:
            raise ValueError("此工具需要 AstrBot 内置聊天流水线。")
        req.conversation = None
        session = self.store.create_session(
            self._scope(event), str(event.get_sender_id()), "", event.unified_msg_origin, original.cid, closed=True
        )
        req.session_id = "mockspeaker:" + session["id"]
        event.set_extra("mockspeaker_turn", (session["id"], req.prompt or event.message_str))

    @filter.llm_tool(name="mockspeaker_list")
    async def tool_list(self, event: AstrMessageEvent) -> str:
        """列出本群已采集的可模仿群友及其用户ID；人格可按配置合并其他已启用群的语料。"""
        if not self._enabled(event):
            return "当前群未启用 MockSpeaker。"
        return json.dumps(
            [
                {"user_id": m["user_id"], "name": m["name"], "messages": m["total"]}
                for m in self.service.profile_members(self._scope(event))
                if m["prompt"]
                and m["total"] >= self.settings.min_messages
                and m["user_id"] not in self.settings.excluded_users
            ],
            ensure_ascii=False,
        )

    @filter.llm_tool(name="mockspeaker_get_prompt")
    async def tool_prompt(self, event: AstrMessageEvent, target_id: str) -> str:
        """读取本群指定群友的模仿人格提示词。只读取，不修改选择。需要持续模仿时调用 mockspeaker_select。

        Args:
            target_id(string): 本群目标群友的用户ID。
        """
        try:
            self._require_group(event)
            member = self.service.ready_profile(self._scope(event), target_id)
            self._isolate_tool_turn(event)
            return member["prompt"]
        except ValueError as exc:
            return str(exc)

    @filter.llm_tool(name="mockspeaker_select")
    async def tool_select(self, event: AstrMessageEvent, target_id: str) -> str:
        """用户明确要求持续模仿某群友时，为当前发言者选择对象；从下一条消息生效，当前回复只确认操作。

        Args:
            target_id(string): 用户明确选择的本群群友ID，不得替其他用户更改。
        """
        try:
            self._require_group(event)
            self.service.ready_profile(self._scope(event), target_id)
            self._isolate_tool_turn(event)
            return await self._select(event, target_id)
        except ValueError as exc:
            return str(exc)

    @filter.llm_tool(name="mockspeaker_pause")
    async def tool_pause(self, event: AstrMessageEvent) -> str:
        """用户明确要求停止或暂停模仿时，恢复当前发言者的原人格；下一条消息生效，当前回复只确认操作。"""
        try:
            if not event.get_group_id():
                raise ValueError("请在原群聊中暂停模仿。")
            self._isolate_tool_turn(event)
            return await self._stop(event)
        except ValueError as exc:
            return str(exc)

    async def terminate(self):
        await self.service.close()
        self.store.close()
