"""Read group history from OneBot or AstrBot's persisted platform messages."""

import asyncio
import math
import re
from datetime import datetime, timezone

STORED_MARKERS = {
    f"[{name}]"
    for name in "Image Record Video File Face Node Nodes Poke Forward RPS Dice Shake Share Contact Location Music Json Unknown".split()
}


def message_time(message):
    try:
        value = float(message.get("time", 0))
    except (TypeError, ValueError):
        return 0
    return value if math.isfinite(value) and value >= 0 else 0


def plain_text(message):
    if isinstance(message, str):
        # Strip CQ components before unescaping, preserving literal escaped CQ text.
        text = re.sub(r"\[CQ:[^\]]*\]", "", message)
        for escaped, literal in (("&#91;", "["), ("&#93;", "]"), ("&#44;", ","), ("&amp;", "&")):
            text = text.replace(escaped, literal)
        return text
    if not isinstance(message, list):
        return ""
    return "".join(
        part["data"]["text"]
        for part in message
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("data"), dict)
        and isinstance(part["data"].get("text"), str)
    )


def history_action(event):
    bot = getattr(event, "bot", None)
    action = getattr(getattr(bot, "api", None), "call_action", None)
    if not callable(action):
        action = getattr(bot, "call_action", None)
    if not callable(action):
        raise ValueError("当前平台不支持群历史导入；需要提供 get_group_msg_history 的 OneBot 适配器（如 NapCat）。")
    return action


def history_source(event, context):
    try:
        history_action(event)
    except ValueError:
        if callable(getattr(getattr(context, "message_history_manager", None), "get", None)):
            return "saved"
        raise ValueError(
            "当前平台不支持群历史导入接口，且 AstrBot 未提供本地群历史读取能力。"
            "qq_official 只能使用已保存的群消息；请使用支持群历史保存的 AstrBot 版本并开启 "
            "provider_ltm_settings.group_message_history_enable。"
        ) from None
    return "onebot"


async def fetch_saved_history(context, event, limit, timeout):
    try:
        records = await asyncio.wait_for(
            context.message_history_manager.get(
                platform_id=str(event.get_platform_id()), user_id=event.unified_msg_origin, page_size=limit
            ),
            timeout,
        )
    except Exception as exc:
        raise ValueError(f"AstrBot 本地群历史读取失败（{type(exc).__name__}），请稍后重试。") from None
    messages = []
    for record in records:
        if (
            record.platform_id != str(event.get_platform_id())
            or record.user_id != event.unified_msg_origin
            or record.id is None
            or not isinstance(record.content, dict)
        ):
            continue
        parts = record.content.get("message") if record.content.get("type") == "user" else []
        if not isinstance(parts, list):
            continue
        text = "".join(
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and part.get("type") == "plain"
            and isinstance(part.get("text"), str)
            and part["text"] not in STORED_MARKERS
        )
        timestamp = record.created_at
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            timestamp = timestamp.timestamp()
        messages.append(
            {
                "message_id": f"astrbot_history:{record.id}",
                "time": timestamp,
                "group_id": str(event.get_group_id()),
                "sender": {"user_id": record.sender_id, "nickname": record.sender_name},
                "message": [{"type": "text", "data": {"text": text}}],
            }
        )
    notice = "本次仅使用 AstrBot 已保存的群历史；qq_official 无法拉取此前未接收、未保存或已清理的群聊记录。"
    if not records:
        notice += " 当前没有本地记录，请确认此前已开启 provider_ltm_settings.group_message_history_enable。"
    return sorted(messages, key=message_time), notice


async def fetch_history(event, limit, timeout):
    """Return unique records, plus a notice when the adapter stops or fails partway."""
    action = history_action(event)
    messages, seen, cursor = [], set(), None
    while len(messages) < limit:
        params = {
            "group_id": int(event.get_group_id()),
            "self_id": str(event.get_self_id()),
            "count": min(100, limit - len(messages) + (cursor is not None)),
        }
        if cursor is not None:
            params["message_seq"] = cursor
        try:
            response = await asyncio.wait_for(action("get_group_msg_history", **params), timeout)
            if not isinstance(response, dict) or response.get("status") == "failed" or response.get("retcode", 0):
                raise ValueError("history action failed")
            data = response.get("data", response)
            if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
                raise ValueError("invalid history response")
            batch = [m for m in data["messages"] if isinstance(m, dict) and m.get("message_id") is not None]
            if not data["messages"]:
                break
            if not batch:
                raise ValueError("history messages have no IDs")
        except Exception as exc:
            # Do not expose adapter errors: they may contain connection credentials.
            notice = f"历史接口读取失败（{type(exc).__name__}）；请确认适配器支持 get_group_msg_history 且机器人有可读取的群历史。"
            if not messages:
                raise ValueError(notice) from None
            return sorted(reversed(messages), key=message_time), notice
        # OneBot returns chronological records; the first minimum also handles
        # several messages sent within the same second without skipping a page.
        oldest = min(batch, key=message_time)
        batch.reverse()
        batch.sort(key=message_time, reverse=True)
        before = len(messages)
        for message in batch:
            message_id = str(message["message_id"])
            if message_id and message_id not in seen and len(messages) < limit:
                messages.append(message)
                seen.add(message_id)
        if len(messages) >= limit:
            break
        next_cursor = oldest.get("message_seq") or oldest["message_id"]
        if len(messages) == before or str(next_cursor) == str(cursor):
            return sorted(reversed(messages), key=message_time), "历史接口未返回更早的消息，已停止翻页。"
        cursor = next_cursor
    return sorted(reversed(messages), key=message_time), ""
