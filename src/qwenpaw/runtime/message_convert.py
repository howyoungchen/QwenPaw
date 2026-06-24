# -*- coding: utf-8 -*-
"""Message conversion between AgentRequest and agentscope Msg."""
from __future__ import annotations

import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config import load_config

logger = logging.getLogger(__name__)

_CURRENT_TIME_PREFIX = "Current time:"


def _media_type_to_block_type(media_type: str | None) -> str:
    """Map a MIME media_type to the 1.x block type the frontend expects.

    AS 2.0 uses ``"data"`` for all media; the frontend renderer still
    expects ``"image"``/``"video"``/``"audio"``.
    """
    if not media_type:
        return "data"
    major = media_type.split("/", 1)[0]
    if major in ("image", "video", "audio"):
        return major
    return "data"


def _get_last_user_text(msgs: List[Any]) -> str | None:
    """Extract the text of the last user message from a list of ``Msg``."""
    if not msgs:
        return None
    last = msgs[-1]
    if hasattr(last, "get_text_content"):
        return last.get_text_content()
    return None


def _ensure_url_scheme(url: str) -> str:
    """Prepend ``file://`` when *url* is an absolute local path.

    Always ``unquote()`` first so percent-encoded non-ASCII characters
    (e.g. ``%E6%B5%8B%E8%AF%95`` → ``测试``) resolve to the real
    filename on disk.  Then uses ``file://`` + raw path (not
    ``Path.as_uri()``) to avoid re-encoding.
    """
    if url.startswith(("/", "~")):
        resolved = str(Path(unquote(url)).expanduser().resolve())
        return "file://" + resolved
    return url


def _current_user_time_line() -> str:
    user_tz = load_config().user_timezone or "UTC"
    try:
        now = datetime.now(ZoneInfo(user_tz))
    except (ZoneInfoNotFoundError, KeyError):
        logger.warning("Invalid timezone %r, falling back to UTC", user_tz)
        now = datetime.now(timezone.utc)
        user_tz = "UTC"
    return (
        f"{_CURRENT_TIME_PREFIX} {now.strftime('%Y-%m-%d %H:%M:%S')} "
        f"{user_tz} ({now.strftime('%A')})"
    )


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        btype = block.get("type")
    else:
        btype = getattr(block, "type", None)
    if hasattr(btype, "value"):
        btype = btype.value
    return btype


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("text") or "")
    return str(getattr(block, "text", None) or "")


def _set_block_text(block: Any, text: str) -> None:
    if isinstance(block, dict):
        block["text"] = text
        return
    setattr(block, "text", text)


def _prepend_time_to_text(text: str, timestamp_line: str) -> str:
    if text.startswith(_CURRENT_TIME_PREFIX):
        return text
    if text.lstrip().startswith("/"):
        return text
    if not text:
        return timestamp_line
    return f"{timestamp_line}\n{text}"


def _prepend_current_time_to_user_messages(msgs: List[Any]) -> None:
    """Prepend fresh current time context to model-facing user messages."""
    timestamp_line: str | None = None

    for msg in msgs:
        role = getattr(msg, "role", None)
        if hasattr(role, "value"):
            role = role.value
        if role != "user":
            continue

        if timestamp_line is None:
            timestamp_line = _current_user_time_line()

        content = getattr(msg, "content", None)
        if isinstance(content, str):
            msg.content = _prepend_time_to_text(content, timestamp_line)
            continue

        try:
            from agentscope.message import TextBlock
        except Exception:
            logger.debug("Failed to import TextBlock for time prefix")
            continue

        if not isinstance(content, list):
            msg.content = [TextBlock(type="text", text=timestamp_line)]
            continue

        for block in content:
            if _block_type(block) != "text":
                continue
            text = _block_text(block)
            _set_block_text(block, _prepend_time_to_text(text, timestamp_line))
            break
        else:
            content.insert(0, TextBlock(type="text", text=timestamp_line))


# pylint: disable=too-many-branches
def _request_input_to_msgs(
    input_list: List[Any],
) -> List[Any]:
    """Convert ``AgentRequest.input`` (list of 1.x Message) to a list of
    agentscope 2.0 ``Msg`` objects.

    Handles text, image, audio, video, and file content blocks.
    """
    try:
        from agentscope.message import Msg, TextBlock, DataBlock
        from agentscope.message._block import URLSource
    except Exception:
        logger.error(
            "Failed to import agentscope.message; user input will be dropped",
            exc_info=True,
        )
        return []

    _MEDIA_TYPES = {
        "image": "image",
        "audio": "audio",
        "video": "video",
    }

    out: List[Any] = []
    for m in input_list:
        role = getattr(m, "role", None)
        if hasattr(role, "value"):
            role = role.value
        role = role or "user"
        if role == "tool":
            role = "assistant"

        blocks: list = []
        for c in getattr(m, "content", None) or []:
            ctype = getattr(c, "type", None)
            if hasattr(ctype, "value"):
                ctype = ctype.value

            if ctype == "text":
                text = getattr(c, "text", None) or ""
                if text:
                    blocks.append(TextBlock(type="text", text=text))

            elif ctype in _MEDIA_TYPES:
                url = (
                    getattr(c, "image_url", None)
                    or getattr(c, "audio_url", None)
                    or getattr(c, "video_url", None)
                    or getattr(c, "url", None)
                )
                if url:
                    url = _ensure_url_scheme(str(url))
                    url_path = urlparse(url).path
                    guessed, _ = mimetypes.guess_type(url_path)
                    if guessed and guessed.startswith(
                        f"{_MEDIA_TYPES[ctype]}/",
                    ):
                        media_type = guessed
                    else:
                        fallback_ext = "jpeg" if ctype == "image" else "mpeg"
                        media_type = f"{_MEDIA_TYPES[ctype]}/{fallback_ext}"
                    try:
                        blocks.append(
                            DataBlock(
                                source=URLSource(
                                    url=url,
                                    media_type=media_type,
                                ),
                            ),
                        )
                    except Exception:
                        logger.debug(
                            "Failed to create DataBlock for %s url=%s",
                            ctype,
                            url,
                        )

            elif ctype == "file":
                url = getattr(c, "file_url", None) or getattr(c, "url", None)
                if url:
                    url = _ensure_url_scheme(str(url))
                    try:
                        blocks.append(
                            DataBlock(
                                source=URLSource(
                                    url=url,
                                    media_type="application/octet-stream",
                                ),
                                name=getattr(c, "file_name", None),
                            ),
                        )
                    except Exception:
                        logger.debug(
                            "Failed to create DataBlock for file url=%s",
                            url,
                        )

        if not blocks:
            continue

        out.append(Msg(name=role, role=role, content=blocks))
    return out
