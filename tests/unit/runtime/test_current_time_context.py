# -*- coding: utf-8 -*-
# pylint: disable=protected-access
from __future__ import annotations

from datetime import datetime as real_datetime
from types import SimpleNamespace

import pytest
from agentscope.message import DataBlock, Msg, TextBlock, URLSource

from qwenpaw.app.chats import utils as chat_utils
from qwenpaw.runtime import message_convert
from qwenpaw.runtime import runtime as runtime_mod
from qwenpaw.runtime.hooks import HookResult
from qwenpaw.runtime.phases import Phase
from qwenpaw.runtime.runtime import Runtime
from qwenpaw.schemas import (
    AgentRequest,
    ContentType,
    Message,
    MessageType,
    Role,
    TextContent,
)


class FixedDatetime:
    @classmethod
    def now(cls, tz=None):
        return real_datetime(2026, 6, 24, 8, 9, 10, tzinfo=tz)


def _patch_time(monkeypatch):
    monkeypatch.setattr(
        message_convert,
        "datetime",
        FixedDatetime,
        raising=False,
    )
    monkeypatch.setattr(
        message_convert,
        "load_config",
        lambda: SimpleNamespace(user_timezone="Asia/Shanghai"),
        raising=False,
    )


def _user_message(text: str) -> Message:
    return Message(
        type=MessageType.MESSAGE,
        role=Role.USER,
        content=[TextContent(type=ContentType.TEXT, text=text)],
    )


def test_env_context_does_not_include_dynamic_time(monkeypatch):
    monkeypatch.setattr(
        chat_utils,
        "load_config",
        lambda: SimpleNamespace(user_timezone="Asia/Shanghai"),
    )

    env_context = chat_utils.build_env_context(
        session_id="qq:session",
        user_id="qq-user",
        channel="qq",
        working_dir="/workspace/agent",
        add_hint=False,
    )

    assert "Current date:" not in env_context
    assert "Current time:" not in env_context


def test_user_text_gets_current_time_prefix(monkeypatch):
    _patch_time(monkeypatch)
    msg = Msg(name="user", role="user", content=[TextBlock(text="hello")])

    message_convert._prepend_current_time_to_user_messages([msg])

    assert msg.get_text_content() == (
        "Current time: 2026-06-24 08:09:10 Asia/Shanghai (Wednesday)\nhello"
    )


def test_media_only_user_message_gets_current_time_text_block(monkeypatch):
    _patch_time(monkeypatch)
    msg = Msg(
        name="user",
        role="user",
        content=[
            DataBlock(
                source=URLSource(
                    url="file:///tmp/image.png",
                    media_type="image/png",
                ),
            ),
        ],
    )

    message_convert._prepend_current_time_to_user_messages([msg])

    assert msg.content[0].type == "text"
    assert msg.content[0].text == (
        "Current time: 2026-06-24 08:09:10 Asia/Shanghai (Wednesday)"
    )


def test_current_time_prefix_skips_non_user_and_existing_prefix(monkeypatch):
    _patch_time(monkeypatch)
    assistant = Msg(
        name="assistant",
        role="assistant",
        content=[TextBlock(text="hello")],
    )
    user = Msg(
        name="user",
        role="user",
        content=[TextBlock(text="Current time: already present\nhello")],
    )

    message_convert._prepend_current_time_to_user_messages([assistant, user])

    assert assistant.get_text_content() == "hello"
    assert user.get_text_content() == "Current time: already present\nhello"


def test_current_time_prefix_skips_raw_slash_magic_command(monkeypatch):
    _patch_time(monkeypatch)
    user = Msg(
        name="user",
        role="user",
        content=[TextBlock(text="/compose write release notes")],
    )

    message_convert._prepend_current_time_to_user_messages([user])

    assert user.get_text_content() == "/compose write release notes"


@pytest.mark.asyncio
async def test_runtime_dispatches_slash_command_without_time_prefix(
    monkeypatch,
    tmp_path,
):
    _patch_time(monkeypatch)
    captured: list[str] = []

    class HookRegistry:
        async def run(self, _phase: Phase, _ctx):
            return HookResult()

    class SlashRegistry:
        async def dispatch(self, raw_text: str, _ctx):
            captured.append(raw_text)
            return Msg(
                name="assistant",
                role="assistant",
                content=[TextBlock(text="ok")],
            )

    workspace = SimpleNamespace(
        workspace_dir=tmp_path,
        plugins=SimpleNamespace(
            hook_registry=HookRegistry(),
            slash_command_registry=SlashRegistry(),
        ),
    )

    runtime = Runtime(workspace=workspace, app_services=None)
    request = AgentRequest(
        session_id="s",
        user_id="u",
        input=[_user_message("/compact")],
    )

    _ = [event async for event in runtime.run(request)]

    assert captured == ["/compact"]


@pytest.mark.asyncio
async def test_runtime_sends_time_prefixed_text_to_agent(
    monkeypatch,
    tmp_path,
):
    _patch_time(monkeypatch)
    captured_inputs: list[list] = []

    class HookRegistry:
        async def run(self, _phase: Phase, _ctx):
            return HookResult()

    class SlashRegistry:
        async def dispatch(self, _raw_text: str, _ctx):
            return None

    class FakeAgent:
        def reply_stream(self, inputs):
            captured_inputs.append(inputs)

            async def _empty_stream():
                for chunk in ():
                    yield chunk

            return _empty_stream()

        async def close(self):
            return None

    class FakeAgentBuilder:
        def __init__(self, *, app_services):
            self.app_services = app_services

        async def build(self, _ctx):
            return FakeAgent()

    monkeypatch.setattr(runtime_mod, "AgentBuilder", FakeAgentBuilder)

    workspace = SimpleNamespace(
        workspace_dir=tmp_path,
        plugins=SimpleNamespace(
            hook_registry=HookRegistry(),
            slash_command_registry=SlashRegistry(),
        ),
    )
    runtime = Runtime(workspace=workspace, app_services=None)
    request = AgentRequest(
        session_id="s",
        user_id="u",
        input=[_user_message("hello")],
    )

    _ = [event async for event in runtime.run(request)]

    assert captured_inputs
    assert captured_inputs[0][0].get_text_content() == (
        "Current time: 2026-06-24 08:09:10 Asia/Shanghai (Wednesday)\nhello"
    )
