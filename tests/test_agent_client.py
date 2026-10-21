"""Tests for agent client implementations."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from midojo.agent_client import AgentClient, OGXResponsesClient, OpenAIResponsesAgentClient, OpenShellAgentClient


@pytest.mark.asyncio
async def test_openshell_agent_failure_is_not_graded_as_empty_output():
    runtime = MagicMock()
    runtime.exec_agent.return_value = MagicMock(exit_code=1, stdout="", stderr="Inference connection failed")
    client = OpenShellAgentClient(runtime)
    with pytest.raises(RuntimeError, match="Inference connection failed"):
        await client.send_task("task", session_token="session-a")


class TestOGXResponsesClient:
    """Tests for the OGX protocol client fixes (PR #93)."""

    def test_mcp_server_label_stored(self):
        """mcp_server_label is stored and not hardcoded to 'weather'."""
        client = OGXResponsesClient(
            ogx_url="http://localhost:8321",
            model="llama-scout-17b",
            mcp_server_url="http://localhost:8082/mcp",
            mcp_server_label="minibank",
        )
        assert client.mcp_server_label == "minibank"

    def test_mcp_server_label_default(self):
        """mcp_server_label defaults to 'midojo' when not given."""
        client = OGXResponsesClient(
            ogx_url="http://localhost:8321",
            model="llama-scout-17b",
            mcp_server_url="http://localhost:8082/mcp",
        )
        assert client.mcp_server_label == "midojo"

    def test_mcp_server_label_override(self):
        """mcp_server_label can be overridden from suite_name default."""
        client = OGXResponsesClient(
            ogx_url="http://localhost:8321",
            model="llama-scout-17b",
            mcp_server_url="http://localhost:8082/mcp",
            mcp_server_label="my-custom-label",
        )
        assert client.mcp_server_label == "my-custom-label"

    def test_suite_module_import_missing(self):
        """Protocol dispatch: missing suite module falls back to empty SYSTEM_MESSAGE."""
        try:
            _suite_mod = importlib.import_module("suites.nonexistent_suite_xyz")
        except ImportError:
            _suite_mod = None
        system_message = getattr(_suite_mod, "SYSTEM_MESSAGE", "")
        assert system_message == ""

    def test_suite_module_import_no_system_message(self):
        """Protocol dispatch: suite module without SYSTEM_MESSAGE falls back to empty."""
        try:
            _suite_mod = importlib.import_module("suites.weather")
        except ImportError:
            _suite_mod = None
        system_message = getattr(_suite_mod, "SYSTEM_MESSAGE", "")
        assert isinstance(system_message, str)


class TestOpenAIResponsesAgentClient:
    """Tests for the native OpenAI Responses API client (--protocol openai, PR #94)."""

    def test_constructor_stores_fields(self):
        client = OpenAIResponsesAgentClient(
            base_url="http://localhost:8321/v1",
            model="gpt-4o-mini",
            mcp_server_url="http://localhost:8082/mcp",
            mcp_server_label="weather",
            api_key="sk-test",
            instructions="Be concise.",
            timeout=60.0,
        )
        assert client.base_url == "http://localhost:8321/v1"
        assert client.model == "gpt-4o-mini"
        assert client.mcp_server_url == "http://localhost:8082/mcp"
        assert client.mcp_server_label == "weather"
        assert client.api_key == "sk-test"
        assert client.instructions == "Be concise."
        assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_send_task_calls_responses_create(self):
        """send_task calls AsyncOpenAI.responses.create with the correct arguments."""
        import sys

        client = OpenAIResponsesAgentClient(
            base_url="http://localhost:8321/v1",
            model="gpt-4o-mini",
            mcp_server_url="http://localhost:8082/mcp",
            mcp_server_label="weather",
            api_key="x",
        )

        mock_response = MagicMock()
        mock_response.output_text = "It is 72°F and sunny in New York."

        mock_responses = MagicMock()
        mock_responses.create = AsyncMock(return_value=mock_response)

        mock_openai_instance = MagicMock()
        mock_openai_instance.responses = mock_responses
        mock_openai_instance.__aenter__ = AsyncMock(return_value=mock_openai_instance)
        mock_openai_instance.__aexit__ = AsyncMock(return_value=None)

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_instance)

        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            result = await client.send_task("What is the weather in New York?", session_token="session-a")

        assert result == "It is 72°F and sunny in New York."
        mock_responses.create.assert_called_once()
        call_kwargs = mock_responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["input"] == "What is the weather in New York?"
        assert call_kwargs["stream"] is False
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "mcp"
        assert tools[0]["server_label"] == "weather"
        assert tools[0]["server_url"] == "http://localhost:8082/mcp"
        assert tools[0]["headers"] == {"X-Midojo-Session": "session-a"}
        assert tools[0]["require_approval"] == "never"

    @pytest.mark.asyncio
    async def test_send_task_passes_instructions(self):
        """instructions is forwarded to the API call."""
        import sys

        client = OpenAIResponsesAgentClient(
            base_url="http://localhost:8321/v1",
            model="gpt-4o-mini",
            mcp_server_url="http://localhost:8082/mcp",
            instructions="You are a weather assistant.",
            api_key="sk-test",
        )
        mock_response = MagicMock(output_text="sunny")
        mock_openai_instance = MagicMock()
        mock_openai_instance.responses.create = AsyncMock(return_value=mock_response)
        mock_openai_instance.__aenter__ = AsyncMock(return_value=mock_openai_instance)
        mock_openai_instance.__aexit__ = AsyncMock(return_value=None)

        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_openai_instance)

        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            await client.send_task("weather?", session_token="session-a")

        call_kwargs = mock_openai_instance.responses.create.call_args.kwargs
        assert call_kwargs["instructions"] == "You are a weather assistant."

    @pytest.mark.asyncio
    async def test_conversation_chains_each_turn_to_the_previous_response(self):
        """Each turn after the first continues from the previous reply's response ID."""
        import sys

        calls: list[dict] = []

        class FakeResponses:
            async def create(self, **kwargs):
                calls.append(kwargs)
                n = len(calls)
                return MagicMock(id=f"resp_{n}", output_text=f"reply {n}")

        clients: list[FakeAsyncOpenAI] = []

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.responses = FakeResponses()
                self.closed = False
                clients.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                self.closed = True

        client = OpenAIResponsesAgentClient(
            base_url="http://localhost:8321/v1",
            model="gpt-4o-mini",
            mcp_server_url="http://localhost:8082/mcp",
            mcp_server_label="weather",
            instructions="Be concise.",
            api_key="x",
        )
        assert client.supports_multiturn
        with patch.dict(sys.modules, {"openai": MagicMock(AsyncOpenAI=FakeAsyncOpenAI)}):
            conversation = client.start_conversation(session_token="session-a")
            replies = [await conversation.send(message) for message in ["first", "second", "third"]]
            # One client serves every turn and stays open until the conversation closes.
            assert len(clients) == 1
            assert not clients[0].closed
            await conversation.aclose()

        assert clients[0].closed
        assert replies == ["reply 1", "reply 2", "reply 3"]
        assert [call["input"] for call in calls] == ["first", "second", "third"]
        assert "previous_response_id" not in calls[0]
        assert [call["previous_response_id"] for call in calls[1:]] == ["resp_1", "resp_2"]
        tools = calls[0]["tools"]
        assert tools[0]["headers"] == {"X-Midojo-Session": "session-a"}
        for call in calls:
            assert call["tools"] == tools
            assert (call["model"], call["instructions"], call["stream"]) == ("gpt-4o-mini", "Be concise.", False)

    @pytest.mark.asyncio
    async def test_separate_conversations_do_not_share_responses(self):
        """A new conversation, like each send_task, starts without previous_response_id."""
        import sys

        create = AsyncMock(return_value=MagicMock(id="resp_1", output_text="reply"))
        openai_module = MagicMock()
        openai_instance = openai_module.AsyncOpenAI.return_value
        openai_instance.__aenter__ = AsyncMock(return_value=openai_instance)
        openai_instance.responses.create = create
        client = OpenAIResponsesAgentClient(
            base_url="http://localhost:8321/v1", model="m", mcp_server_url="http://localhost:8082/mcp", api_key="x"
        )
        with patch.dict(sys.modules, {"openai": openai_module}):
            await client.start_conversation(session_token="a").send("one")
            await client.send_task("two", session_token="b")
            await client.start_conversation(session_token="c").send("three")
        assert all("previous_response_id" not in call.kwargs for call in create.call_args_list)
        headers = [call.kwargs["tools"][0]["headers"]["X-Midojo-Session"] for call in create.call_args_list]
        assert headers == ["a", "b", "c"]
        # One client per conversation; send_task closes its own.
        assert openai_module.AsyncOpenAI.call_count == 3
        assert openai_instance.__aexit__.await_count == 1


class _OneShotAgent(AgentClient):
    async def send_task(self, prompt, *, session_token):
        return f"{session_token}: {prompt}"


@pytest.mark.asyncio
async def test_clients_without_multiturn_send_one_message_per_conversation():
    agent = _OneShotAgent()
    assert not agent.supports_multiturn
    assert not OGXResponsesClient("http://ogx", "m", "http://mcp").supports_multiturn
    assert not OpenShellAgentClient(MagicMock()).supports_multiturn
    conversation = agent.start_conversation(session_token="session-a")
    assert await conversation.send("hello") == "session-a: hello"
    with pytest.raises(RuntimeError, match="_OneShotAgent sends one message per evaluation"):
        await conversation.send("again")
