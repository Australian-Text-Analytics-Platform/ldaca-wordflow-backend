"""Unit probes for the AI-annotation inference knobs and OpenAI SDK dispatch.

Covers the pure ``InferenceConfig`` factory and the effort→budget mapping in
``infrastructure/providers/annotation_ai.py``. These need no network or workspace, so they guard the
clamping/normalisation rules the endpoints rely on (see the endpoint tests for
the wire-through) without spinning up the app.

The second half swaps in a fake ``openai.AsyncOpenAI`` client (the SDK is imported
lazily inside the functions, so patching ``openai.AsyncOpenAI`` intercepts it) to
verify the OpenAI path always sends ``stream=False`` so the non-streaming SDK
path cannot receive an SSE response unexpectedly.
"""

import asyncio

import pytest

from ldaca_wordflow.infrastructure.providers.annotation_ai import (
    AnnotationContextLimitError,
    DEFAULT_REASONING_EFFORT,
    InferenceConfig,
    _complete_openai,
    annotate_all,
    _reasoning_budget_tokens,
    list_models,
    resolve_provider_wire,
)


def test_inference_config_defaults_match_prior_behaviour():
    config = InferenceConfig()
    assert config.temperature == 0.0
    assert config.reasoning_enabled is False
    assert config.reasoning_effort == DEFAULT_REASONING_EFFORT


def test_from_request_clamps_temperature_to_supported_range():
    assert InferenceConfig.from_request(-1.0, False, "medium").temperature == 0.0
    assert InferenceConfig.from_request(5.0, False, "medium").temperature == 2.0
    assert InferenceConfig.from_request(0.4, False, "medium").temperature == 0.4


def test_from_request_normalizes_effort_and_falls_back():
    assert InferenceConfig.from_request(0.0, True, "HIGH").reasoning_effort == "high"
    assert InferenceConfig.from_request(0.0, True, " low ").reasoning_effort == "low"
    # Unknown levels degrade to the default rather than reaching a provider SDK.
    assert (
        InferenceConfig.from_request(0.0, True, "turbo").reasoning_effort
        == DEFAULT_REASONING_EFFORT
    )


def test_reasoning_budget_tokens_orders_low_below_high():
    low = _reasoning_budget_tokens("low")
    medium = _reasoning_budget_tokens("medium")
    high = _reasoning_budget_tokens("high")
    assert low < medium < high
    # An unrecognised level uses the medium budget as a safe default.
    assert _reasoning_budget_tokens("bogus") == medium


def test_custom_provider_wire_uses_the_immutable_openai_compatible_base_url():
    wire = resolve_provider_wire("custom", "http://127.0.0.1:8080/v1")

    assert wire.chat_style == "openai"
    assert wire.base_url == "http://127.0.0.1:8080/v1"
    assert wire.supports_json_response_format is False


# --- OpenAI SDK dispatch probes -------------------------------------------------
#
# The endpoint tests monkeypatch ``annotate_batch``/``list_models`` wholesale, so
# the real OpenAI-path streaming branch has no
# other coverage. The fakes below stand in for the parts of the async SDK those
# branches touch: ``chat.completions.create`` returns one completion (recording its
# kwargs so we can assert ``stream=False``) and ``models.list()`` yields an async
# iterator of id-bearing objects like the SDK's paginator.


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeModel:
    def __init__(self, model_id: str) -> None:
        self.id = model_id


class _AsyncModelIterator:
    """Async-iterable stand-in for the SDK's ``models.list()`` paginator."""

    def __init__(self, ids: list[str]) -> None:
        self._models = [_FakeModel(model_id) for model_id in ids]

    def __aiter__(self):
        async def _gen():
            for model in self._models:
                yield model

        return _gen()


def _install_fake_openai(monkeypatch, *, model_ids: list[str] | None = None) -> dict:
    """Patch ``openai.AsyncOpenAI`` with a fake and return the create-kwargs sink.

    The returned dict is populated with whatever ``chat.completions.create`` is
    called with, letting a test assert the wire shape (notably ``stream``).
    """
    create_kwargs: dict = {}

    class _FakeCompletions:
        async def create(self, **kwargs):
            create_kwargs.update(kwargs)
            return _FakeCompletion("positive")

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeModels:
        def list(self):
            return _AsyncModelIterator(model_ids or [])

    class _FakeAsyncOpenAI:
        def __init__(self, **_kwargs) -> None:
            self.chat = _FakeChat()
            self.models = _FakeModels()

    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
    return create_kwargs


async def test_complete_openai_always_disables_streaming(monkeypatch):
    # Pin stream=False so a server cannot hand back an SSE body unexpectedly.
    create_kwargs = _install_fake_openai(monkeypatch)
    wire = resolve_provider_wire("openai")
    result = await _complete_openai(
        wire, "some-model", "key", "system", "user", InferenceConfig()
    )
    assert result == "positive"
    assert create_kwargs["stream"] is False


async def test_complete_openai_classifies_context_limit_errors(monkeypatch):
    class _ContextLimitedCompletions:
        async def create(self, **_kwargs):
            raise RuntimeError("maximum context length exceeded")

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _ContextLimitedCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self, **_kwargs) -> None:
            self.chat = _FakeChat()

    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)

    with pytest.raises(AnnotationContextLimitError):
        await _complete_openai(
            resolve_provider_wire("openai"),
            "some-model",
            "key",
            "system",
            "user",
            InferenceConfig(),
        )


async def test_custom_model_discovery_uses_its_base_url_and_allows_no_key(
    monkeypatch,
):
    constructor_kwargs: dict = {}

    class _FakeModels:
        def list(self):
            return _AsyncModelIterator(["local-model"])

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            constructor_kwargs.update(kwargs)
            self.models = _FakeModels()

    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
    wire = resolve_provider_wire("custom", "http://localhost:8080/v1")

    models = await list_models(wire, None)

    assert models == ["local-model"]
    assert constructor_kwargs["base_url"] == "http://localhost:8080/v1"
    assert constructor_kwargs["api_key"] == "no-key-required"


async def test_custom_chat_completion_uses_its_base_url_and_allows_no_key(
    monkeypatch,
):
    constructor_kwargs: dict = {}
    create_kwargs: dict = {}

    class _FakeCompletions:
        async def create(self, **kwargs):
            create_kwargs.update(kwargs)
            return _FakeCompletion('{"labels": ["positive"]}')

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            constructor_kwargs.update(kwargs)
            self.chat = _FakeChat()

    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
    wire = resolve_provider_wire("custom", "http://localhost:8080/v1")

    result = await _complete_openai(
        wire, "local-model", None, "system", "user", InferenceConfig()
    )

    assert result == '{"labels": ["positive"]}'
    assert constructor_kwargs["base_url"] == "http://localhost:8080/v1"
    assert constructor_kwargs["api_key"] == "no-key-required"
    assert create_kwargs["model"] == "local-model"


async def test_annotate_all_uses_one_hundred_row_batches_with_ten_in_flight(
    monkeypatch,
):
    active = 0
    max_active = 0
    chunk_sizes: list[int] = []

    async def fake_annotate_batch(
        _wire,
        _model,
        _api_key,
        _instruction,
        _classes,
        texts,
        _config,
        _examples,
    ):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        chunk_sizes.append(len(texts))
        await asyncio.sleep(0.01)
        active -= 1
        return list(texts)

    monkeypatch.setattr(
        "ldaca_wordflow.infrastructure.providers.annotation_ai.annotate_batch",
        fake_annotate_batch,
    )

    texts = [str(index) for index in range(1001)]
    labels = await annotate_all(
        resolve_provider_wire("openai"),
        "some-model",
        "key",
        "instruction",
        [],
        texts,
    )

    assert labels == texts
    assert sorted(chunk_sizes) == [1, *([100] * 10)]
    assert max_active == 10


async def test_annotate_all_splits_only_batches_rejected_by_the_context_limit(
    monkeypatch,
):
    attempted_sizes: list[int] = []

    async def fake_annotate_batch(
        _wire,
        _model,
        _api_key,
        _instruction,
        _classes,
        texts,
        _config,
        _examples,
    ):
        attempted_sizes.append(len(texts))
        if len(texts) > 25:
            raise AnnotationContextLimitError("maximum context length exceeded")
        return list(texts)

    monkeypatch.setattr(
        "ldaca_wordflow.infrastructure.providers.annotation_ai.annotate_batch",
        fake_annotate_batch,
    )

    texts = [str(index) for index in range(100)]
    labels = await annotate_all(
        resolve_provider_wire("openai"),
        "some-model",
        "key",
        "instruction",
        [],
        texts,
    )

    assert labels == texts
    assert sorted(attempted_sizes) == [25, 25, 25, 25, 50, 50, 100]
