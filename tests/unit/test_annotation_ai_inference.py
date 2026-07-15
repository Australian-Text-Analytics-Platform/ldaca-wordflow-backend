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

from ldaca_wordflow.infrastructure.providers.annotation_ai import (
    DEFAULT_REASONING_EFFORT,
    InferenceConfig,
    _complete_openai,
    _reasoning_budget_tokens,
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
