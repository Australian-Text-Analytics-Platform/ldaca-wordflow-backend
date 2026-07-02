"""Server-side AI annotation engine.

Used by:
- The Annotation router's ``/annotation/ai/models``, ``/annotation/ai/preview``,
  and ``/annotation/ai/annotate-all`` endpoints (see
  ``api/workspaces/annotation.py``) because all provider/LLM traffic now runs on
  the backend: the browser only ever talks to our own API, never directly to a
  model provider. This module is the single place that knows how to speak each
  provider's wire format.

Why it exists:
- The Annotation tab previously called provider SDKs straight from the browser,
  which leaked API keys into client code, hit CORS edge cases, and stalled on the
  SDKs' 10-minute default timeout with no server visibility. Centralising the
  calls here lets us bound timeouts/retries, fan batches out concurrently, and
  keep prompt-building + label-coercion identical across providers.

Flow (per call):
- Resolve the provider's wire format from its id (built-in) or an
  OpenAI-compatible base URL (user-defined custom provider).
- Build a shared system prompt (instruction + labelled class list + strict JSON
  contract) and a user prompt (the texts as a JSON array, order preserved).
- Dispatch one request per batch through the provider's *native* async SDK
  (``AsyncOpenAI`` for openai/openrouter/custom, ``AsyncAnthropic`` for anthropic,
  ``google-genai`` aio for google).
- Loosely parse the JSON reply and coerce every returned label to a known class
  name (case-insensitive) or ``None``, always returning exactly one label per
  input text so callers can map results back to rows positionally.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Literal, cast

# Per-request network timeout. Chosen well below the provider SDKs' 10-minute
# default so a slow/rate-limited provider surfaces as a bounded error instead of
# a silent multi-minute hang (the browser-side stall this refactor fixes).
REQUEST_TIMEOUT_SECONDS = 90.0
# One transient retry inside the SDK (429/5xx/timeout). Kept low so a persistently
# failing provider fails fast rather than multiplying the timeout by the retry count.
MAX_RETRIES = 1
# Rows per provider request. 20 keeps per-request token cost/latency reasonable
# while still classifying a meaningful chunk in one round trip.
DEFAULT_BATCH_SIZE = 20
# Ceiling on batches in flight at once for annotate-all. Bounded so a large table
# fans out concurrently (not one batch after another) without hammering the
# provider into rate limits.
MAX_CONCURRENCY = 6

AnnotationChatStyle = Literal["openai", "anthropic", "google"]


class AnnotationAiError(Exception):
    """A provider/LLM call failed (auth, rate limit, network, bad response).

    Raised by the dispatch helpers so the router can translate any provider
    failure into a single ``BadGatewayError`` (HTTP 502) with the provider's
    message, instead of leaking SDK-specific exception types to the API surface.
    """


@dataclass(frozen=True)
class ProviderWire:
    """How one provider is addressed on the wire for the annotation call.

    Used by:
    - ``resolve_provider_wire`` (produces it) and the ``_complete_*`` /
      ``list_models`` dispatchers (consume it) because those are the only places
      that need to branch on chat format, base URL, and JSON-mode support.
    """

    chat_style: AnnotationChatStyle
    # OpenAI-SDK ``base_url``: OpenRouter points at its ``/api/v1``; OpenAI uses
    # the SDK default (None); custom providers carry their ``https://host/v1``.
    # Ignored by the anthropic/google styles, whose SDKs target their own hosts.
    base_url: str | None
    # True when the provider honours OpenAI's ``response_format=json_object`` for
    # guaranteed-valid JSON. Only the hosted OpenAI-style providers set it.
    supports_json_response_format: bool


# Built-in provider catalogue, mirrored from the frontend's ANNOTATION_AI_PROVIDERS
# so a provider id resolves to the same wire format on both sides.
BUILTIN_PROVIDER_WIRE: dict[str, ProviderWire] = {
    "openrouter": ProviderWire("openai", "https://openrouter.ai/api/v1", True),
    "openai": ProviderWire("openai", None, True),
    "anthropic": ProviderWire("anthropic", None, False),
    "google": ProviderWire("google", None, False),
}


def resolve_provider_wire(provider_id: str, base_url: str | None) -> ProviderWire:
    """Return the wire format for a provider id, falling back to a custom endpoint.

    Called by:
    - Every AI endpoint before dispatching, because the request only carries the
      provider id (+ a base URL for user-defined providers); the backend owns the
      mapping from id to chat format so the frontend never has to.

    Built-ins resolve from ``BUILTIN_PROVIDER_WIRE``. Any other id is treated as a
    user-defined, OpenAI-compatible custom provider: it speaks the ``openai``
    chat style against the supplied ``base_url`` (trailing slashes trimmed so the
    SDK can append ``/chat/completions`` cleanly) with JSON mode left off since a
    custom endpoint's support is unknown.
    """
    builtin = BUILTIN_PROVIDER_WIRE.get(provider_id)
    if builtin is not None:
        return builtin
    normalized = (base_url or "").rstrip("/") or None
    return ProviderWire("openai", normalized, False)


# Reasoning-effort levels, ordered low→high. Kept in sync with the frontend's
# AnnotationInferenceSettings dropdown so a persisted value always resolves to a
# level every provider can honour.
REASONING_EFFORTS: tuple[str, ...] = ("low", "medium", "high")
DEFAULT_REASONING_EFFORT = "medium"
# Extended-thinking token budget per effort level for the providers whose SDKs
# take a raw budget (Anthropic, Google). OpenAI-style providers instead accept
# the effort string directly, so they ignore this table.
_REASONING_BUDGET_TOKENS: dict[str, int] = {"low": 1024, "medium": 4096, "high": 12000}
# Head-room added above a thinking budget for the visible answer, since a
# provider's max output tokens must exceed the tokens it may spend thinking.
ANSWER_TOKEN_HEADROOM = 4096


@dataclass(frozen=True)
class InferenceConfig:
    """Provider-agnostic sampling/reasoning knobs for one annotation request.

    Used by:
    - ``annotate_batch`` / ``annotate_all`` (threaded into every ``_complete_*``
      dispatcher) because the Annotation tab's collapsible "Model Configuration"
      section lets the user pick a sampling temperature and optionally enable
      reasoning with a thinking effort. The value travels from the preview and
      annotate-all request bodies so the same knobs apply whether one page or the
      whole column is classified.

    Why it exists:
    - Temperature and reasoning are the two knobs that differ per provider SDK
      (OpenAI takes ``reasoning_effort``; Anthropic/Google take a thinking-token
      budget), so centralising them in one immutable value keeps the dispatchers
      from each re-deriving the mapping and keeps the defaults (temperature 0,
      reasoning off) identical to the pre-feature behaviour.

    Fields:
    - ``temperature``: sampling temperature (0 = deterministic, the default).
    - ``reasoning_enabled``: when False (default) no reasoning/thinking params are
      sent, so non-reasoning models keep working exactly as before.
    - ``reasoning_effort``: one of ``REASONING_EFFORTS``; mapped to the provider's
      native reasoning control only when ``reasoning_enabled`` is True.
    """

    temperature: float = 0.0
    reasoning_enabled: bool = False
    reasoning_effort: str = DEFAULT_REASONING_EFFORT

    @classmethod
    def from_request(
        cls, temperature: float, reasoning_enabled: bool, reasoning_effort: str
    ) -> InferenceConfig:
        """Build a config from raw request fields, clamped to safe ranges.

        Called by the ``/annotation/ai/preview`` and ``/annotation/ai/annotate-all``
        endpoints so an out-of-range temperature or unknown effort from the wire can
        never reach a provider SDK: temperature is clamped to ``[0, 2]`` (the range
        every supported provider accepts) and the effort falls back to the default
        when it is not a recognised level.
        """
        clamped = min(2.0, max(0.0, temperature))
        effort = reasoning_effort.strip().lower()
        if effort not in REASONING_EFFORTS:
            effort = DEFAULT_REASONING_EFFORT
        return cls(
            temperature=clamped,
            reasoning_enabled=reasoning_enabled,
            reasoning_effort=effort,
        )


def _reasoning_budget_tokens(effort: str) -> int:
    """Map a reasoning effort level to a provider thinking-token budget.

    Called by the Anthropic and Google dispatchers, whose SDKs take a raw token
    budget rather than an effort string. Unknown levels fall back to the medium
    budget so a bad value degrades gracefully instead of raising.
    """
    return _REASONING_BUDGET_TOKENS.get(effort, _REASONING_BUDGET_TOKENS["medium"])


@dataclass(frozen=True)
class AnnotationClassOption:
    """A class the model may assign, with an optional guiding description.

    Used by:
    - ``build_annotation_system_prompt`` to render the labelled choice list and by
      ``annotate_batch`` to derive the canonical class names label coercion maps
      onto. Loaded server-side from the class-description node so the valid label
      set stays authoritative regardless of what the client sends.
    """

    name: str
    description: str = ""


def build_annotation_system_prompt(
    instruction: str, classes: list[AnnotationClassOption]
) -> str:
    """Assemble the system message: instruction + labelled classes + JSON contract.

    Called by ``annotate_batch``. Kept byte-for-byte aligned with the frontend's
    former ``buildAnnotationSystemPrompt`` so moving the call server-side does not
    change model behaviour: the instruction leads, each class is listed (with its
    description when present), and a strict ``{"labels": [...]}`` output rule keeps
    the reply machine-parseable across every provider.
    """
    class_lines = "\n".join(
        f"- {option.name}: {option.description.strip()}"
        if option.description.strip()
        else f"- {option.name}"
        for option in classes
    )
    return "\n".join(
        [
            instruction.strip(),
            "",
            "Classify each input text into exactly one of these classes:",
            class_lines,
            "",
            "Rules:",
            "- Use the exact class name shown above for each text.",
            "- If no class applies, use null.",
            "- Respond with ONLY a JSON object of the form {\"labels\": [...]} containing one",
            "  entry per input text, in the same order. No prose, no markdown.",
        ]
    )


def build_annotation_user_prompt(texts: list[str]) -> str:
    """Render the page texts as a JSON array so special characters stay unambiguous.

    Called by ``annotate_batch``. Mirrors the frontend's former
    ``buildAnnotationUserPrompt``; the count is stated up front and the order is
    preserved so the model's positional ``labels`` array lines up with the rows.
    """
    return "\n".join(
        [
            f"Classify these {len(texts)} texts (JSON array, preserve order):",
            json.dumps(texts),
        ]
    )


def loose_parse_json(content: str) -> object | None:
    """Best-effort parse of a model reply into JSON.

    Called by ``align_labels``. Ports the frontend's ``looseParseJson``: strip a
    markdown code fence, try a direct parse, then fall back to the first
    ``{...}``/``[...]`` span so a chatty model that wraps the payload in prose
    still yields usable output. Returns ``None`` when nothing parses.
    """
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        # Drop an optional language tag on the opening fence (e.g. ```json).
        newline = stripped.find("\n")
        if newline != -1 and " " not in stripped[:newline]:
            stripped = stripped[newline + 1 :]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    stripped = stripped.strip()

    def attempt(text: str) -> object | None:
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return None

    direct = attempt(stripped)
    if direct is not None:
        return direct
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = stripped.find(open_char)
        end = stripped.rfind(close_char)
        if start != -1 and end > start:
            span = attempt(stripped[start : end + 1])
            if span is not None:
                return span
    return None


def align_labels(
    content: str, count: int, class_names: list[str]
) -> list[str | None]:
    """Coerce a model reply into exactly ``count`` known-class-or-null labels.

    Called by ``annotate_batch``. Ports the frontend's ``alignLabels``: parse the
    JSON, read a ``labels`` array (or a bare array), and map each entry
    case-insensitively to a canonical class name or ``None``. Always returns
    ``count`` entries so a short/long reply still lines up with the rows
    positionally, and unknown labels degrade to ``None`` rather than corrupting
    the column.
    """
    parsed = loose_parse_json(content)
    source = (
        cast("dict[str, object]", parsed).get("labels")
        if isinstance(parsed, dict)
        else parsed
    )
    raw_labels = source if isinstance(source, list) else []
    canonical = {name.strip().lower(): name for name in class_names}
    result: list[str | None] = []
    for index in range(count):
        raw = raw_labels[index] if index < len(raw_labels) else None
        if isinstance(raw, str):
            result.append(canonical.get(raw.strip().lower()))
        else:
            result.append(None)
    return result


async def _complete_openai(
    wire: ProviderWire,
    model: str,
    api_key: str,
    system: str,
    user: str,
    config: InferenceConfig,
) -> str:
    """Run the completion through the native OpenAI SDK (also OpenRouter/custom).

    Called by ``annotate_batch`` for the ``openai`` chat style. The three
    OpenAI-compatible providers differ only by ``base_url``; a placeholder key is
    used for keyless endpoints (the SDK rejects an empty string). JSON mode is
    requested when the provider supports it. Any SDK error is wrapped in
    ``AnnotationAiError`` so the router can return a single 502 shape.

    Reasoning: OpenAI's reasoning models constrain temperature (many accept only
    their default), so when the user enables reasoning we send the native
    ``reasoning_effort`` and let the model default its own temperature; otherwise
    we honour the chosen temperature and send no reasoning param.
    """
    from openai import AsyncOpenAI, Omit, omit
    from openai.types import ReasoningEffort
    from openai.types.chat import ChatCompletionMessageParam

    client = AsyncOpenAI(
        api_key=api_key or "no-key-required",
        base_url=wire.base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    reasoning_effort: ReasoningEffort | Omit = (
        cast(ReasoningEffort, config.reasoning_effort)
        if config.reasoning_enabled
        else omit
    )
    temperature: float | Omit = omit if config.reasoning_enabled else config.temperature
    try:
        # Two call shapes (rather than a splat) keep JSON mode strongly typed:
        # only the hosted OpenAI-style providers accept response_format.
        #
        # ``stream=False`` is passed explicitly (not just left to default) because
        # some OpenAI-compatible servers — notably Apple's on-device ``fm serve`` —
        # stream a Server-Sent-Events body whenever the request omits ``stream``.
        # The SDK's non-streaming path then tries to JSON-decode that SSE text and
        # yields a bare ``str`` (``'str' object has no attribute 'choices'``).
        # Sending the literal ``False`` forces a single JSON completion for every
        # provider and is a no-op for the hosted ones that already default to it.
        if wire.supports_json_response_format:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                response_format={"type": "json_object"},
                stream=False,
            )
        else:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                stream=False,
            )
    except Exception as error:  # noqa: BLE001 - normalise every SDK failure shape
        raise AnnotationAiError(str(error) or "OpenAI request failed") from error
    return completion.choices[0].message.content or ""


async def _complete_anthropic(
    model: str, api_key: str, system: str, user: str, config: InferenceConfig
) -> str:
    """Run the completion through the native Anthropic SDK.

    Called by ``annotate_batch`` for the ``anthropic`` chat style. The system
    prompt is passed as Anthropic's top-level ``system`` field; text blocks in the
    reply are concatenated into the raw JSON payload. Errors are wrapped in
    ``AnnotationAiError``.

    Reasoning: when enabled we turn on extended thinking with a token budget mapped
    from the effort level, raise ``max_tokens`` above that budget so there is room
    for the answer, and omit temperature (Anthropic requires the default
    temperature while thinking is on). When disabled we honour the temperature and
    send no thinking config.
    """
    from anthropic import AsyncAnthropic, Omit, omit
    from anthropic.types import TextBlock, ThinkingConfigParam

    client = AsyncAnthropic(
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )
    max_tokens = 4096
    thinking: ThinkingConfigParam | Omit = omit
    temperature: float | Omit = config.temperature
    if config.reasoning_enabled:
        budget = _reasoning_budget_tokens(config.reasoning_effort)
        max_tokens = budget + ANSWER_TOKEN_HEADROOM
        thinking = {"type": "enabled", "budget_tokens": budget}
        temperature = omit
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            thinking=thinking,
        )
    except Exception as error:  # noqa: BLE001 - normalise every SDK failure shape
        raise AnnotationAiError(str(error) or "Anthropic request failed") from error
    return "".join(
        block.text for block in message.content if isinstance(block, TextBlock)
    )


async def _complete_google(
    model: str, api_key: str, system: str, user: str, config: InferenceConfig
) -> str:
    """Run the completion through the native Google GenAI SDK (async client).

    Called by ``annotate_batch`` for the ``google`` chat style. The system prompt
    becomes ``system_instruction`` and ``response_mime_type`` asks Gemini for raw
    JSON. Errors are wrapped in ``AnnotationAiError``.

    Reasoning: Gemini allows temperature alongside thinking, so temperature is
    always sent; a ``ThinkingConfig`` with the effort's token budget is attached
    only when reasoning is enabled (otherwise the provider default applies).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    thinking_config = (
        types.ThinkingConfig(
            thinking_budget=_reasoning_budget_tokens(config.reasoning_effort)
        )
        if config.reasoning_enabled
        else None
    )
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=config.temperature,
                response_mime_type="application/json",
                thinking_config=thinking_config,
            ),
        )
    except Exception as error:  # noqa: BLE001 - normalise every SDK failure shape
        raise AnnotationAiError(str(error) or "Google request failed") from error
    return response.text or ""


async def annotate_batch(
    wire: ProviderWire,
    model: str,
    api_key: str,
    instruction: str,
    classes: list[AnnotationClassOption],
    texts: list[str],
    config: InferenceConfig = InferenceConfig(),
) -> list[str | None]:
    """Classify one batch of texts in a single provider request.

    Called by:
    - The ``/annotation/ai/preview`` endpoint (one page of texts) and by
      ``annotate_all`` (once per batch). Builds the shared prompt, dispatches to
      the matching native SDK by ``wire.chat_style``, then coerces the reply to one
      known-class-or-null label per text, aligned to input order.

    ``config`` carries the temperature/reasoning knobs; it defaults to
    deterministic sampling with reasoning off so callers that do not care (and the
    existing tests) keep the original behaviour.
    """
    if not texts:
        return []
    system = build_annotation_system_prompt(instruction, classes)
    user = build_annotation_user_prompt(texts)
    if wire.chat_style == "anthropic":
        content = await _complete_anthropic(model, api_key, system, user, config)
    elif wire.chat_style == "google":
        content = await _complete_google(model, api_key, system, user, config)
    else:
        content = await _complete_openai(wire, model, api_key, system, user, config)
    return align_labels(content, len(texts), [option.name for option in classes])


async def annotate_all(
    wire: ProviderWire,
    model: str,
    api_key: str,
    instruction: str,
    classes: list[AnnotationClassOption],
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = MAX_CONCURRENCY,
    config: InferenceConfig = InferenceConfig(),
) -> list[str | None]:
    """Classify every text by fanning batches out concurrently, order preserved.

    Called by:
    - The ``/annotation/ai/annotate-all`` endpoint because a full-column run must
      dispatch its batches concurrently (not one after another) yet return labels
      in the original row order so the whole column can be written in one go.

    Flow:
    - Split ``texts`` into ``batch_size`` chunks.
    - Run them through ``annotate_batch`` (with the shared ``config``) under an
      ``asyncio.Semaphore`` cap so at most ``concurrency`` requests are in flight
      (avoids provider rate limits).
    - ``asyncio.gather`` preserves submission order, so flattening the per-batch
      results reproduces the input order exactly.
    """
    if not texts:
        return []
    size = max(1, batch_size)
    chunks = [texts[start : start + size] for start in range(0, len(texts), size)]
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(chunk: list[str]) -> list[str | None]:
        async with semaphore:
            return await annotate_batch(
                wire, model, api_key, instruction, classes, chunk, config
            )

    batches = await asyncio.gather(*(run(chunk) for chunk in chunks))
    return [label for batch in batches for label in batch]


def _strip_google_model_prefix(name: str) -> str:
    """Drop Gemini's ``models/`` namespace so the id matches the generate call.

    Called by ``list_models`` for the google style, matching the frontend's former
    ``parseGoogleModels`` normalisation.
    """
    prefix = "models/"
    return name[len(prefix) :] if name.startswith(prefix) else name


async def list_models(provider_id: str, base_url: str | None, api_key: str) -> list[str]:
    """List a provider's available model ids through its native SDK.

    Called by:
    - The ``/annotation/ai/models`` endpoint, which backs the model-picker
      dropdown. Returns a de-duplicated, alphabetically sorted id list.

    Built-in providers list through their native SDK. User-defined custom
    providers are OpenAI-compatible, so they are listed through the OpenAI SDK's
    ``/models`` route against their own base URL (many local servers — e.g. Apple's
    ``fm serve``, Ollama, LM Studio, vLLM — expose it). A custom provider with no
    base URL is skipped (returns ``[]``) so the SDK never falls back to
    ``api.openai.com`` with a placeholder key; if a custom endpoint simply lacks a
    ``/models`` route the SDK error propagates as ``AnnotationAiError`` and the
    dropdown surfaces it while the field's free-text entry still works. SDK errors
    (auth/network) propagate the same way for built-ins.
    """
    wire = resolve_provider_wire(provider_id, base_url)
    is_custom = provider_id not in BUILTIN_PROVIDER_WIRE
    # A custom provider is only listable when it carries a base URL; without one the
    # OpenAI SDK would target api.openai.com with the placeholder key and 401.
    if is_custom and wire.base_url is None:
        return []
    ids: set[str] = set()
    try:
        if wire.chat_style == "anthropic":
            from anthropic import AsyncAnthropic

            anthropic_client = AsyncAnthropic(
                api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES
            )
            async for model in anthropic_client.models.list():
                if isinstance(model.id, str) and model.id:
                    ids.add(model.id)
        elif wire.chat_style == "google":
            from google import genai

            google_client = genai.Client(api_key=api_key)
            async for model in await google_client.aio.models.list():
                name = getattr(model, "name", None)
                if isinstance(name, str) and name:
                    ids.add(_strip_google_model_prefix(name))
        else:
            from openai import AsyncOpenAI

            openai_client = AsyncOpenAI(
                api_key=api_key or "no-key-required",
                base_url=wire.base_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )
            async for model in openai_client.models.list():
                if isinstance(model.id, str) and model.id:
                    ids.add(model.id)
    except AnnotationAiError:
        raise
    except Exception as error:  # noqa: BLE001 - normalise every SDK failure shape
        raise AnnotationAiError(str(error) or "Model listing failed") from error
    return sorted(ids, key=str.lower)
