"""
BaseAnalyst: shared LLM call logic for all specialist analysts.

Extracted from 001_llm_agent_analyser.py and 003_llm_group_analyser.py,
which share ~70% of their annotation/audit/narrative pipeline.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path

from terralingua.anthropologist.analysis_utils import load_agent_log
from terralingua.utils.llm_client import LLMClient, Response
from terralingua.utils.llm_utils import MAX_CONTEXT_TOKENS, is_context_enough

TAGS_PATH = Path(__file__).resolve().parents[1] / "tags.json"

log = logging.getLogger(__name__)


class BaseAnalyst:
    """Shared LLM call logic for all specialist analysts."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        provider: str = "anthropic",
        audit: bool = True,
        parallel: bool = True,
        force_long_context: bool = False,
        fallback_model: str | None = None,
    ):
        self.model = model
        self.provider = provider
        self.audit = audit
        self.parallel = parallel
        self.force_long_context = force_long_context
        # If the primary model raises (e.g. content-policy refusal), the call
        # is retried once with this smaller/different model. None disables
        # fallback (existing behavior).
        self.fallback_model = fallback_model
        self._tags = self._load_tags()

    def _load_tags(self) -> dict:
        with open(TAGS_PATH, "r") as f:
            return json.load(f)

    def _make_llm_client(self, messages: list[dict]) -> LLMClient:
        """Create an LLMClient with long_context auto-selected based on message size."""
        if self.force_long_context:
            return LLMClient(client=self.provider, long_context=True)
        if is_context_enough(
            messages=messages,
            max_input_tokens=MAX_CONTEXT_TOKENS[self.model]["base"],
            model=self.model,
        ):
            return LLMClient(client=self.provider, long_context=False)
        log.info("[%s] Using long-context model (prompt too large for base)", self.__class__.__name__)
        return LLMClient(client=self.provider, long_context=True)

    async def _call_llm_with_fallback(
        self,
        messages: list[dict],
        chat_params: dict,
        output_json: bool,
        api_key: str | None = None,
    ) -> Response:
        """Try self.model first; on any exception fall back to self.fallback_model.

        This catches content-policy refusals (BadRequestError) and JSON-parse
        failures after retry exhaustion. Each attempt still gets the full
        retry budget of get_response_async — fallback fires only after the
        primary's retries are exhausted.
        """
        llm = self._make_llm_client(messages)
        try:
            return await llm.get_response_async(
                model=self.model,
                messages=messages,
                chat_parameters=chat_params,
                output_json=output_json,
                api_key=api_key,
            )
        except Exception as e:
            if self.fallback_model is None or self.fallback_model == self.model:
                raise
            log.warning(
                "[%s] %s failed (%s); retrying with fallback %s",
                self.__class__.__name__, self.model, type(e).__name__,
                self.fallback_model,
            )
            return await llm.get_response_async(
                model=self.fallback_model,
                messages=messages,
                chat_parameters=chat_params,
                output_json=output_json,
                api_key=api_key,
            )

    async def _run_annotation(
        self,
        messages: list[dict],
        chat_params: dict | None = None,
        api_key: str | None = None,
    ) -> Response:
        """Single LLM annotation call (JSON output)."""
        return await self._call_llm_with_fallback(
            messages, chat_params or {}, output_json=True, api_key=api_key,
        )

    async def _run_audit(
        self,
        messages: list[dict],
        chat_params: dict | None = None,
        api_key: str | None = None,
    ) -> Response:
        """Single LLM audit call (JSON output). Uses same context logic."""
        return await self._call_llm_with_fallback(
            messages, chat_params or {}, output_json=True, api_key=api_key,
        )

    async def _get_narrative(
        self,
        messages: list[dict],
        chat_params: dict | None = None,
        api_key: str | None = None,
    ) -> Response:
        """LLM free-text narrative call (non-JSON output)."""
        params = deepcopy(chat_params or {})
        params.pop("response_format", None)
        return await self._call_llm_with_fallback(
            messages, params, output_json=False, api_key=api_key,
        )

    def _apply_audit_revisions(self, annotations: dict, audits: dict) -> dict:
        """Apply audit revise/fail verdicts to annotations in-place. Returns modified dict."""
        to_remove = {"behaviors": [], "events": []}
        for ann_type in ["behaviors", "events"]:
            for audit in audits.get(f"{ann_type}_audit", []):
                verdict = audit.get("verdict", "pass")
                confidence = int(audit.get("confidence", 0))
                log.debug("  %s audit verdict: %s (confidence %d)", ann_type, verdict, confidence)
                if verdict == "revise" and confidence > 6:
                    proposed_fix = audit.get("proposed_fix", {})
                    index = audit.get("index")
                    if index is not None and 0 <= index < len(
                        annotations.get(ann_type, [])
                    ):
                        log.debug("  Annotation: %s", annotations[ann_type][index])
                        for key in proposed_fix:
                            if (
                                proposed_fix[key] is not None
                                and key in annotations[ann_type][index]
                            ):
                                annotations[ann_type][index][key] = proposed_fix[key]
                        log.debug("  -> REVISED")
                elif verdict == "fail" and confidence > 6:
                    index = audit.get("index")
                    if index is not None:
                        log.debug(
                            "  Annotation: %s",
                            annotations[ann_type][index]
                            if index < len(annotations.get(ann_type, []))
                            else "(index OOB)",
                        )
                        to_remove[ann_type].append(index)
                        log.debug("  -> REMOVED")

        for ann_type in ["behaviors", "events"]:
            for index in sorted(to_remove[ann_type], reverse=True):
                if 0 <= index < len(annotations.get(ann_type, [])):
                    annotations[ann_type].pop(index)

        return annotations

    def _load_agent_log_windowed(
        self,
        path: Path,
        time_range: tuple[int, int] | None = None,
    ) -> dict:
        """Load an agent JSONL log, optionally filtering to a timestep range."""
        data = load_agent_log(path, reduce=True)
        if time_range is not None:
            start, end = time_range
            data = {k: v for k, v in data.items() if start <= k <= end}
        return data
