"""effortless-hermes — per-prompt reasoning effort, picked by a fast judge.

A Hermes port of HeyCubit/effortless (MIT). An `llm_request` middleware reads the
next user message once per turn, asks a judge (Jev/TypeSafe, a local
OpenAI-compatible model, or any custom API) for the effort, and rewrites the
effort carrier in the provider request. Only the effort for that one request
changes — the conversation, model, and prompt-cache prefix stay untouched
(unless the user opts into `model_map`, which trades cache for a cheaper model
on fresh chats only).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from . import judge as J

logger = logging.getLogger("hermes.plugins.effortless-hermes")

_MAX_CONTEXT_MSGS = 6
_MAX_CONTEXT_CHARS = 1600
_RECENT_VERDICTS = 50
_RECENT_TURNS = 50

_lock = threading.Lock()
_last_turn: dict[str, str] = {}          # session_id -> turn_id already judged
_last_effort: dict[str, str] = {}        # session_id -> effort applied this turn (reapplied on retries)
_last_model: dict[str, str] = {}         # session_id -> model override applied this turn
_plugin_ctx: Any = None                  # for the /effortless command

# v0.4 outcome feedback: a failed turn escalates the session's effort, a
# streak of clean turns decays it back. Quality-class failures only
# (incomplete/empty responses, finish_reason=error) — rate limits, auth,
# billing and request-shape 400s never get "fixed" with more thinking.
_MAX_SESSIONS = 200
_FAILURE_BIAS_MAX = 2
_CLEAN_STREAK_TARGET = 2
_ESCALATE_REASONS = frozenset({"incomplete_response", "empty_response"})
_session_bias: dict[str, int] = {}       # session_id -> +steps on the effort ladder
_clean_streak: dict[str, int] = {}       # session_id -> consecutive clean first-calls


def _prune_sessions() -> None:
    """Bound the per-session maps: a long-lived gateway sees unbounded sessions."""
    for d in (_last_turn, _last_effort, _last_model, _session_bias, _clean_streak):
        if len(d) > _MAX_SESSIONS:
            for stale in list(d)[:-_MAX_SESSIONS]:
                d.pop(stale, None)


def _bump_failure(session_id: str, why: str) -> None:
    with _lock:
        bias = min(_session_bias.get(session_id, 0) + 1, _FAILURE_BIAS_MAX)
        _session_bias[session_id] = bias
        _clean_streak[session_id] = 0
        _prune_sessions()
    try:
        _record(_plugin_ctx, None, "", f"failure feedback: session bias +{bias} ({why})")
    except Exception:
        pass


def _note_clean_turn(session_id: str) -> None:
    with _lock:
        if session_id not in _session_bias:
            return  # nothing to decay
        streak = _clean_streak.get(session_id, 0) + 1
        if streak >= _CLEAN_STREAK_TARGET:
            bias = _session_bias.get(session_id, 0)
            if bias <= 1:
                _session_bias.pop(session_id, None)
                _clean_streak.pop(session_id, None)
                return
            _session_bias[session_id] = bias - 1
            streak = 0
        _clean_streak[session_id] = streak


def _feedback_enabled() -> bool:
    ctx_obj = _plugin_ctx
    if ctx_obj is None:
        return True
    try:
        return bool(_cfg(ctx_obj, "feedback_enabled", True))
    except Exception:
        return True


def on_api_request_error(**payload: Any) -> None:
    """Observer hook: a quality-class provider failure escalates the session's
    effort for the next judged turn (and for a retry of the failed one)."""
    try:
        if not _feedback_enabled():
            return
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            return
        reason = str(payload.get("reason") or "")
        if reason not in _ESCALATE_REASONS:
            return
        _bump_failure(session_id[:40], reason)
    except Exception:
        logger.debug("effortless-hermes: failure feedback tracking failed", exc_info=True)


# ── message helpers ──────────────────────────────────────────────────────────

def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def _messages(request: dict) -> list:
    msgs = request.get("messages")
    if not isinstance(msgs, list):
        msgs = request.get("input")
    return msgs if isinstance(msgs, list) else []


def _last_user_message(msgs: list) -> str:
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            return _text_of(m.get("content"))
    return ""


def _recent_context(msgs: list) -> str:
    lines = []
    for m in msgs[-_MAX_CONTEXT_MSGS:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        text = _text_of(m.get("content")).strip()
        if not text and isinstance(m.get("tool_calls"), list):
            text = f"[{len(m['tool_calls'])} tool call(s)]"
        if text:
            lines.append(f"{role}: {text[:300]}")
    context = "\n".join(lines)
    return context[-_MAX_CONTEXT_CHARS:]


def _count_user_turns(msgs: list) -> int:
    return sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "user")


# ── effort carriers ──────────────────────────────────────────────────────────

def _read_current_effort(request: dict) -> Optional[str]:
    v = request.get("reasoning_effort")
    if isinstance(v, str) and v:
        return v.lower()
    eb = request.get("extra_body")
    if isinstance(eb, dict):
        r = eb.get("reasoning")
        if isinstance(r, dict) and isinstance(r.get("effort"), str):
            return r["effort"].lower()
    r = request.get("reasoning")
    if isinstance(r, dict) and isinstance(r.get("effort"), str):
        return r["effort"].lower()
    return None


def _write_effort(request: dict, effort: str) -> bool:
    """Rewrite every effort carrier already present. Never invents a new one."""
    wrote = False
    if isinstance(request.get("reasoning_effort"), str):
        request["reasoning_effort"] = effort
        wrote = True
    eb = request.get("extra_body")
    if isinstance(eb, dict) and isinstance(eb.get("reasoning"), dict):
        eb["reasoning"]["effort"] = effort
        eb["reasoning"]["enabled"] = True
        wrote = True
    r = request.get("reasoning")
    if isinstance(r, dict) and isinstance(r.get("effort"), str):
        r["effort"] = effort
        wrote = True
    return wrote


# ── verdict + turn logs (durable, profile-scoped) ────────────────────────────

def _record(ctx_obj: Any, verdict: Optional[J.Verdict], prompt: str, note: str,
            extra: Optional[dict] = None) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "effort": verdict.effort if verdict else None,
        "by": verdict.by if verdict else None,
        "sure": verdict.sure if verdict else None,
        "why": verdict.why if verdict else note,
        "ms": verdict.ms if verdict else 0,
        "prompt": prompt[:120],
        "note": note,
    }
    if extra:
        entry.update(extra)
    try:
        state = ctx_obj.state
        history = state.get("verdicts") or []
        history.append(entry)
        state.set("verdicts", history[-_RECENT_VERDICTS:])
    except Exception:
        logger.debug("effortless-hermes: could not persist verdict", exc_info=True)


def on_post_api_request(**payload: Any) -> None:
    """Observer hook: keep the last turns' token usage for stats and the context
    meter, and feed the v0.4 outcome loop (finish_reason=error escalates the
    session's effort; clean turns decay it)."""
    try:
        if int(payload.get("api_call_count") or 0) != 1:
            return  # one record per turn (the first request carries the full context)
        session_id = str(payload.get("session_id") or "")[:40]
        finish_reason = str(payload.get("finish_reason") or "")
        if session_id and _feedback_enabled():
            if finish_reason == "error":
                _bump_failure(session_id, "finish_reason=error")
            else:
                _note_clean_turn(session_id)
        usage = payload.get("usage") or {}

        def _tok(*names: str) -> Optional[int]:
            for n in names:
                v = usage.get(n)
                if isinstance(v, (int, float)) and v > 0:
                    return int(v)
            return None

        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "model": str(payload.get("model") or "")[:60],
            "finish_reason": finish_reason or None,
            "input_tokens": _tok("prompt_tokens", "input_tokens"),
            "output_tokens": _tok("completion_tokens", "output_tokens"),
            "total_tokens": _tok("total_tokens"),
            "context_length": payload.get("context_length"),
            "duration_s": payload.get("api_duration"),
        }
        ctx_obj = _plugin_ctx
        if ctx_obj is None:
            return
        state = ctx_obj.state
        turns = state.get("turns") or []
        turns.append({k: v for k, v in entry.items() if v is not None})
        state.set("turns", turns[-_RECENT_TURNS:])
    except Exception:
        logger.debug("effortless-hermes: post_api_request tracking failed", exc_info=True)


def _turn_summary(ctx_obj: Any) -> Optional[dict]:
    try:
        turns = (ctx_obj.state.get("turns") or [])
        return turns[-1] if turns else None
    except Exception:
        return None


def _context_fill(turn: Optional[dict]) -> Optional[int]:
    if not turn:
        return None
    tok = turn.get("input_tokens")
    ctx_len = turn.get("context_length")
    if isinstance(tok, int) and isinstance(ctx_len, int) and ctx_len > 0:
        return min(100, round(tok / ctx_len * 100))
    return None


# ── config ───────────────────────────────────────────────────────────────────

def _cfg(ctx_obj: Any, key: str, default: Any) -> Any:
    try:
        value = ctx_obj.get_config(key, default)
    except Exception:
        value = default
    return default if value is None or value == "" else value


_CFG_DEFAULTS = [
    ("enabled", True), ("judge_backend", "jev"), ("fallback_backend", "local"),
    ("jev_url", "https://api.typesafe.ai/v1/systemone"), ("jev_model", "jev-latest"),
    ("local_base_url", "http://127.0.0.1:8080/v1"), ("local_model", ""),
    ("custom_base_url", ""), ("custom_model", ""), ("judge_timeout_seconds", 3.0),
    ("effort_floor", "low"), ("effort_ceiling", "max"), ("effort_bias", 0),
    ("model_map", {}), ("model_map_always", False), ("feedback_enabled", True),
]


def _load_cfg(ctx_obj: Any) -> dict:
    return {key: _cfg(ctx_obj, key, default) for key, default in _CFG_DEFAULTS}


# ── the judge pipeline ───────────────────────────────────────────────────────

def _run_judges(cfg: dict, prompt: str, current_effort: Optional[str],
                context: str) -> tuple[Optional[J.Verdict], str]:
    """Primary judge, then the fallback when the primary is unsure/unavailable."""
    primary = str(cfg.get("judge_backend") or "jev").lower()
    fallback = str(cfg.get("fallback_backend") or "local").lower()
    errors: list[str] = []

    def backend(name: str) -> Optional[J.Verdict]:
        if name in ("none", ""):
            return None
        if name == "jev":
            key = J.find_typesafe_key()
            if not key:
                errors.append("jev: no TYPESAFE_API_KEY")
                return None
            try:
                return J.ask_jev(cfg, prompt, current_effort, context, key)
            except Exception as exc:
                errors.append(str(exc))
                return None
        if name in ("local", "custom"):
            try:
                return J.ask_openai_compat(cfg, name, prompt, current_effort, context)
            except Exception as exc:
                errors.append(str(exc))
                return None
        errors.append(f"unknown backend {name}")
        return None

    verdict = backend(primary)
    if verdict is None or verdict.unsure:
        second = backend(fallback)
        if second is not None and not second.unsure:
            return second, ("; ".join(errors) if verdict is None else
                            f"{primary} unsure -> {fallback}")
    if verdict is not None:
        return verdict, "; ".join(errors)
    return None, "; ".join(errors) or "no judge answered"


def _apply_model_map(cfg: dict, request: dict, msgs: list, effort: str) -> Optional[str]:
    """Opt-in effort→model mapping. Returns the model id applied, or None.

    Cache safety: a model change invalidates the provider prompt cache for the
    rest of the conversation, so by default this only fires on a fresh chat
    (first user turn). `model_map_always: true` opts into the cache cost.
    Mapped ids must be servable by the chat's own provider.
    """
    model_map = cfg.get("model_map")
    if not isinstance(model_map, dict) or not model_map:
        return None
    target = model_map.get(effort)
    if not isinstance(target, str) or not target.strip():
        return None
    target = target.strip()
    current = str(request.get("model") or "")
    if current == target:
        return None  # already on the mapped model
    always = bool(cfg.get("model_map_always"))
    if not always and _count_user_turns(msgs) > 1:
        return None  # established chat: switching would re-read it at full price
    request["model"] = target
    return target


def on_llm_request(request: dict, **ctx: Any) -> Optional[dict]:
    """`llm_request` middleware: judge once per user turn, rewrite the effort carrier."""
    try:
        return _on_llm_request(request, **ctx)
    except Exception:
        logger.debug("effortless-hermes: middleware error (request unchanged)", exc_info=True)
        return None


def _on_llm_request(request: dict, **ctx: Any) -> Optional[dict]:
    global _plugin_ctx
    ctx_obj = _plugin_ctx
    if ctx_obj is None:
        return None
    cfg = _load_cfg(ctx_obj)
    if not cfg["enabled"]:
        return None

    session_id = str(ctx.get("session_id") or "?")
    turn_id = str(ctx.get("turn_id") or "")
    base_url = str(ctx.get("base_url") or "")

    floor = str(cfg["effort_floor"]); ceiling = str(cfg["effort_ceiling"])
    feedback_on = bool(cfg.get("feedback_enabled", True))
    with _lock:
        session_bias = _session_bias.get(session_id, 0) if feedback_on else 0
        _prune_sessions()

    # sanitize an effort the provider would reject (e.g. a config-level
    # 'medium' on a route that only accepts low/high/max). Runs on EVERY
    # request — follow-ups, judge-down turns and retries can ship a 400 too,
    # not just the first call of a turn.
    sanitized_reason = ""
    _early_effort = _read_current_effort(request)
    if _early_effort:
        _early_model = str(request.get("model") or ctx.get("model") or "")
        fixed = J.clamp_to_wire(_early_effort, _early_model, base_url,
                                request.get("tools"))
        if fixed != _early_effort and _write_effort(request, fixed):
            sanitized_reason = f"sanitized {_early_effort} -> {fixed} (outside provider knob)"

    def _ret(result: Optional[dict], note: str = "") -> Optional[dict]:
        """Never drop a sanitized (mutated) request on a None return."""
        if result is not None:
            return result
        if sanitized_reason:
            return {"request": request, "source": "effortless-hermes",
                    "reason": note or sanitized_reason}
        return None

    with _lock:
        if _last_turn.get(session_id) == turn_id and turn_id:
            # already judged this turn; a retry rebuilt the request, so reapply
            cached_effort = _last_effort.get(session_id)
            cached_model = _last_model.get(session_id)
            # the rebuilt request may be agentic (tools) or on a different
            # route than the one the effort was judged for — re-clamp it
            # against the current request, never paste it verbatim.
            if cached_effort:
                r_model = str(request.get("model") or ctx.get("model") or "")
                clamped = J.clamp_to_wire(
                    cached_effort, r_model, base_url, request.get("tools"))
                cached_effort = clamped
                if session_bias:
                    # the turn failed after it was judged: the retry ships one
                    # step MORE thinking, not the same verdict that just failed
                    cached_effort = J.apply_bounds(
                        cached_effort, floor, ceiling, session_bias)
                    cached_effort = J.clamp_to_wire(
                        cached_effort, r_model, base_url, request.get("tools"))
            changed = False
            if cached_effort and _write_effort(request, cached_effort):
                changed = True
            if cached_model and request.get("model") != cached_model:
                request["model"] = cached_model
                changed = True
            if changed:
                reason = f"reapply {cached_effort} (retried request)"
                if session_bias:
                    reason += f" (+{session_bias} failure bias)"
                return {"request": request, "source": "effortless-hermes",
                        "reason": reason}
            return _ret(None)  # tool-loop follow-ups keep the request as built
        _last_turn[session_id] = turn_id

    if sanitized_reason:
        _record(ctx_obj, None, "", sanitized_reason)
    msgs = _messages(request)
    if not msgs:
        return _ret(None)
    prompt = _last_user_message(msgs)
    if not prompt.strip():
        return _ret(None)
    context = _recent_context(msgs)
    current_effort = _read_current_effort(request)
    model = str(request.get("model") or ctx.get("model") or "")
    request_tools = request.get("tools")

    def _clamp(effort: str) -> str:
        return J.clamp_to_wire(effort, model, base_url, request_tools)

    # judge verdict + any failure-feedback bias from this session's outcomes
    bias = int(cfg["effort_bias"] or 0) + session_bias

    # 1. named level in the message wins, no judge
    named = J.named_effort(prompt)
    if named:
        effort = J.apply_bounds(named, floor, ceiling, bias)
        effort = _clamp(effort)
        if _write_effort(request, effort):
            _last_effort[session_id] = effort
            applied_model = _apply_model_map(cfg, request, msgs, effort)
            if applied_model:
                _last_model[session_id] = applied_model
            _record(ctx_obj, None, prompt, f"named in message -> {effort}",
                    extra={"model_override": applied_model})
            reason = f"named: {effort}" + (f" · model: {applied_model}" if applied_model else "")
            return {"request": request, "source": "effortless-hermes", "reason": reason}
        return _ret(None)

    # 2. short follow-up keeps the current effort
    if current_effort and J.keeps_effort(prompt, context):
        _last_effort[session_id] = current_effort
        _record(ctx_obj, None, prompt, "follow-up, kept effort")
        return _ret(None)

    # 3. ask the judges
    verdict, note = _run_judges(cfg, prompt, current_effort, context)
    if verdict is None:
        _record(ctx_obj, None, prompt, note)
        return _ret(None)

    effort = J.apply_bounds(verdict.effort, floor, ceiling, bias)
    effort = _clamp(effort)
    if effort == current_effort:
        _last_effort[session_id] = effort
        _record(ctx_obj, verdict, prompt, "same as current, unchanged")
        return _ret(None)
    if _write_effort(request, effort):
        _last_effort[session_id] = effort
        applied_model = _apply_model_map(cfg, request, msgs, effort)
        if applied_model:
            _last_model[session_id] = applied_model
        _record(ctx_obj, verdict, prompt, f"-> {effort} (was {current_effort})",
                extra={"model_override": applied_model})
        reason = f"{verdict.by}: {effort}"
        if session_bias:
            reason += f" (+{session_bias} failure bias)"
        if verdict.why and verdict.why != "unsure, keeping":
            reason += f" ({verdict.why})"
        if applied_model:
            reason += f" · model: {applied_model}"
        return {"request": request, "source": "effortless-hermes", "reason": reason}
    _record(ctx_obj, verdict, prompt, "no effort carrier on this provider; skipped")
    return _ret(None)


# ── /effortless command ──────────────────────────────────────────────────────

def _fmt_stats(ctx_obj: Any) -> str:
    try:
        verdicts = (ctx_obj.state.get("verdicts") or [])[-_RECENT_VERDICTS:]
        turns = (ctx_obj.state.get("turns") or [])[-_RECENT_TURNS:]
    except Exception:
        verdicts, turns = [], []
    judged = [v for v in verdicts if v.get("by")]
    lines = [f"verdicts recorded: {len(verdicts)} (judged: {len(judged)})"]
    if judged:
        by_effort: dict[str, int] = {}
        by_judge: dict[str, int] = {}
        lat_ms = []
        for v in judged:
            by_effort[str(v.get("effort") or "?")] = by_effort.get(str(v.get("effort") or "?"), 0) + 1
            by_judge[str(v.get("by") or "?")] = by_judge.get(str(v.get("by") or "?"), 0) + 1
            if isinstance(v.get("ms"), int) and v["ms"] > 0:
                lat_ms.append(v["ms"])
        lines.append("by effort : " + ", ".join(f"{k}×{v}" for k, v in sorted(by_effort.items())))
        lines.append("by judge  : " + ", ".join(f"{k}×{v}" for k, v in sorted(by_judge.items())))
        if lat_ms:
            lines.append(f"judge latency: avg {sum(lat_ms) // len(lat_ms)} ms over {len(lat_ms)} calls")
    if turns:
        toks = [t["input_tokens"] for t in turns if isinstance(t.get("input_tokens"), int)]
        fills = [f for f in (_context_fill(t) for t in turns) if f is not None]
        if toks:
            lines.append(f"context: avg {sum(toks) // len(toks):,} input tokens over {len(toks)} turns")
        if fills:
            lines.append(f"context fill: latest {fills[-1]}%, avg {sum(fills) // len(fills)}%")
        lines.append("no cost-savings estimate: what a prompt would have cost "
                     "at another effort is not known")
    last = _turn_summary(ctx_obj)
    if last:
        fill = _context_fill(last)
        lines.append(f"last turn: {last.get('model', '?')} · {last.get('input_tokens', '?')} in"
                     + (f" · {fill}% of window" if fill is not None else ""))
    biased = {s: b for s, b in _session_bias.items() if b}
    if biased:
        recent = ", ".join(f"{s[:12]}… +{b}" for s, b in list(biased.items())[-5:])
        lines.append(f"failure bias: {len(biased)} session(s) ({recent})")
    return "\n".join(lines)


def _command_effortless(raw_args: str = "") -> str:
    arg = (raw_args or "").strip().lower()
    ctx_obj = _plugin_ctx
    if ctx_obj is None:
        return "effortless-hermes: not loaded."
    cfg = _load_cfg(ctx_obj)
    if arg in ("test", "check"):
        out = [f"primary judge: {cfg['judge_backend']}, fallback: {cfg['fallback_backend']}"]
        sample = "rename one variable in a single file"
        try:
            v, note = _run_judges(cfg, sample, "medium", "")
            out.append(f"test prompt: {sample!r}")
            out.append(f"verdict: {v!r}" if v else f"no verdict: {note}")
        except Exception as exc:
            out.append(f"test failed: {exc}")
        return "\n".join(out)
    if arg == "stats":
        return "effortless-hermes — stats (recent window)\n" + _fmt_stats(ctx_obj)
    lines = []
    last = _turn_summary(ctx_obj)
    if last:
        fill = _context_fill(last)
        lines.append(f"context: {last.get('input_tokens', '?')} in"
                     + (f" · {fill}% of window" if fill is not None else ""))
        lines.append("")
    history = []
    try:
        history = (ctx_obj.state.get("verdicts") or [])[-10:]
    except Exception:
        pass
    if not history:
        lines.append("No verdicts yet. Next prompt will be judged.")
        lines.append("Try '/effortless test' (judges) or '/effortless stats'.")
        return "\n".join(lines)
    lines.append("Last verdicts (newest last):")
    for h in history:
        lines.append(f"  {h['ts']}  {str(h['effort'] or '-'):6} {str(h['by'] or '-'):6} "
                     f"{h['why']}  | {h['prompt'][:60]}")
    return "\n".join(lines)


# ── registration ─────────────────────────────────────────────────────────────

def register(ctx) -> None:
    global _plugin_ctx
    _plugin_ctx = ctx
    ctx.register_middleware("llm_request", on_llm_request)
    try:
        ctx.register_hook("post_api_request", on_post_api_request)
    except Exception:
        logger.debug("effortless-hermes: post_api_request hook registration failed", exc_info=True)
    try:
        ctx.register_hook("api_request_error", on_api_request_error)
    except Exception:
        logger.debug("effortless-hermes: api_request_error hook registration failed", exc_info=True)
    try:
        ctx.register_command("effortless", _command_effortless,
                             description="effortless-hermes: verdicts, 'test' the judges, "
                                         "'stats' for the session window",
                             argument_mode="text")
    except Exception:
        logger.debug("effortless-hermes: command registration failed", exc_info=True)
    logger.info("effortless-hermes registered (judge=%s fallback=%s)",
                ctx.get_config("judge_backend", "jev"), ctx.get_config("fallback_backend", "local"))
