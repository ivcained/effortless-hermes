"""Offline tests for effortless-hermes. Run: python3 -m pytest tests/ -q
(or python3 tests/test_plugin.py). No network, no Hermes install required for
the logic tests; live-judge tests are skipped unless their env is present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


J = _load("effortless_judge", PLUGIN / "judge.py")
core = _load("effortless_core", PLUGIN / "__init__.py")


# ── heuristics ───────────────────────────────────────────────────────────────

def test_named_effort():
    assert J.named_effort("please use max effort on this") == "max"
    assert J.named_effort("effort: high") == "high"
    assert J.named_effort("set effort to xhigh now") == "xhigh"
    assert J.named_effort("don't use max effort") is None
    assert J.named_effort("a medium sized image") is None
    assert J.named_effort("") is None


def test_follow_up():
    assert J.is_follow_up("go")
    assert J.is_follow_up("ok")
    assert J.is_follow_up("yes")
    assert J.is_follow_up("not really though") is False  # >12 chars
    assert J.is_follow_up("please continue with the work") is False


def test_keeps_effort():
    assert J.keeps_effort("go", "user: do X\nassistant: Working on it.")
    # "ok" answering a question is NOT a plain follow-up
    assert not J.keeps_effort("ok", "user: hi\nassistant: Should I archive it?")


def test_ends_on_question():
    assert J.ends_on_question("user: x\nassistant: shall I continue?")
    assert not J.ends_on_question("user: x\nassistant: done.")


# ── verdict parsing ──────────────────────────────────────────────────────────

def test_parse_verdict_text():
    assert J.parse_verdict_text('{"effort":"high","sure":0.9,"why":"refactor"}') == ("high", 0.9, "refactor")
    assert J.parse_verdict_text('Sure! {"effort":"low","sure":0.8,"why":"quick"}') == ("low", 0.8, "quick")
    assert J.parse_verdict_text('{"effort":"bogus"}') is None
    assert J.parse_verdict_text("no json here") is None
    assert J.parse_verdict_text('{"effort":"low","sure":"weird"}') == ("low", None, "")


# ── wire clamping ────────────────────────────────────────────────────────────

def test_clamp_to_wire():
    assert J.clamp_to_wire("xhigh", "glm-5.3") == "max"
    assert J.clamp_to_wire("low", "glm-5.2") == "high"
    assert J.clamp_to_wire("medium", "kimi-k3-thinking") == "high"
    assert J.clamp_to_wire("xhigh", "k3") == "max"
    assert J.clamp_to_wire("low", "some-model") == "low"
    assert J.clamp_to_wire("xhigh", "some-model") == "xhigh"
    assert J.clamp_to_wire("low", "deepseek-v4.1") == "low"


def test_wire_vocabulary():
    assert J.wire_vocabulary("GLM-5.3") == ("low", "medium", "high", "max")
    assert "xhigh" in J.wire_vocabulary("unknown-model")


def test_apply_bounds():
    assert J.apply_bounds("low", "medium", "max", 0) == "medium"    # floor
    assert J.apply_bounds("max", "low", "high", 0) == "high"        # ceiling
    assert J.apply_bounds("medium", "low", "max", -2) == "low"      # bias cheaper
    assert J.apply_bounds("low", "low", "max", 2) == "high"         # bias smarter


# ── effort carriers ──────────────────────────────────────────────────────────

def test_read_write_effort_carriers():
    req = {"reasoning_effort": "medium", "messages": []}
    assert core._read_current_effort(req) == "medium"
    assert core._write_effort(req, "high")
    assert req["reasoning_effort"] == "high"

    req2 = {"extra_body": {"reasoning": {"enabled": True, "effort": "low"}}}
    assert core._read_current_effort(req2) == "low"
    core._write_effort(req2, "max")
    assert req2["extra_body"]["reasoning"] == {"enabled": True, "effort": "max"}

    req3 = {"reasoning": {"effort": "medium"}}
    assert core._read_current_effort(req3) == "medium"

    # no carrier -> never invents one
    req4 = {"model": "x", "messages": []}
    assert core._read_current_effort(req4) is None
    assert core._write_effort(req4, "high") is False


def test_message_extraction():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": [{"type": "text", "text": "image question?"},
                                     {"type": "image_url", "image_url": {"url": "x"}}]},
    ]
    assert core._last_user_message(msgs) == "image question?"
    ctx = core._recent_context(msgs)
    assert "hello there" in ctx and "image question?" in ctx


# ── middleware decision paths (offline; judges stubbed) ─────────────────────

class FakeState:
    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


class FakeCtx:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self._state = FakeState()

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    @property
    def state(self):
        return self._state


def _req(prompt="do a big refactor of everything", effort="medium"):
    return {"model": "glm-5.3", "reasoning_effort": effort,
            "messages": [{"role": "user", "content": prompt}]}


def _run_middleware(fake_ctx, request, turn="t1", session="s1"):
    core._plugin_ctx = fake_ctx
    core._last_turn.pop(session, None)
    core._last_effort.pop(session, None)
    return core.on_llm_request(request, session_id=session, turn_id=turn,
                               model="glm-5.3", provider="x", api_mode="chat_completions",
                               api_call_count=1)


def test_middleware_named_level_skips_judge(monkeypatch_none=None):
    ctx = FakeCtx({"judge_backend": "jev"})
    req = _req("please use max effort for this migration")
    res = _run_middleware(ctx, req)
    assert req["reasoning_effort"] == "max"
    assert res and res["reason"] == "named: max"


def test_middleware_followup_keeps():
    ctx = FakeCtx({"judge_backend": "jev"})
    req = _req("go")
    res = _run_middleware(ctx, req)
    assert res is None and req["reasoning_effort"] == "medium"
    assert ctx.state.get("verdicts")[-1]["why"] == "follow-up, kept effort"


def test_middleware_disabled():
    ctx = FakeCtx({"enabled": False})
    req = _req()
    assert _run_middleware(ctx, req) is None


def test_middleware_all_judges_fail_keeps():
    ctx = FakeCtx({"judge_backend": "jev", "fallback_backend": "local",
                   "local_base_url": "http://127.0.0.1:1/v1"})  # nothing listens
    req = _req()
    res = _run_middleware(ctx, req)
    assert res is None and req["reasoning_effort"] == "medium"
    note = ctx.state.get("verdicts")[-1]["why"]
    assert "jev" in note or "no judge" in note or "unreachable" in note


def test_middleware_judges_once_per_turn():
    ctx = FakeCtx({"judge_backend": "local", "local_base_url": "http://127.0.0.1:1/v1",
                   "fallback_backend": "none"})
    req = _req()
    _run_middleware(ctx, req, turn="t1", session="sx")
    calls = []
    orig = core._run_judges
    core._run_judges = lambda *a, **k: calls.append(1) or (None, "stub")
    res = core.on_llm_request(req, session_id="sx", turn_id="t1", model="glm-5.3")
    assert not calls  # same turn: no second judge call
    core._run_judges = orig


def test_middleware_verdict_applied(monkeypatch_none=None):
    ctx = FakeCtx({"judge_backend": "jev"})
    v = J.Verdict("high", "big job", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = _req()
        res = _run_middleware(ctx, req)
        assert req["reasoning_effort"] == "high"
        assert res and "high" in res["reason"]
    finally:
        core._run_judges = orig


def test_middleware_unsure_verdict_falls_to_fallback_result():
    # primary unsure -> _run_judges already resolves; here unsure keeps current
    ctx = FakeCtx({"judge_backend": "jev"})
    v = J.Verdict("medium", "unsure, keeping", "jev", sure=0.4, unsure=True)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "fallback also unsure")
    try:
        req = _req("a tricky little thing", effort="medium")
        res = _run_middleware(ctx, req)
        assert res is None and req["reasoning_effort"] == "medium"
    finally:
        core._run_judges = orig


def test_no_carrier_no_crash():
    ctx = FakeCtx({"judge_backend": "jev"})
    v = J.Verdict("high", "big", "stub", sure=0.9)
    orig = core._run_judges
    core._run_judges = lambda cfg, p, c, x: (v, "")
    try:
        req = {"model": "some-model", "messages": [{"role": "user", "content": "hi"}]}
        res = _run_middleware(ctx, req)
        assert res is None  # nothing to rewrite -> skipped, never invented
    finally:
        core._run_judges = orig


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__} {exc}")
    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURES'}")
    sys.exit(1 if failures else 0)
