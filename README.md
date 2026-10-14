# effortless-hermes

A Hermes port of [HeyCubit/effortless](https://github.com/HeyCubit/effortless) (MIT): a fast
judge reads each prompt once per turn and picks the reasoning effort for it — easy questions
run cheap, hard jobs get high — by rewriting the LLM request before it ships.

## How it works

- An `llm_request` middleware fires once per user turn (keyed on `turn_id`), reads the next
  user message plus recent context, asks a judge, and rewrites every reasoning-effort
  carrier already present in the provider kwargs (`reasoning_effort`, `extra_body.reasoning.effort`,
  `reasoning.effort`). It never invents a carrier a provider doesn't use, never touches
  `messages`, and never changes the model — so the prompt-cache prefix stays intact.
- On a retried request within the same turn the cached verdict is reapplied.
- A named level in the message ("max effort", "effort: high") wins without a judge; a short
  follow-up ("go", "ok") keeps the current effort; a judge that is unsure (<0.5) or down
  falls through to the fallback judge, and if that fails too the current effort is kept.
- Verdicts are logged (last 50) in profile-scoped plugin state and shown by `/effortless`
  in interactive sessions.

## Judges

| Backend | Config | Notes |
|---|---|---|
| `jev` (default) | `typesafe_key` setting (stored as `EFFORTLESS_HERMES_TYPESAFE_KEY` in `.env`), or the shared `TYPESAFE_API_KEY` env var | TypeSafe systemone, ~0.3 s, returns confidence |
| `local` (default fallback) | `local_base_url` (default `http://127.0.0.1:8080/v1`), `local_model` | any OpenAI-compatible server; must be running |
| `custom` | `custom_base_url`, `custom_model`, `custom_api_key` (stored as `EFFORTLESS_HERMES_CUSTOM_API_KEY` in `.env`) | any OpenAI-compatible API |

Secrets are read only through Hermes' profile-scoped secret store (`agent.secret_scope`),
never from `.env` files or `os.environ` directly. When no key is set, the jev judge is
skipped and the fallback chain runs; when nothing answers, the current effort is kept —
the plugin never blocks a prompt.

## Settings (`plugins.entries.effortless-hermes.settings` in config.yaml)

`enabled` (bool, true) · `judge_backend` (jev|local|custom) · `fallback_backend` (local|custom|jev|none)
· `judge_timeout_seconds` (3.0) · `effort_floor` (low) · `effort_ceiling` (max) ·
`effort_bias` (-2..+2) · judge connection keys per the table above.

## Commands

- `/effortless` — show the last verdicts
- `/effortless test` — send a sample prompt through both judges

## Security disclosure

- **Network calls**: with the default config the plugin sends your next message (up to
  4000 chars) and a short excerpt of recent conversation (up to 1600 chars) to
  `https://api.typesafe.ai/v1/systemone` (TypeSafe Jev) on the first request of each turn,
  solely to pick the effort. The `local` backend sends the same text to the base URL you
  configure (default `127.0.0.1:8080`); the `custom` backend to your configured API.
  No other data leaves the machine; there is no telemetry, no usage reporting, no file
  access beyond the plugin's own state.
- **Latency budget**: each judge is bounded by `judge_timeout_seconds` (default 3 s; local
  and custom judges allow up to 8 s). A judge that hangs is abandoned, not awaited.
- **Credentials**: keys come from the settings hub (stored in Hermes' `.env`) or the
  `TYPESAFE_API_KEY` env var, resolved through `agent.secret_scope.get_secret` so
  profile-scoped secrets stay scoped. The plugin never reads `.env` files itself and never
  writes credentials anywhere.
- **Code**: pure-stdlib Python (urllib), no pip dependencies, no shell-outs, no background
  processes, no self-updating code.

## Verified

- 2026-10-14, Hermes v0.21.7, glm-5.3 via surplusintelligence:
  "What day comes after Tuesday" → `reasoning_effort: low` in the live request dump
  (jev, 99% sure, 331 ms); "Refactor the entire auth system…" → `high` (82% sure);
  retry reapply confirmed via double dumps; `hermes plugins validate --install-deps` clean.

## Not in v0 (deliberate)

Per-prompt model routing (effortless keeps it off by default too), the desktop status bar
UI, the cache-countdown timer, handoff/compact triggers, and Anthropic `thinking` budget
carriers (providers without a text effort carrier are skipped, never guessed).

## License

MIT. Judge prompts and heuristics adapted from HeyCubit/effortless (MIT).
