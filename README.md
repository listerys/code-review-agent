# code-review-agent

AI code review for Python/Django branches. It indexes the repository into a code graph (calls, routes, signals,
Celery tasks, templates, cache keys, tests) with BM25 and optional embeddings. Five specialist agents then review
the branch diff in parallel with read-only tools, and a verifier agent re-checks every candidate finding before it
is reported.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
export OPENAI_API_KEY=sk-...
```

## Review a branch

```bash
review run -r path/to/repo --base main
review run -r path/to/repo --base main --json review.json --markdown review.md
```

- `--profile/-p` runs a subset of specialists: `impact_mapper`, `bug_hunter`, `state_regression`,
  `security_integrity`, `test_verifier`.
- `--no-verify` skips the verifier; findings are then marked unverified.
- The index is built automatically. Run `review index -r path/to/repo --base main` first to add summaries and
  embeddings (dense retrieval); without them, graph and BM25 retrieval still work.

Each finding has a location, severity, category (`bug.*`, `security.*`, `state.*`, `performance.*`, `test_gap.*`),
explanation, evidence, suggested fix, and the verifier's reason. Rejected and invalid candidates are listed
separately so you can see what was filtered out.

## How a review runs

1. The diff between the merge base and `--head` is loaded; secret-like files are withheld.
2. For each specialist, the retrieval engine packs the most relevant code for its focus into a token budget.
3. The specialists run in parallel on the OpenAI Responses API. Each can call 16 read-only tools
   (`review tools` lists them) until it answers with structured findings.
4. Findings pointing at files or lines that do not exist are dropped. Findings with the same file, category, and
   overlapping lines are merged.
5. The verifier checks each remaining candidate with the same tools and confirms or rejects it.

All text sent to the model passes through the secret scrubber. Requests use `store=False`.

## Configuration

Optional `~/.config/code-review-agent/config.toml`:

```toml
[models]
review_model = "gpt-5"            # or set CRA_REVIEW_MODEL
review_reasoning_effort = "medium"

[agent]
specialist_max_turns = 12
verifier_max_turns = 16
verifier_batch_size = 10
max_review_cost_usd = 2.0         # once reached, agents must answer without more tool calls
```

A repository can set `[index]`, `[django]`, and `[money]` in a `review.toml` at its root.

## Evaluate review quality

```bash
review eval demo                   # builds the tinyshop demo, reviews it, checks the 4 injected bugs and 1 planted false alarm
review eval mine -r repo           # mine golden cases from past bug-fix commits
review eval findings -r repo       # full reviews on golden cases: recall and precision against the fixed lines
review eval retrieval -r repo      # retrieval-only metrics (Recall@k, MRR)
```

`review demo create DIR` creates the demo repository to try things by hand.

## Tests

```bash
.venv/bin/python -m pytest
```

The tests run against a scripted fake OpenAI server and need no API key.
