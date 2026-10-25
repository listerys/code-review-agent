# Code Review Agent

A command-line AI code reviewer for Python and Django Git repositories. It reviews committed branch changes,
finds related code across the repository, and reports potential bugs with file locations, evidence, and suggested
fixes. It is especially focused on Django behavior such as tenant isolation, signals, Celery tasks, transactions,
templates, and cache invalidation.

The agent builds a searchable code index, runs five specialist reviewers in parallel, and uses a separate verifier
to check their findings. Reviews use read-only tools: they do not modify your source code or run your application's
test suite. Results appear in the terminal and can also be saved as JSON or Markdown.

## Requirements

- Python **3.11 or newer**, with `pip` and `venv` available.
- Git installed and available on your `PATH`.
- A local Git repository with committed changes and a base branch to compare against.
- An `OPENAI_API_KEY` for AI reviews, summaries, and embeddings. Indexing and code exploration can also run
  without a key using `--no-llm`.

You do not need to install or start the application being reviewed. The agent reads code from Git revisions.

## Installation

Clone or download this project, then open a terminal in its root directory (the folder containing `pyproject.toml`).
The following commands use Bash on Linux, macOS, or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
review --help
```

The installation creates the `review` command. Activate the same virtual environment in each new terminal before
using it, or invoke the executable directly as `/path/to/code-review-agent/.venv/bin/review`.

To install the test dependencies as well:

```bash
python -m pip install -e '.[dev]'
```

Set your API key in the terminal where you will run reviews:

```bash
export OPENAI_API_KEY="your-api-key"
```

The CLI reads environment variables directly; it does not automatically load a `.env` file. AI commands make
requests to the configured API and incur usage charges.

## Run your first review

Point the CLI at the repository you want to review:

```bash
review run --repo /path/to/your/project --base main
```

This reviews the repository's current `HEAD` against its merge base with `main`. The local index is built
automatically. If you are already inside the target repository, you can omit `--repo`:

```bash
review run --base main
```

To review a specific branch and save the report:

```bash
review run --repo /path/to/your/project \
  --base main \
  --head feature/my-change \
  --json review.json \
  --markdown review.md
```

Report paths are relative to your current working directory. Existing files at those paths are overwritten.

**Only committed changes are reviewed.** Staged, unstaged, and untracked files are not included. Both revisions
must exist locally; fetch the relevant branches yourself if needed. If `--base` is omitted, the CLI tries
`origin/HEAD`, `origin/main`, `origin/master`, `main`, then `master`.

### Choose reviewers

All five specialists run by default:

| Profile | Focus |
| --- | --- |
| `impact_mapper` | Related callers, routes, templates, admin screens, and tests affected by the change. |
| `bug_hunter` | Correctness bugs, edge cases, and behavior that callers depend on. |
| `state_regression` | Signals, Celery tasks, transactions, cache keys, and ordering assumptions. |
| `security_integrity` | Authorization, tenant isolation, query scoping, money handling, and injection. |
| `test_verifier` | Existing test coverage, missing tests, and similar historical bugs. |

Repeat `--profile` (or `-p`) to select a subset:

```bash
review run -r /path/to/your/project --base main \
  -p bug_hunter -p security_integrity
```

The verifier runs by default. Add `--no-verify` to skip it; those findings are marked unverified.

Each finding includes a location, severity, category, explanation, evidence, and suggested fix. Categories include
`bug.*`, `security.*`, `state.*`, `performance.*`, and `test_gap.*`. Reports also retain rejected and invalid
candidates so you can inspect what was filtered out.

## Try the included demo

The bundled Tinyshop demo creates a small Django Git repository with a `main` branch and a
`feature/partial-refunds` branch containing four deliberate bugs. Choose a new or empty directory:

```bash
review demo create /tmp/tinyshop-demo
review run -r /tmp/tinyshop-demo --base main
```

To explore the demo without an API key or model calls:

```bash
review index -r /tmp/tinyshop-demo --base main --no-llm
review search "refund" -r /tmp/tinyshop-demo --base main --mode lexical --no-llm
review context -r /tmp/tinyshop-demo --base main -p state_regression --no-llm --text
```

This builds the index, searches it, and shows the context a reviewer would receive. Generating an actual AI review
still requires an API key. You do not need to run the demo's Django server or set up a database.

## Indexing and code exploration

The index combines a graph of code relationships with BM25 keyword search and optional semantic embeddings.
To generate summaries and embeddings before reviewing a repository:

```bash
review index -r /path/to/your/project --base main
review run -r /path/to/your/project --base main
```

Index enrichment uses the API. Without embeddings, reviews still use the graph and keyword search. Use
`review index --no-llm` for local indexing, or `--no-summaries` and `--no-embeddings` to control enrichment separately.

Other useful commands:

```bash
review search "refund validation" -r /path/to/your/project --base main --no-llm
review context -r /path/to/your/project --base main -p bug_hunter --no-llm --text
review stats -r /path/to/your/project --base main
review tools
```

`review search` supports `--mode hybrid`, `lexical`, `dense`, or `graph`. `review context` previews the code packed
for a specialist, and `review tools` lists the read-only tools available to agents. Use `review COMMAND --help`
for the full options of any command.

For asynchronous enrichment, submit work with `review index --batch -r /path/to/your/project --base main`, then run
`review batch-collect -r /path/to/your/project` after the batches complete. Add `--memory 20` to `review index` to
mine up to 20 recent bug-fix commits into the history corpus.

## Configuration

### Environment variables

These defaults come from the project's configuration:

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | API credential for model requests. | Unset |
| `OPENAI_BASE_URL` | Override the API endpoint; it must support the APIs used by the selected commands. | SDK default |
| `CRA_REVIEW_MODEL` | Model used by specialist reviewers and the verifier. | `gpt-5` |
| `CRA_MINI_MODEL` | Model used for summaries and reranking. | `gpt-5-mini` |
| `CRA_EMBEDDING_MODEL` | Embedding model for semantic retrieval. | `text-embedding-3-large` |
| `CRA_EMBEDDING_DIMS` | Number of embedding dimensions. | `1024` |
| `CRA_CACHE_DIR` | Directory for repository indexes and cached artifacts. | `~/.cache/code-review-agent` |

The default cache directory respects `XDG_CACHE_HOME` when set. Use `review stats -r /path/to/your/project` to see
the repository's index path and recorded usage.

### User configuration

Create `~/.config/code-review-agent/config.toml` (or `$XDG_CONFIG_HOME/code-review-agent/config.toml` if set) to
override defaults:

```toml
[models]
review_model = "gpt-5"
review_reasoning_effort = "medium"
llm_concurrency = 8

[agent]
specialist_max_turns = 12
verifier_max_turns = 16
verifier_batch_size = 10
max_review_cost_usd = 2.0
```

Model environment variables take precedence over this file. `max_review_cost_usd` is a soft threshold based on
the project's configured price estimates: agents stop requesting tools and finish their responses once it is
reached. It is not a hard billing cap; parallel requests and final responses can exceed it, and separate index
enrichment has its own usage.

### Repository configuration

Add a `review.toml` at the root of the repository being reviewed:

```toml
[index]
exclude = ["vendor/**", "generated/**"]
max_file_bytes = 4000000

[django]
tenant_models = ["Tenant"]
tenant_fields = ["tenant", "tenant_id"]
```

Only `[index]`, `[django]`, and `[money]` are accepted from repository configuration. These sections override the
corresponding user settings. The file is read from the Git revision being indexed, so commit it before expecting
the settings to take effect. See [config.py](src/code_review_agent/config.py) for all settings and defaults.

## How it works

1. Read the diff between the merge base and the selected head revision, withholding secret-like files.
2. Parse code into chunks and build relationships between calls, routes, signals, tasks, templates, cache usage,
   and tests.
3. Retrieve relevant context for each specialist using the graph, keyword search, and available embeddings.
4. Run specialist agents in parallel. They inspect code using read-only tools and return structured findings.
5. Drop invalid file or line references, merge overlapping findings, and have the verifier confirm or reject the
   remaining candidates.
6. Print the results and optionally write JSON and Markdown reports.

Model-bound text passes through a secret scrubber, and Responses API requests set `store=False`. AI operations
send code context to the configured API endpoint. Commands supporting `--no-llm` disable model requests while
retaining local analysis. Indexes and cached artifacts are stored on disk in the cache directory.

## Evaluate review quality

Run the demo evaluation to check the four injected bugs and one planted false alarm (requires an API key):

```bash
review eval demo
```

You can also mine reference cases from a repository's past bug-fix commits:

```bash
review eval mine -r /path/to/your/project
review eval list -r /path/to/your/project
```

Inspect the generated case files at the path printed by `eval mine`, confirm their expected findings, and set
`"labels_confirmed": true` in each case's JSON data. Then run:

```bash
review eval retrieval -r /path/to/your/project --no-llm
review eval findings -r /path/to/your/project
```

Retrieval evaluation measures Recall@k and MRR. Findings evaluation uses AI reviews to measure recall and
precision against the known bug locations and requires an API key.

## Development and tests

From this project's root, with the virtual environment activated:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

The tests use a scripted fake OpenAI server and require no API key.

## Troubleshooting

- **`review: command not found`:** Activate the environment with `source .venv/bin/activate`, or use the full path
  to `.venv/bin/review`. Make sure you installed the package in that environment.
- **`OPENAI_API_KEY is not set`:** Export the key in the same terminal session. For local exploration, use commands
  that support `--no-llm`.
- **No embeddings for this revision:** Run `review index` with an API key and the same `--base` and `--head` values.
  You can also continue reviewing with graph and keyword retrieval alone.
- **Cannot find a target branch or revision:** Pass an explicit `--base` that exists locally. Fetch missing remote
  branches if needed, and ensure the repository has enough history to find a merge base.
- **Your latest changes are missing:** Commit them and check the `--head` revision; working-tree changes are not
  part of the review.
