from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..demo import FEATURE_BRANCH, INJECTED_BUGS, create_demo_repo
from ..evals.golden import load_cases, mine_cases
from ..evals.retrieval import ABLATIONS, run_retrieval_eval, save_report
from ..git.repo import GitRepo
from ..index.builder import IndexBuilder, open_store, repo_cache_dir, settings_for_repo
from ..index.enrich import Enricher
from ..index.memory import build_memory
from ..llm.client import LLMClient
from ..retrieval.models import SearchFilters
from ..retrieval.profiles import PROFILES
from ..session import ReviewSession
from ..tools.executor import ToolCall, ToolExecutor
from ..tools.impl import build_registry

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Repository intelligence and hybrid retrieval for code review agents.")
eval_app = typer.Typer(no_args_is_help=True, help="Golden-set mining and retrieval evaluation.")
demo_app = typer.Typer(no_args_is_help=True, help="Create the tinyshop demo repository.")
app.add_typer(eval_app, name="eval")
app.add_typer(demo_app, name="demo")
console = Console()

RepoOption = typer.Option(".", "--repo", "-r", help="Path inside the git repository to review")
BaseOption = typer.Option(None, "--base", "-b", help="Target branch to diff against (default: origin/HEAD, main, or master)")
HeadOption = typer.Option("HEAD", "--head", help="Revision to review")


def progress(message: str) -> None:
    console.print(f"[dim]· {message}[/dim]")


def open_session(repo: str, base: str | None, head: str, use_llm: bool) -> ReviewSession:
    return ReviewSession.open(repo, base=base, head=head, use_llm=use_llm, progress=progress)


def print_usage(llm: LLMClient | None) -> None:
    if llm is None or llm.totals.calls == 0:
        return
    totals = llm.totals
    console.print(
        f"[bold]LLM usage[/bold]: {totals.calls} calls, {totals.input_tokens:,} input tokens "
        f"({totals.cached_tokens:,} cached), {totals.output_tokens:,} output tokens, ${totals.cost_usd:.4f}"
    )


@app.command()
def index(
    repo: str = RepoOption,
    base: str = BaseOption,
    head: str = HeadOption,
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip summaries and embeddings; nothing leaves the machine"),
    summaries: bool = typer.Option(True, "--summaries/--no-summaries", help="Generate one-line chunk summaries with the mini model"),
    embeddings: bool = typer.Option(True, "--embeddings/--no-embeddings", help="Embed code and summary views of each chunk"),
    batch: bool = typer.Option(False, "--batch", help="Submit summaries and embeddings through the Batch API instead of synchronously"),
    memory: int = typer.Option(0, "--memory", help="Also mine this many recent bug-fix commits into the memory corpus"),
    dims: int = typer.Option(None, "--dims", help="Override embedding dimensions (builds a separate vector table)"),
    force: bool = typer.Option(False, "--force", help="Rebuild views even if cached"),
):
    started = time.monotonic()
    git_repo = GitRepo(repo)
    settings = settings_for_repo(git_repo, head, use_llm=not no_llm)
    if dims:
        settings.models.embedding_dims = dims
    builder = IndexBuilder(git_repo, settings, progress=progress)
    head_sha = git_repo.rev_parse(head)
    target = base or git_repo.default_target()
    base_sha = git_repo.merge_base(git_repo.rev_parse(target), head_sha)
    view, report = builder.build_overlay(base_sha, head_sha, force=force)
    if memory:
        build_memory(git_repo, builder.store, memory, head_sha, progress)

    table = Table(title=f"Index for {git_repo.root.name}", show_header=False)
    table.add_row("base (merge-base)", f"{base_sha[:12]} ({target})")
    table.add_row("head", head_sha[:12])
    table.add_row("view", view.view_key[:60])
    table.add_row("changed files", str(len(view.changed_paths())))
    table.add_row("chunks in view", f"{report.chunks_in_view:,}")
    table.add_row("files parsed now", str(report.parsed_now))
    if report.resolution:
        table.add_row("graph edges (this view)", f"{report.resolution.get('edges', 0):,}")
        table.add_row("url routes", f"{report.resolution.get('routes', 0):,}")
    if report.excluded:
        table.add_row("excluded", ", ".join(f"{reason}={count}" for reason, count in report.excluded.most_common()))
    table.add_row("store", str(builder.store.path))
    console.print(table)

    if no_llm or not settings.llm_enabled:
        if not no_llm:
            console.print("[yellow]OPENAI_API_KEY not set: skipped summaries and embeddings (graph + BM25 still work)[/yellow]")
    else:
        llm = LLMClient(settings, builder.store)
        enricher = Enricher(builder.store, llm, settings, progress)
        root_view, _ = builder.build_root(base_sha)
        views = [root_view] if view.view_key == root_view.view_key else [root_view, view]
        if batch:
            for target_view in views:
                batch_ids = enricher.submit_batches(target_view, summaries, embeddings)
                console.print(f"submitted batches {batch_ids}; run `review batch-collect` later")
        else:
            for target_view in views:
                if summaries:
                    console.print(f"summaries written: {enricher.summarize(target_view)}")
                if embeddings:
                    console.print(f"vectors written: {enricher.embed(target_view)}")
        print_usage(llm)
    console.print(f"[green]done in {time.monotonic() - started:.1f}s[/green]")


@app.command("batch-collect")
def batch_collect(repo: str = RepoOption):
    git_repo = GitRepo(repo)
    settings = settings_for_repo(git_repo)
    store = open_store(git_repo, settings)
    llm = LLMClient(settings, store)
    counts = Enricher(store, llm, settings, progress).collect_batches()
    console.print(counts)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language or code-shaped query"),
    repo: str = RepoOption,
    base: str = BaseOption,
    head: str = HeadOption,
    mode: str = typer.Option("hybrid", "--mode", help="hybrid, lexical, dense, or graph"),
    kind: list[str] = typer.Option(None, "--kind", help="Restrict to chunk kinds"),
    tag: list[str] = typer.Option(None, "--tag", help="Require any of these tags"),
    path: str = typer.Option(None, "--path", help="Path prefix filter"),
    version: str = typer.Option("current", "--version", help="current, base, or head"),
    exclude_changed: bool = typer.Option(False, "--exclude-changed"),
    k: int = typer.Option(10, "--k"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    no_llm: bool = typer.Option(False, "--no-llm"),
    as_json: bool = typer.Option(False, "--json"),
):
    session = open_session(repo, base, head, not no_llm)
    filters = SearchFilters(kinds=kind or None, tags_any=tag or None, path_prefix=path, version=version, exclude_changed=exclude_changed)
    ranked, lists = session.engine.search(query, filters, mode, k, rerank_results=rerank)
    rows = {row["id"]: row for row in session.view.chunks_for_row_ids([item.row_id for item in ranked])}
    if as_json:
        output = []
        for item in ranked:
            row = rows[item.row_id]
            output.append({"ref": f"c{row['id']}", "symbol": row["symbol"], "kind": row["kind"], "at": f"{row['path']}:{row['start_line']}-{row['end_line']}", "score": round(item.score, 4), "sources": item.sources, "relevance": item.relevance})
        console.print_json(json.dumps(output))
        return
    table = Table(title=f"'{query}' ({mode}; retrievers: {', '.join(sorted({ranked_list.source for ranked_list in lists})) or 'none'})")
    for column in ("ref", "symbol", "kind", "location", "score", "via"):
        table.add_column(column)
    for item in ranked:
        row = rows[item.row_id]
        via = ",".join(item.sources) + (f" r{item.relevance}" if item.relevance is not None else "")
        table.add_row(f"c{row['id']}", row["symbol"], row["kind"], f"{row['path']}:{row['start_line']}-{row['end_line']}", f"{item.score:.4f}", via)
    console.print(table)
    print_usage(session.llm)


@app.command()
def context(
    repo: str = RepoOption,
    base: str = BaseOption,
    head: str = HeadOption,
    profile: str = typer.Option("bug_hunter", "--profile", "-p", help=f"One of: {', '.join(PROFILES)}"),
    budget: int = typer.Option(None, "--budget", help="Token budget for the packed context"),
    no_llm: bool = typer.Option(False, "--no-llm"),
    show_text: bool = typer.Option(False, "--text", help="Print the packed context exactly as an agent would see it"),
    out: Path = typer.Option(None, "--out", help="Write the packed context to a file"),
):
    session = open_session(repo, base, head, not no_llm)
    wave = session.engine.first_wave(session.diff, profile, budget)
    pack = wave.pack
    console.print(f"[bold]{profile}[/bold]: {len(wave.changed)} changed chunks, {len(wave.queries)} first-wave queries, {len(wave.lists)} ranked lists")
    table = Table(title=f"Packed context: {pack.tokens:,} / {pack.budget:,} tokens")
    for column in ("ref", "role", "state", "symbol", "location", "why"):
        table.add_column(column)
    for item in pack.included:
        table.add_row(item.ref, item.role, item.version, item.symbol, f"{item.path}:{item.start_line}-{item.end_line}", "; ".join(item.reasons[:2])[:70])
    console.print(table)
    if pack.omitted:
        console.print(f"[yellow]{len(pack.omitted)} retrieved but not shown (budget)[/yellow]: " + ", ".join(f"{item.ref} {item.symbol}" for item in pack.omitted[:10]))
    if pack.history:
        console.print("[bold]history hints[/bold]:\n" + "\n".join(pack.history))
    if show_text:
        console.print(pack.text, markup=False, highlight=False)
    if out:
        out.write_text(pack.text)
        console.print(f"wrote {out}")
    print_usage(session.llm)


@app.command()
def tool(
    name: str = typer.Argument(..., help="Tool name (see `review tools`)"),
    arguments: str = typer.Argument("{}", help="JSON object of arguments"),
    repo: str = RepoOption,
    base: str = BaseOption,
    head: str = HeadOption,
    no_llm: bool = typer.Option(False, "--no-llm"),
):
    session = open_session(repo, base, head, not no_llm)
    registry = build_registry()
    spec = registry.get(name)
    parsed = json.loads(arguments)
    if spec is not None:
        for field_name in spec.arguments.model_fields:
            parsed.setdefault(field_name, None)
        defaults = {"version": "current", "depth": 1, "exclude_changed": False}
        for field_name, value in defaults.items():
            if field_name in spec.arguments.model_fields and parsed.get(field_name) is None:
                parsed[field_name] = value
    executor = ToolExecutor(session, registry, session.settings.tools)
    result = executor.run(ToolCall(id="cli", name=name, arguments=parsed))
    style = "green" if result.ok else "red"
    console.print(f"[{style}]{name} ok={result.ok} {result.duration_ms}ms truncated={result.truncated}[/{style}]")
    if result.content.startswith("{"):
        console.print_json(result.content)
    else:
        console.print(result.content, markup=False, highlight=False)


@app.command()
def tools():
    registry = build_registry()
    table = Table(title="Read-only tools")
    table.add_column("name")
    table.add_column("arguments")
    table.add_column("description")
    for definition in registry.openai_definitions():
        table.add_row(definition["name"], ", ".join(definition["parameters"]["properties"]), definition["description"])
    console.print(table)


@app.command()
def stats(repo: str = RepoOption, base: str = BaseOption, head: str = HeadOption):
    session = open_session(repo, base, head, use_llm=False)
    store = session.store
    view = session.view
    table = Table(title="Chunks by kind (current view)")
    table.add_column("kind")
    table.add_column("count", justify="right")
    for row in store.query(
        f"SELECT c.kind, count(*) AS n FROM chunks c JOIN {view.active_table} a ON c.path = a.path AND c.blob_sha = a.blob_sha "
        f"WHERE a.superseded = 0 GROUP BY c.kind ORDER BY n DESC"
    ):
        table.add_row(row["kind"], f"{row['n']:,}")
    console.print(table)
    edge_table = Table(title="Graph edges (root view)")
    for column in ("kind", "confidence", "count"):
        edge_table.add_column(column)
    for row in store.query("SELECT kind, confidence, count(*) AS n FROM edges WHERE view_key = ? GROUP BY kind, confidence ORDER BY n DESC", (view.root_key,)):
        edge_table.add_row(row["kind"], row["confidence"], f"{row['n']:,}")
    console.print(edge_table)
    tag_table = Table(title="Risk tags (derived)")
    tag_table.add_column("tag")
    tag_table.add_column("chunks", justify="right")
    for tag_name in ("view", "no_auth_gate", "unscoped_query", "tenant_scoped", "task_in_atomic_without_on_commit", "signal_receiver", "cache_write", "admin"):
        count = store.query_one("SELECT count(*) AS n FROM chunk_tags WHERE view_key = ? AND tag = ?", (view.view_key, tag_name))["n"]
        tag_table.add_row(tag_name, f"{count:,}")
    console.print(tag_table)
    vectors = store.query("SELECT model, dims, table_name FROM vector_tables")
    for row in vectors:
        count = store.query_one("SELECT count(*) AS n FROM vector_rows WHERE table_name = ?", (row["table_name"],))["n"]
        console.print(f"vectors {row['model']}@{row['dims']}: {count:,}")
    memory_count = store.query_one("SELECT count(*) AS n FROM memory")["n"]
    usage = store.query_one("SELECT count(*) AS calls, COALESCE(sum(cost_usd), 0) AS cost FROM usage")
    console.print(f"memory entries: {memory_count:,}; lifetime LLM calls: {usage['calls']:,} (${usage['cost']:.4f}); store: {store.path}")


@eval_app.command("mine")
def eval_mine(
    repo: str = RepoOption,
    limit: int = typer.Option(20, "--limit", help="Number of golden cases to create"),
    scan: int = typer.Option(400, "--scan", help="How many fix-like commits to scan"),
    rev: str = typer.Option("HEAD", "--rev"),
):
    git_repo = GitRepo(repo)
    settings = settings_for_repo(git_repo, use_llm=False)
    cases = mine_cases(git_repo, repo_cache_dir(git_repo, settings), limit, rev, scan, progress)
    console.print(f"[green]created {len(cases)} cases in {repo_cache_dir(git_repo, settings) / 'golden' / 'cases'}[/green]")
    console.print("Review each case file: confirm expected_findings categories and set labels_confirmed=true.")


@eval_app.command("list")
def eval_list(repo: str = RepoOption):
    git_repo = GitRepo(repo)
    settings = settings_for_repo(git_repo, use_llm=False)
    table = Table(title="Golden cases")
    for column in ("id", "date", "subject", "primary", "relevant", "confirmed"):
        table.add_column(column)
    for case in load_cases(repo_cache_dir(git_repo, settings)):
        table.add_row(case.id, case.date, case.subject[:60], case.primary_path, str(len(case.relevant)), "yes" if case.labels_confirmed else "no")
    console.print(table)


@eval_app.command("retrieval")
def eval_retrieval(
    repo: str = RepoOption,
    case: list[str] = typer.Option(None, "--case", help="Limit to case ids"),
    profile: str = typer.Option("impact_mapper", "--profile", help="Profile name or 'all'"),
    ablation: list[str] = typer.Option(None, "--ablation", help=f"Any of: {', '.join(ABLATIONS)} (default: all available)"),
    k: str = typer.Option("10,20", "--k", help="Comma-separated cutoffs"),
    enrich: bool = typer.Option(False, "--enrich", help="Summarize and embed each case view first (uses OPENAI_API_KEY)"),
    dims: str = typer.Option("", "--dims", help="Extra embedding sizes to compare, e.g. 512,3072"),
    graph_scale: str = typer.Option("", "--graph-scale", help="Extra graph weight multipliers to sweep, e.g. 0.25,0.5"),
    no_llm: bool = typer.Option(False, "--no-llm"),
):
    git_repo = GitRepo(repo)
    settings = settings_for_repo(git_repo, use_llm=not no_llm)
    cache_dir = repo_cache_dir(git_repo, settings)
    store = open_store(git_repo, settings)
    cases = load_cases(cache_dir, case or None)
    if not cases:
        console.print("[red]no golden cases; run `review eval mine` first[/red]")
        raise typer.Exit(1)
    profiles = list(PROFILES) if profile == "all" else [profile]
    ks = [int(value) for value in k.split(",") if value.strip()]
    dims_variants = [int(value) for value in dims.split(",") if value.strip()]
    graph_scales = [float(value) for value in graph_scale.split(",") if value.strip()]
    report = run_retrieval_eval(
        git_repo, settings, store, cases, profiles, ablation or list(ABLATIONS), ks, enrich, dims_variants, progress, graph_scales
    )
    path = save_report(cache_dir, report)
    table = Table(title=f"Retrieval eval: {len({result.case_id for result in report.results})} cases")
    table.add_column("profile")
    table.add_column("ablation", no_wrap=True)
    for cutoff in ks:
        table.add_column(f"Recall@{cutoff}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("in pack", justify="right")
    table.add_column("ms/case", justify="right")
    for summary in sorted(report.summaries, key=lambda item: (item.profile, -item.mrr)):
        table.add_row(
            summary.profile,
            summary.ablation,
            *[f"{summary.recall[f'@{cutoff}']:.3f}" for cutoff in ks],
            f"{summary.mrr:.3f}",
            f"{summary.pack_recall:.3f}",
            f"{summary.mean_latency_ms:.0f}",
        )
    console.print(table)
    for note in report.skipped:
        console.print(f"[yellow]skipped[/yellow] {note}")
    console.print(f"report: {path}")


@demo_app.command("create")
def demo_create(path: Path = typer.Argument(..., help="Empty directory to create the demo repository in")):
    target = create_demo_repo(path)
    console.print(f"[green]created {target}[/green] on branch {FEATURE_BRANCH} (base: main)")
    table = Table(title="Injected issues on the feature branch")
    table.add_column("file")
    table.add_column("category")
    table.add_column("what is wrong")
    for bug in INJECTED_BUGS:
        table.add_row(bug["path"], bug["category"], bug["summary"])
    console.print(table)
    console.print(f"try: review index -r {target} --base main --no-llm && review context -r {target} --base main -p state_regression")


if __name__ == "__main__":
    app()
