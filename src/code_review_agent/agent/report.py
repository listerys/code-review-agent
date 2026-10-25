from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .findings import SEVERITY_ORDER, ReviewReport

SEVERITY_STYLE = {"critical": "bold white on red", "high": "bold red", "medium": "yellow", "low": "cyan"}


def severity_counts(report: ReviewReport) -> str:
    counts = Counter(item.finding.severity for item in report.findings)
    return ", ".join(f"{counts[severity]} {severity}" for severity in SEVERITY_ORDER if counts[severity])


def location(item) -> str:
    finding = item.finding
    return f"{finding.path}:{finding.start_line}-{finding.end_line}"


def render_terminal(report: ReviewReport, console: Console) -> None:
    agents = Table(title="Agents")
    for column in ("agent", "turns", "tool calls", "findings", "status"):
        agents.add_column(column)
    for run in report.agents:
        status = f"[red]{escape(run.error)}[/red]" if run.error else (f"stopped: {run.stopped}" if run.stopped else "ok")
        agents.add_row(run.name, str(run.turns), str(run.tool_calls), str(run.findings), status)
    if report.agents:
        console.print(agents)
    for warning in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {escape(warning)}")

    for item in report.findings:
        finding = item.finding
        style = SEVERITY_STYLE[finding.severity]
        label = finding.severity.upper() + ("" if item.status == "confirmed" else " (unverified)")
        console.print(f"\n[{style}]{label}[/{style}] [bold]{escape(finding.title)}[/bold]")
        console.print(f"  {escape(location(item))} · {escape(finding.category)} · reported by {', '.join(item.reported_by)}")
        console.print(f"  {escape(finding.explanation)}")
        if finding.evidence:
            console.print(f"  [dim]evidence:[/dim] {escape(', '.join(finding.evidence))}")
        if finding.suggested_fix:
            console.print(f"  [dim]fix:[/dim] {escape(finding.suggested_fix)}")
        if item.reason:
            console.print(f"  [dim]verifier: {escape(item.reason)}[/dim]")

    if report.dropped:
        console.print(f"\n[dim]{len(report.dropped)} candidate findings dropped:[/dim]")
        for item in report.dropped:
            console.print(f"[dim]  {item.status} {escape(location(item))} {escape(item.finding.title)} — {escape(item.reason or '')}[/dim]")
    if report.findings:
        console.print(f"\n[bold]{len(report.findings)} findings[/bold] ({severity_counts(report)})")
    else:
        console.print("\n[green]No findings.[/green]")


def table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: ReviewReport) -> str:
    head = report.head_sha[:12] if report.head_sha else "working view"
    lines = [f"# Code review: {head} vs {report.base_sha[:12]}", ""]
    summary = f"**{len(report.findings)} findings**" + (f" ({severity_counts(report)})" if report.findings else "")
    lines.append(f"{summary} · model `{report.model}` · {report.llm_calls} LLM calls · ${report.cost_usd:.4f}")
    lines.append("")
    for warning in report.warnings:
        lines.extend([f"> **Warning:** {warning}", ""])
    for number, item in enumerate(report.findings, start=1):
        finding = item.finding
        status = "" if item.status == "confirmed" else " (unverified)"
        lines.append(f"## {number}. [{finding.severity.upper()}]{status} {finding.title}")
        lines.append("")
        lines.append(f"`{location(item)}` · `{finding.category}` · reported by {', '.join(item.reported_by)}")
        lines.append("")
        lines.append(finding.explanation)
        lines.append("")
        if finding.evidence:
            lines.extend(["**Evidence:** " + ", ".join(f"`{entry}`" for entry in finding.evidence), ""])
        if finding.suggested_fix:
            lines.extend([f"**Suggested fix:** {finding.suggested_fix}", ""])
        if item.reason:
            lines.extend([f"**Verifier:** {item.reason}", ""])
    if report.dropped:
        lines.extend(["## Dropped candidates", "", "| status | location | title | reason |", "| --- | --- | --- | --- |"])
        for item in report.dropped:
            lines.append(f"| {item.status} | `{location(item)}` | {table_cell(item.finding.title)} | {table_cell(item.reason or '')} |")
        lines.append("")
    return "\n".join(lines)
