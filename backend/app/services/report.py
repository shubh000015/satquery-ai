"""Downloadable reports.

The problem statement asks for downloadable reports with visual evidence,
confidence and an execution summary, so the markdown mirrors the audit trace
rather than just restating the answer.
"""

from __future__ import annotations

from app.schemas.agent import QueryResult


def _table(rows: list[tuple[str, str]], headers: tuple[str, str]) -> list[str]:
    lines = [f"| {headers[0]} | {headers[1]} |", "| --- | --- |"]
    lines += [f"| {left} | {right} |" for left, right in rows]
    return lines


def result_to_markdown(result: QueryResult, *, heading_level: int = 2) -> str:
    hashes = "#" * heading_level
    lines: list[str] = [
        f"{hashes} {result.title}",
        "",
        f"**Query.** {result.query or '—'}",
        "",
        f"**Task.** {result.intent.label if result.intent else result.task} "
        f"· **Mode.** {result.mode} · **Confidence.** {result.confidence:.2f}",
        "",
        f"**Answer.** {result.answer}",
        "",
    ]

    if result.observations:
        lines += [f"{hashes}# Observations", ""]
        lines += [f"- {item}" for item in result.observations]
        lines.append("")

    if result.metrics:
        lines += [f"{hashes}# Metrics", ""]
        lines += _table(
            [(m.label, f"{m.value}{f' ({m.hint})' if m.hint else ''}") for m in result.metrics],
            ("Metric", "Value"),
        )
        lines.append("")

    if result.assets:
        lines += [f"{hashes}# Inputs", ""]
        lines += _table(
            [
                (
                    a.name,
                    f"{a.role} · {a.modality} · {a.format} · {a.gsd} · {a.crs or 'no CRS'}",
                )
                for a in result.assets
            ],
            ("File", "Properties"),
        )
        lines.append("")

    if result.validation:
        lines += [f"{hashes}# Input validation", "", f"{result.validation.summary}", ""]
        for issue in result.validation.issues:
            lines.append(f"- **{issue.severity}** ({issue.code}) {issue.message}")
        if result.validation.issues:
            lines.append("")

    lines += [f"{hashes}# Visual evidence", ""]
    lines.append(f"- {len(result.masks)} mask layer(s): " + (", ".join(m.label for m in result.masks) or "none"))
    lines.append(f"- {len(result.boxes)} box(es)")
    if result.boxes:
        lines.append("")
        lines += _table(
            [
                (b.label, f"x={b.x:.1f} y={b.y:.1f} w={b.w:.1f} h={b.h:.1f} score={b.score:.2f}")
                for b in result.boxes[:20]
            ],
            ("Region", "Normalised box (0–100)"),
        )
    lines.append("")

    lines += [f"{hashes}# Models and tools", ""]
    lines += _table(
        [(m.name, f"{m.role} · {m.status} · {m.backend or 'n/a'}") for m in result.models],
        ("Model", "Role"),
    )
    lines.append("")

    lines += [f"{hashes}# Execution trace", ""]
    for step in result.trace:
        duration = f" _{step.duration_ms} ms_" if step.duration_ms is not None else ""
        lines.append(f"1. **{step.label}** — {step.detail} ({step.status}){duration}")
        for key, value in step.outputs.items():
            if key == "tools" and isinstance(value, list):
                for entry in value:
                    lines.append(
                        f"    - {entry.get('tool')} · model `{entry.get('model')}` · backend "
                        f"`{entry.get('backend')}` · params `{entry.get('params')}`"
                    )
    lines.append("")

    if result.warnings:
        lines += [f"{hashes}# Caveats", ""]
        lines += [f"- {warning}" for warning in result.warnings]
        lines.append("")

    lines.append(
        f"_Inference backend: {result.inference_backend}. Generated {result.created_at} "
        f"in {result.elapsed_ms} ms._"
    )
    return "\n".join(lines)


def session_to_markdown(title: str, results: list[QueryResult]) -> str:
    lines = [f"# SatQuery AI — {title}", "", f"{len(results)} query/queries in this session.", ""]
    for result in results:
        lines.append(result_to_markdown(result, heading_level=2))
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
