"""Agentic controller.

Owns the pipeline from the architecture diagram — validate, route, select tools,
execute, integrate, log — and emits an auditable execution trace as it goes. The
trace is the deliverable the problem statement is evaluated on, so every stage
records the tool it picked, the parameters it used and the outputs it produced.

`run_stream` yields events for the SSE endpoint; `run` is the same pipeline
collapsed into a single response.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime
from time import perf_counter
from typing import Any

from app.agent import integrator, registry, router, validator
from app.core.config import Settings, get_settings
from app.core.errors import AssetNotFoundError, InputValidationError
from app.schemas.agent import AgentStep, QueryResult
from app.schemas.imagery import Asset, InputMode
from app.services.sessions import SessionStore, get_session_store
from app.services.storage import AssetStore, get_asset_store
from app.tools import plan
from app.tools.base import ToolContext, ToolOutput

Event = dict[str, Any]

_STEP_BLUEPRINT: tuple[tuple[str, str, str], ...] = (
    ("v", "Input validation", "Checking count, format, modality and pair compatibility"),
    ("m", "Modality detection", "Reading bands, geotransform and sensor family"),
    ("q", "Query parse", "Classifying the requested task"),
    ("t", "Tool selection", "Choosing specialists from the registry"),
    ("e", "Visual evidence", "Running specialists and collecting spatial evidence"),
    ("i", "Integration", "Fusing text, evidence and confidence"),
)


def _order_assets(assets: list[Asset], mode: InputMode) -> list[Asset]:
    """Primary first; for bi-temporal pairs, earliest acquisition becomes t0."""
    ordered = sorted(assets, key=lambda a: 0 if a.role == "primary" else 1)
    if mode != "bi-temporal" or len(ordered) != 2:
        return ordered

    def parsed(asset: Asset):
        try:
            return datetime.strptime(asset.date, "%d %b %Y")
        except ValueError:
            return None

    first, second = parsed(ordered[0]), parsed(ordered[1])
    if first and second and second < first:
        ordered.reverse()
    return ordered


class AgentController:
    def __init__(
        self,
        settings: Settings | None = None,
        asset_store: AssetStore | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.assets = asset_store or get_asset_store()
        self.sessions = session_store or get_session_store()

    def run(
        self,
        query: str,
        asset_ids: list[str],
        session_id: str | None = None,
        mode_hint: InputMode | None = None,
    ) -> QueryResult:
        result: QueryResult | None = None
        for event in self.run_stream(query, asset_ids, session_id, mode_hint):
            if event["type"] == "result":
                result = event["result"]
        assert result is not None  # run_stream either yields a result or raises
        return result

    def run_stream(
        self,
        query: str,
        asset_ids: list[str],
        session_id: str | None = None,
        mode_hint: InputMode | None = None,
    ) -> Iterator[Event]:
        started = perf_counter()
        query_id = uuid.uuid4().hex[:12]
        steps = [AgentStep(id=sid, label=label, detail=detail) for sid, label, detail in _STEP_BLUEPRINT]
        by_id = {step.id: step for step in steps}

        yield {"type": "trace", "queryId": query_id, "steps": [s.model_dump(by_alias=True) for s in steps]}

        def begin(step_id: str) -> tuple[AgentStep, float]:
            step = by_id[step_id]
            step.status = "running"
            return step, perf_counter()

        def finish(step: AgentStep, clock: float, detail: str, **outputs: Any) -> Event:
            step.status = "done"
            step.detail = detail
            step.duration_ms = int((perf_counter() - clock) * 1000)
            if outputs:
                step.outputs.update(outputs)
            return {"type": "step", "step": step.model_dump(by_alias=True)}

        def fail(step: AgentStep, clock: float, detail: str) -> Event:
            step.status = "failed"
            step.detail = detail
            step.duration_ms = int((perf_counter() - clock) * 1000)
            return {"type": "step", "step": step.model_dump(by_alias=True)}

        # 1. Input validation ------------------------------------------------
        step, clock = begin("v")
        yield {"type": "step", "step": step.model_dump(by_alias=True)}

        loaded: list[Asset] = []
        for asset_id in asset_ids:
            asset = self.assets.get(asset_id)
            if asset is None:
                yield fail(step, clock, f"Unknown asset {asset_id}")
                raise AssetNotFoundError(f"Asset {asset_id} is not in the store.")
            loaded.append(asset)

        report = validator.validate(loaded, mode_hint, self.settings)
        if not report.ok:
            yield fail(step, clock, report.errors[0].message if report.errors else "Input rejected")
            raise InputValidationError(report)

        mode = report.mode
        ordered = _order_assets(loaded, mode)
        yield finish(
            step,
            clock,
            report.summary,
            issues=[i.model_dump(by_alias=True) for i in report.issues],
            co_registered=report.co_registered,
            overlap_percent=report.overlap_percent,
        )

        # 2. Modality detection ---------------------------------------------
        step, clock = begin("m")
        yield {"type": "step", "step": step.model_dump(by_alias=True)}
        scenes = []
        for asset in ordered:
            try:
                scenes.append(self.assets.scene(asset.id))
            except KeyError:
                yield fail(step, clock, f"Raster for {asset.name} is missing from disk")
                raise AssetNotFoundError(f"Raster for asset {asset.id} is missing.")

        modality_detail = " · ".join(
            f"{'primary' if i == 0 else 'secondary'} {a.modality} ({s.bands}-band, {a.modality_source})"
            for i, (a, s) in enumerate(zip(ordered, scenes, strict=True))
        )
        yield finish(
            step,
            clock,
            modality_detail,
            assets=[{"id": a.id, "modality": a.modality, "bands": s.bands, "gsd": a.gsd} for a, s in zip(ordered, scenes, strict=True)],
        )

        # 3. Query parse ----------------------------------------------------
        step, clock = begin("q")
        yield {"type": "step", "step": step.model_dump(by_alias=True)}
        decision = router.classify(query, mode)
        intent = decision.intent
        effective_task = router.degrade(intent.task, mode)
        degraded = effective_task != intent.task
        if degraded:
            intent = intent.model_copy(
                update={
                    "task": effective_task,
                    "label": router.TASK_LABELS[effective_task],
                    "specialists": [
                        registry.display_name(key, self.settings)
                        for key in router.TASK_MODEL_KEYS[effective_task]
                    ],
                }
            )
        parse_detail = f"{intent.label} · target '{decision.target}'"
        if degraded:
            parse_detail += " · degraded to match available inputs"
        yield finish(
            step,
            clock,
            parse_detail,
            task=intent.task,
            target=decision.target,
            matched_terms=decision.matched,
            degraded=degraded,
        )

        # 4. Tool selection -------------------------------------------------
        step, clock = begin("t")
        yield {"type": "step", "step": step.model_dump(by_alias=True)}
        tools = plan(intent.task, mode, decision.target)
        chain = " → ".join(tool.name for tool in tools)
        yield finish(
            step,
            clock,
            f"{chain} → Integrator",
            tools=[{"name": t.name, "model": t.model_key, "backend": t.backend_label(self.settings)} for t in tools],
        )

        # 5. Specialist execution -------------------------------------------
        step, clock = begin("e")
        yield {"type": "step", "step": step.model_dump(by_alias=True)}
        ctx = ToolContext(
            query=query,
            task=intent.task,
            target=decision.target,
            mode=mode,
            assets=ordered,
            scenes=scenes,
            validation=report,
            settings=self.settings,
        )

        outputs: list[ToolOutput] = []
        audit: list[dict[str, Any]] = []
        for tool in tools:
            tool_clock = perf_counter()
            output = tool.run(ctx)
            outputs.append(output)
            audit.append(
                {
                    "tool": tool.name,
                    "model": tool.model_key,
                    "backend": output.backend,
                    "params": output.params,
                    "outputs": output.outputs,
                    "confidence": output.confidence,
                    "durationMs": int((perf_counter() - tool_clock) * 1000),
                }
            )

        box_count = sum(len(o.boxes) for o in outputs)
        mask_count = sum(len(o.masks) for o in outputs)
        yield finish(
            step,
            clock,
            f"{box_count} box(es) · {mask_count} mask(s) from {len(tools)} specialist(s)",
            tools=audit,
        )

        # 6. Integration ----------------------------------------------------
        step, clock = begin("i")
        yield {"type": "step", "step": step.model_dump(by_alias=True)}

        session = self.sessions.get_or_create(session_id, mode)
        self.sessions.attach_assets(session.id, [a.id for a in ordered], mode)

        elapsed_ms = int((perf_counter() - started) * 1000)
        result = integrator.integrate(
            ctx,
            intent,
            outputs,
            report,
            steps,
            elapsed_ms,
            query_id=query_id,
            session_id=session.id,
        )
        yield finish(
            step,
            clock,
            f"confidence {result.confidence:.2f} · {len(result.metrics)} metric(s) · {result.inference_backend}",
            confidence=result.confidence,
        )

        result.trace = steps  # pick up the final status of the integration step
        self.sessions.add_result(session.id, result)
        yield {"type": "result", "result": result}


_controller: AgentController | None = None


def get_controller() -> AgentController:
    global _controller
    if _controller is None:
        _controller = AgentController()
    return _controller
