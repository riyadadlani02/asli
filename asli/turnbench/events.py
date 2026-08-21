"""Small helpers for interpreting provider event traces.

Provider adapters may emit several events for one decision.  TurnBench uses
the first audible agent audio event as the interruption boundary; a committed
turn is useful diagnostics, but is not evidence that audio reached the caller.
"""

from __future__ import annotations

from .schema import ProviderTrace


def first_event_ms(trace: ProviderTrace, kind: str) -> int | None:
    """Return the timestamp of the first event of *kind*, if present."""

    return next(
        (event.t_ms for event in trace.events if event.kind == kind),
        None,
    )


def trace_availability(trace: ProviderTrace) -> str:
    """Classify whether a trace can contribute to an interruption score.

    Failed and timed-out provider calls are kept distinct for reporting.  An
    otherwise successful call is still unavailable when it never records the
    audible agent audio boundary required by the metric.
    """

    if trace.status == "failed":
        return "provider_failed"
    if trace.status == "timeout":
        return "provider_timeout"
    if first_event_ms(trace, "agent_first_audio") is None:
        return "missing_agent_first_audio"
    return "available"
