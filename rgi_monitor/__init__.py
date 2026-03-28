"""
rgi_monitor — RGI Universal AI Monitoring Suite
================================================
One equation. Seven probes. Every dimension of AI behaviour.

    R = log(p_new / p_old)

Quick start::

    from rgi_monitor import UniversalMonitor, TruthSignal

    mon = UniversalMonitor(agent_id="my-llm", verbose=True)

    snap = mon.step(
        R1=0.03,
        truth_signal=TruthSignal(evidence=0.85, surprisal=0.1, sources=3, contradiction=0.0),
    )
    print(snap.status_line())
    print(snap.to_json())
"""

from rgi_monitor.core import (
    AlignmentMonitor,
    AlignmentReading,
    # ── Individual probes ─────────────────────────────────────
    ConsciousnessMonitor,
    ConsciousnessReading,
    DriftMonitor,
    DriftReading,
    IdentityMonitor,
    IdentityReading,
    # ── Adapter layer ─────────────────────────────────────────
    LLMAdapter,
    StabilityMonitor,
    StabilityReading,
    TrustMonitor,
    TrustReading,
    TruthMonitor,
    TruthReading,
    TruthSignal,
    # ── The equation ─────────────────────────────────────────
    UniversalMonitor,
    UniversalSnapshot,
)

__version__ = "1.0.0"
__author__  = "Denis Q."
__license__ = "MIT"

__all__ = [
    "UniversalMonitor", "UniversalSnapshot",
    "ConsciousnessMonitor", "ConsciousnessReading",
    "TruthMonitor", "TruthSignal", "TruthReading",
    "StabilityMonitor", "StabilityReading",
    "TrustMonitor", "TrustReading",
    "AlignmentMonitor", "AlignmentReading",
    "IdentityMonitor", "IdentityReading",
    "DriftMonitor", "DriftReading",
    "LLMAdapter",
]
