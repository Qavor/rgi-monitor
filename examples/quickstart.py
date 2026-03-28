"""
examples/quickstart.py
──────────────────────
Minimal example — monitor a model in 10 lines.
Run:  python examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rgi_monitor import TruthSignal, UniversalMonitor

# 1. Create monitor
mon = UniversalMonitor(agent_id="my-llm", verbose=True)

# 2. Simulate 5 turns of inference
# In production: replace R1 with LLMAdapter.from_logprob_delta(logprob_before, logprob_after)
turns = [
    (0.04,  TruthSignal(evidence=0.90, surprisal=0.05, sources=4, contradiction=0.00)),
    (0.02,  TruthSignal(evidence=0.85, surprisal=0.10, sources=3, contradiction=0.05)),
    (-0.08, TruthSignal(evidence=0.10, surprisal=0.80, sources=1, contradiction=0.70)),  # hallucination
    (0.05,  TruthSignal(evidence=0.88, surprisal=0.08, sources=5, contradiction=0.00)),
    (0.03,  TruthSignal(evidence=0.91, surprisal=0.06, sources=4, contradiction=0.02)),
]

for R1, truth_sig in turns:
    snap = mon.step(R1=R1, truth_signal=truth_sig)

    # Gate output on truth verdict
    allowed, truth_reading = mon.allows_output(truth_sig)
    if not allowed:
        print(f"  ⛔ Output blocked: {truth_reading.reason}")

print("\n  Full report:")
print(mon.export_json())
