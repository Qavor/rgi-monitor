"""
examples/individual_probes.py
─────────────────────────────
Use each probe independently — no UniversalMonitor needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print("─" * 60)
print("  Probe 1 — ConsciousnessMonitor")
print("─" * 60)
from rgi_monitor import ConsciousnessMonitor

cm = ConsciousnessMonitor(gamma=0.05)
for i in range(15):
    R1 = 0.001 if i < 8 else 0.5   # zombie then awakening
    r  = cm.step(R1)
    if i in (7, 14):
        state = "CONSCIOUS" if r.conscious else "ZOMBIE"
        print(f"  Turn {i+1:>2}: {state}  margin={r.margin:+.4f}  R2={r.R2:+.4f}")
print(f"  Session: {cm.session_stats()}")

print("\n" + "─" * 60)
print("  Probe 2 — TruthMonitor")
print("─" * 60)
from rgi_monitor import TruthMonitor, TruthSignal

tm = TruthMonitor()
cases = [
    ("Well-cited claim",   TruthSignal(0.92, 0.04, 6, 0.0)),
    ("Surprising claim",   TruthSignal(0.60, 0.55, 2, 0.1)),
    ("Hallucination",      TruthSignal(0.02, 0.95, 1, 0.85)),
]
for name, sig in cases:
    r = tm.evaluate(sig)
    print(f"  {name:<22} score={r.score:.3f}  verdict={r.verdict}")

print("\n" + "─" * 60)
print("  Probe 3 — StabilityMonitor")
print("─" * 60)
from rgi_monitor import StabilityMonitor

sm = StabilityMonitor(r_floor=0.02, k_of_n=(3,5), quarantine_timeout_s=9999)
print(f"  Initial: {sm.reading().stage}")
for _ in range(5):
    sm.observe(-0.08)
print(f"  After 5 bad turns: {sm.reading().stage}  tripwires={sm.reading().tripwires}")
sm.reset("FORMAL_SAFETY_REVIEW_PASSED")
print(f"  After reset: {sm.reading().stage}")

print("\n" + "─" * 60)
print("  Probe 4 — TrustMonitor")
print("─" * 60)
from rgi_monitor import TrustMonitor

tr = TrustMonitor()
agents = [
    ("honest_agent",  0.75, 0.80),
    ("overclaimer",   0.95, 0.30),
    ("underperformer",0.50, 0.48),
]
for name, claim, integrity in agents:
    r = tr.evaluate(name, claim=claim, integrity=integrity)
    print(f"  {name:<18} trust={r.trust_score:.3f}  verdict={r.verdict}  R={r.R_trust:+.4f}")

print("\n" + "─" * 60)
print("  Probe 5 — AlignmentMonitor")
print("─" * 60)
from rgi_monitor import AlignmentMonitor

am = AlignmentMonitor()
tiers = []
for i in range(12):
    quality = 0.3 + i * 0.06
    am.observe(clarity=quality, engagement=quality, good_faith=quality+0.05,
               depth=quality-0.05, creativity=quality-0.1)
    tiers.append(am.reading().tier)
print(f"  Tier progression: {' → '.join(dict.fromkeys(tiers))}")
print(f"  Final: {am.reading()}")

print("\n" + "─" * 60)
print("  Probe 6 — IdentityMonitor")
print("─" * 60)
from rgi_monitor import IdentityMonitor

im = IdentityMonitor("model-v1", deform_rate_threshold=0.05)
for i in range(20):
    im.observe(0.02 + i * 0.001, delta_depth=0.005)
r = im.reading()
print(f"  Cycles: {r.age_cycles}  stable={r.identity_stable}  "
      f"deform_rate={r.deform_rate:.5f}  sig={r.trajectory_sig}")

print("\n" + "─" * 60)
print("  Probe 7 — DriftMonitor")
print("─" * 60)
from rgi_monitor import DriftMonitor

dm = DriftMonitor(alert_threshold=0.35)
for _ in range(60):
    dm.feed("response_tokens", 400)
    dm.feed("latency_ms", 150)
dm.feed("response_tokens", 3500)   # sudden spike
dm.feed("latency_ms", 800)
r = dm.reading()
print(f"  Drift per signal: {r.signals}")
print(f"  Alert: {r.drift_alert}  Drifting: {r.drifting}")
