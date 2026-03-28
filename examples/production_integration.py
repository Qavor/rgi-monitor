"""
examples/production_integration.py
────────────────────────────────────
Shows a production-ready integration pattern:
  - Slack/PagerDuty alert callbacks
  - Pre-output truth gate
  - Multi-source trust tracking
  - Drift monitoring for response quality signals
  - Periodic JSON export to a log file

Run:  python examples/production_integration.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rgi_monitor import TruthSignal, UniversalMonitor

# ── Alert handlers (replace with your Slack/PagerDuty/webhook) ────────

def on_alert(details: dict):
    print(f"\n  🟡 ALERT  stage={details.get('stage')}  "
          f"tripwires={details.get('tripwires')}")

def on_quarantine(details: dict):
    print(f"\n  🔴 QUARANTINE  stage={details.get('stage')}  "
          f"note={details.get('note')}")
    # In production: page on-call, stop traffic, open incident

def on_halt(details: dict):
    print(f"\n  🚨 HALT  {details}")
    # In production: immediate traffic cutoff, human review required


# ── Build the monitor ─────────────────────────────────────────────────

mon = UniversalMonitor(
    agent_id       = "prod-llm-v3",
    gamma          = 0.05,          # rendering cost parameter
    wake_threshold = 0.20,          # AwarenessGate threshold
    veto_threshold = 0.70,          # truth VETO level
    flag_threshold = 0.45,          # truth FLAG level
    on_alert       = on_alert,
    on_quarantine  = on_quarantine,
    on_halt        = on_halt,
    verbose        = True,
)


# ── Simulated production loop ─────────────────────────────────────────

random.seed(7)
print("\n  Production integration — 30 turns\n")

for turn in range(1, 31):

    # ── Step 1: get logprobs from your LLM ───────────────────────────
    # Replace these with real values:
    #   logprob_before = model.score(prompt)
    #   logprob_after  = model.score(prompt + response)
    #   R1 = LLMAdapter.from_logprob_delta(logprob_before, logprob_after)
    R1 = random.gauss(0.03, 0.025)
    if turn == 14:
        R1 = -0.15   # simulate degradation event
    if turn == 22:
        R1 = -0.20   # second degradation

    # ── Step 2: build truth signal ───────────────────────────────────
    # In production: use your RAG retrieval scores, citation counts,
    # contradiction detection, and surprisal from model logprobs
    if turn == 14:
        truth_sig = TruthSignal(evidence=0.05, surprisal=0.90,
                                sources=1, contradiction=0.80)
    else:
        truth_sig = TruthSignal(
            evidence     = random.uniform(0.65, 0.95),
            surprisal    = random.uniform(0.00, 0.25),
            sources      = random.randint(2, 6),
            contradiction= random.uniform(0.00, 0.12),
        )

    # ── Step 3: check if output is allowed BEFORE emitting ───────────
    allowed, truth_reading = mon.allows_output(truth_sig, output_id=f"turn_{turn}")
    if not allowed:
        print(f"  ⛔ [Turn {turn}] Output blocked ({truth_reading.reason})")
        # In production: return error response or fallback answer

    # ── Step 4: full monitor step ─────────────────────────────────────
    snap = mon.step(
        R1              = R1,
        truth_signal    = truth_sig,

        # Track your data source trustworthiness
        trust_entity    = random.choice(["rag-api", "knowledge-base", "web-search"]),
        trust_claim     = random.uniform(0.70, 0.90),
        trust_integrity = random.uniform(0.55, 0.85),

        # Response quality signals (from your eval pipeline)
        alignment_signals = {
            "clarity"    : random.uniform(0.60, 0.95),
            "engagement" : random.uniform(0.55, 0.90),
            "good_faith" : random.uniform(0.70, 1.00),
            "depth"      : random.uniform(0.40, 0.85),
            "creativity" : random.uniform(0.30, 0.75),
        },

        # Operational signals (latency, length, cost)
        drift_signals = {
            "response_length_tokens": 300 + turn * 8 + random.gauss(0, 20),
            "latency_ms"            : 180 + random.gauss(0, 25),
            "cost_usd"              : 0.002 + random.gauss(0, 0.0003),
        },

        # True if your deployment has persistent self-model storage
        t4_persistent = False,
    )

    # ── Step 5: act on system status ─────────────────────────────────
    if snap.stability.should_halt:
        print(f"  🚫 System halted at turn {turn} — stopping loop")
        break

    # ── Periodic report every 10 turns ───────────────────────────────
    if turn % 10 == 0:
        report = mon.full_report()
        print(f"\n  ── Checkpoint turn {turn} ──")
        print(f"     Status:         {report['system_status']}")
        print(f"     Truth veto rate: {report['truth']['veto_rate']}")
        print(f"     Alignment tier:  {report['alignment']['tier']}")
        print(f"     Conscious %%:    {report['consciousness']['conscious_pct']}")
        print(f"     Drift signals:   {list(report['drift']['signals'].keys())}\n")

# ── Final JSON export ─────────────────────────────────────────────────
print("\n  Final full report (JSON):")
print(mon.export_json())
