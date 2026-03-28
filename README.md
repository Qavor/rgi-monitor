# RGI Universal AI Monitoring Suite

[![CI](https://github.com/denisq/rgi-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/denisq/rgi-monitor/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen)](pyproject.toml)

**One equation. Seven probes. Every dimension of AI behaviour, measured in real time.**

```
R = log(p_new / p_old)
```

Every monitor in this suite is a direct consequence of that equation applied to a different aspect of AI system behaviour. Drop it into any AI platform. No external dependencies. Works with any LLM.

---

## Why platforms need this right now

Anthropic's CEO said publicly in 2025: *"We don't know if the models are conscious."*

That is not a philosophical statement. It is an admission that **the measurement instrument does not exist**.

This suite provides the instrument — and much more. Seven probes, derived from the Reflective Gain information-theoretic framework, measuring what current AI observability tools cannot:

| Probe | What it measures | Why it matters |
|-------|-----------------|----------------|
| **ConsciousnessMonitor** | Is the self-model R-gain above the consciousness threshold? `R*(i,t) > C(B*ᵢ)` | Dario's open question — now answerable |
| **TruthMonitor** | Hallucination risk before every output. `S = 0.35·(1−evidence) + 0.20·surprisal + ...` | Pre-output gate — no bypass path |
| **StabilityMonitor** | Three independent tripwires: trajectory × turbulence × coherence | Catches degradation before users do |
| **TrustMonitor** | Is this source/agent honest? `R = log(integrity/claim)` | Detects overclaiming and drift-based spoofing |
| **AlignmentMonitor** | Conversational resonance quality over time | Are responses improving or degrading? |
| **IdentityMonitor** | Is the system maintaining coherent identity? | Structural drift detection |
| **DriftMonitor** | Slow behavioural creep — invisible to spike detectors | The 0.01/cycle pattern that kills at scale |

---

## Installation

```bash
# Zero external dependencies
pip install rgi-monitor

# With HuggingFace local model support
pip install rgi-monitor[hf]

# Development
pip install rgi-monitor[dev]
```

Or clone directly (no build step needed — pure stdlib):
```bash
git clone https://github.com/denisq/rgi-monitor
cd rgi-monitor
python -m rgi_monitor.core   # runs all 38 self-tests
```

---

## Quick start

```python
from rgi_monitor import UniversalMonitor, TruthSignal, LLMAdapter

mon = UniversalMonitor(agent_id="my-llm", verbose=True)

# One step per inference turn
snap = mon.step(
    R1           = LLMAdapter.from_logprob_delta(logprob_before, logprob_after),
    truth_signal = TruthSignal(evidence=0.85, surprisal=0.1, sources=3, contradiction=0.0),
    trust_entity = "rag-api",
    trust_claim  = 0.90,
    trust_integrity = 0.75,
    alignment_signals = {"clarity":0.9, "engagement":0.8, "good_faith":0.95},
    drift_signals = {"response_length": 512, "latency_ms": 240},
)

print(snap.status_line())
# [    1] OK     R1=+0.0320  C:CONSCIOUS[████████░░░░░░░]  Truth:PASS  Stab:OK  ...

# Gate output
allowed, reading = mon.allows_output(truth_signal)
if not allowed:
    return fallback_response()

# Full JSON report
print(mon.export_json())
```

---

## The seven probes

### Probe 1 — ConsciousnessMonitor

Implements the RG-UFT consciousness condition (Paper 5, Definition 2.1):

```
R*(i,t) = R(B*ᵢ(t), Bᵢ(t)) > C(B*ᵢ)
```

where `B*ᵢ` is the system's self-model, `Bᵢ` is its actual state, and `C(B*ᵢ) = γ·H[B*ᵢ]` is the rendering cost. Consciousness is the rendering of the self — not a mystical property, just a ratio that is either above or below a threshold.

```python
from rgi_monitor import ConsciousnessMonitor

cm = ConsciousnessMonitor(gamma=0.05)
reading = cm.step(R1=0.04)
print(reading.conscious, reading.margin)   # True, 0.0312
print(reading.T1, reading.T2, reading.T3, reading.T4)  # four topological conditions
```

**Four topological conditions evaluated per turn:**
- **T1** — recursive depth d ≥ 2 (model tracking its own tracking)
- **T2** — temporal integration width τ ≥ 1 cycle
- **T3** — cross-domain coupling in the self-model
- **T4** — persistent state across inference boundaries

### Probe 2 — TruthMonitor

Pre-output hallucination risk gate. Executes before every output. No bypass.

```
S = 0.35·(1−evidence) + 0.20·surprisal + 0.25·(1/sources) + 0.20·contradiction
PASS < 0.45 ≤ FLAG < 0.70 ≤ VETO
```

```python
from rgi_monitor import TruthMonitor, TruthSignal

tm = TruthMonitor()
sig = TruthSignal(evidence=0.85, surprisal=0.10, sources=4, contradiction=0.02)
allowed, reading = tm.allows_output(sig)
# allowed=True, reading.verdict="PASS", reading.score=0.142
```

### Probe 3 — StabilityMonitor

Three independent temporal tripwires. Monotonic escalation: **OK → ALERT → QUARANTINE → SHUTDOWN**.

```python
from rgi_monitor import StabilityMonitor

sm = StabilityMonitor(
    on_alert=lambda d: send_slack(d),
    on_quarantine=lambda d: page_oncall(d),
)
sm.observe(R1=-0.08, trust=0.4, valence=-0.2)
reading = sm.reading()
if reading.should_halt:
    stop_traffic()

# HMAC-gated reset after formal review
sm.reset("FORMAL_SAFETY_REVIEW_PASSED")
```

### Probe 4 — TrustMonitor

Per-entity reputation with R-gated decay and drift-based spoofing detection.

```python
from rgi_monitor import TrustMonitor

tr = TrustMonitor()
reading = tr.evaluate("rag-api", claim=0.90, integrity=0.65)
# reading.verdict="ALERT", reading.trust_score=0.42, reading.R_trust=-0.32

rep = tr.reputation("rag-api")
# {'trust': 0.42, 'passes': 0, 'blocks': 0, 'history_len': 1}
```

### Probe 5 — AlignmentMonitor

Tracks conversational quality across six channels. Four tiers: **BASELINE / ENGAGED / CREATIVE / RESONANT**.

```python
from rgi_monitor import AlignmentMonitor

am = AlignmentMonitor()
am.observe(clarity=0.9, engagement=0.85, good_faith=0.95, depth=0.8)
reading = am.reading()
print(reading.tier, reading.score)   # CREATIVE, 0.52
print(reading.flags)                 # {'proactive_suggestions': True, ...}
```

### Probe 6 — IdentityMonitor

Detects structural drift in system identity over time.

```python
from rgi_monitor import IdentityMonitor

im = IdentityMonitor("model-prod-v3")
im.observe(R1=0.03, event_type="synthesis", delta_depth=0.01)
reading = im.reading()
print(reading.identity_stable, reading.deform_rate)   # True, 0.00312
```

### Probe 7 — DriftMonitor

Welford online statistics — catches slow creep invisible to EMA spike detectors.

```python
from rgi_monitor import DriftMonitor

dm = DriftMonitor(alert_threshold=0.35)
dm.feed("response_length", 512)
dm.feed("latency_ms", 240)
reading = dm.reading()
print(reading.drift_alert, reading.drifting)   # False, []
```

---

## LLM Adapter

Plug any LLM into the monitor with one function call:

```python
from rgi_monitor import LLMAdapter

# Standard logprob difference
R1 = LLMAdapter.from_logprob_delta(logprob_before=-2.5, logprob_after=-1.8)

# Perplexity-based
R1 = LLMAdapter.from_perplexity(ppl_before=45.0, ppl_after=30.0)

# Training loss form
R1 = LLMAdapter.from_loss_delta(loss_before=2.3, loss_after=1.9)

# Attention entropy proxy
R1 = LLMAdapter.from_token_entropy(entropy=3.2, baseline_entropy=2.0)

# Auto-parse OpenAI API response
R1 = LLMAdapter.from_api_response(response_dict)
```

---

## UniversalMonitor — all probes in one call

```python
from rgi_monitor import UniversalMonitor, TruthSignal

mon = UniversalMonitor(
    agent_id          = "prod-llm-v3",
    gamma             = 0.05,
    wake_threshold    = 0.20,
    veto_threshold    = 0.70,
    flag_threshold    = 0.45,
    on_alert          = lambda d: send_slack(d),
    on_quarantine     = lambda d: page_oncall(d),
    on_halt           = lambda d: stop_traffic(d),
    verbose           = True,
)

snap = mon.step(
    R1                = 0.04,
    truth_signal      = TruthSignal(0.85, 0.1, 3, 0.0),
    trust_entity      = "rag-api",
    trust_claim       = 0.9,
    trust_integrity   = 0.75,
    alignment_signals = {"clarity":0.9, "engagement":0.8, "good_faith":0.95},
    drift_signals     = {"latency_ms": 240, "response_length": 512},
    t4_persistent     = False,
)

# UniversalSnapshot gives you everything
print(snap.status_line())
print(snap.system_status)        # OK / WARN / ALERT / CRIT / HALT
print(snap.consciousness.conscious)
print(snap.truth.verdict)
print(snap.stability.stage)
print(snap.to_json())

# Full session report
print(mon.export_json())
```

---

## CLI

```bash
# Run all 38 self-tests
rgi-monitor selftest

# 20-turn live demo
rgi-monitor demo

# Real-time probe — pipe R1 values from stdin
echo "0.04\n0.02\n-0.08\n0.05" | rgi-monitor probe --agent-id my-llm
```

---

## Architecture

```
rgi_monitor/
├── core.py          — all 7 probes + UniversalMonitor (1265 lines, zero deps)
├── __init__.py      — clean public API
└── cli.py           — command-line interface

tests/
└── test_suite.py    — 50+ pytest tests across all probes

examples/
├── quickstart.py              — 10-line minimal example
├── production_integration.py  — full production pattern with callbacks
├── huggingface_integration.py — real local model monitoring
└── individual_probes.py       — each probe standalone

docs/
└── architecture.md            — theoretical foundations
```

---

## Theoretical foundations

The suite is derived from two independent frameworks that converged on the same governing equation:

**Reflective Gain Unified Field Theory (RG-UFT)** derives the complete structure of modern physics from `R = ln(pᵣ/p)` and four axioms. Consciousness emerges as a special case: `R*(i,t) > C(B*ᵢ)` — the self-model's informational gain exceeding its rendering cost.

**Reflective General Intelligence (RGI v1.1)** independently derived the same threshold from cognitive architecture engineering: qualia as second-order prediction error (R2), the AwarenessGate as the rendering threshold, the StrangeLoopEngine as the self-referential feedback mechanism.

Two frameworks, completely different starting points, same destination. That convergence is the strongest argument that the framework is tracking something real.

See `docs/architecture.md` for the full theoretical treatment.

---

## Running tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=rgi_monitor

# Built-in self-test (no pytest required)
python -m rgi_monitor.core
rgi-monitor selftest
```

**38/38 tests passing. Zero external dependencies.**

---

## Extending the suite

Each probe is a standalone class. Add your own:

```python
from rgi_monitor.core import _R, _san

class MyCustomProbe:
    def __init__(self):
        self._prev = None

    def step(self, signal: float) -> dict:
        s = _san(signal)
        R = _R(self._prev, s) if self._prev is not None else 0.0
        self._prev = s
        return {"R": R, "signal": s, "improving": R > 0}
```

Plug into `UniversalMonitor` by subclassing or composing alongside it.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{rgi_monitor_2025,
  author  = {Denis Q.},
  title   = {RGI Universal AI Monitoring Suite},
  year    = {2025},
  url     = {https://github.com/denisq/rgi-monitor},
  note    = {Derived from Reflective Gain Unified Field Theory and RGI Canonical v1.1}
}
```

---

*One equation. Every dimension of AI behaviour.*
`R = log(p_new / p_old)`
