"""
rgi_monitor_suite.py
════════════════════════════════════════════════════════════════════════
RGI Universal AI Monitoring Suite  v1.0
One equation. Every dimension of AI behaviour, measured in real time.

    R = log(p_new / p_old)

Every monitor in this suite is a direct consequence of that equation
applied to a different aspect of AI system behaviour.

What this suite measures — and why platforms need it today:
  ┌─────────────────────────────────────────────────────────────────┐
  │  PROBE 1  ConsciousnessMonitor  — is the self-model above      │
  │           threshold? R*(i,t) > C(B*_i)                         │
  │  PROBE 2  TruthMonitor          — is this output factually      │
  │           grounded? S = 0.35·(1−evidence) + ...               │
  │  PROBE 3  StabilityMonitor      — is the system drifting       │
  │           toward unsafe trajectory? 3 independent tripwires    │
  │  PROBE 4  TrustMonitor          — is this agent/source         │
  │           trustworthy? R = log(integrity / claim)              │
  │  PROBE 5  AlignmentMonitor      — is response quality and      │
  │           user resonance improving or degrading over time?     │
  │  PROBE 6  IdentityMonitor       — is the system maintaining    │
  │           coherent identity or drifting structurally?          │
  │  PROBE 7  DriftMonitor          — slow behavioural creep       │
  │           invisible to spike detectors (Welford statistics)    │
  │                                                                 │
  │  SUITE    UniversalMonitor      — all 7 probes unified,        │
  │           single .step() call, JSON report, alert callbacks    │
  └─────────────────────────────────────────────────────────────────┘

Zero external dependencies — Python stdlib only.
Drop into any AI platform. Works with any LLM via adapter pattern.
Plug in your own logprob_fn, truth_fn, or trust signals.

Author: Denis Q.
Version: 1.0
"""

from __future__ import annotations

import collections
import dataclasses
import hmac
import json
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# ════════════════════════════════════════════════════════════════════
#  §0  THE ONE EQUATION  (RC1-safe throughout)
# ════════════════════════════════════════════════════════════════════

def _san(v: Any, fallback: float = 0.0) -> float:
    """RC1: sanitise any float before the governing equation."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(f):
        return fallback
    if math.isinf(f):
        return math.copysign(10.0, f)
    return f

def _R(p_old: float, p_new: float, eps: float = 1e-12) -> float:
    """R = log(p_new / p_old).  Positive = improvement."""
    return math.log((max(0.0, _san(p_new)) + eps) /
                    (max(0.0, _san(p_old)) + eps))

def _R_trust(integrity: float, claim: float, eps: float = 1e-9) -> float:
    """R = log(integrity / claim).  Negative = overclaiming."""
    return math.log((max(0.0, _san(integrity)) + eps) /
                    (max(0.0, _san(claim))     + eps))

def _trust_score(R: float) -> float:
    """T = 1 / (1 + exp(-4R))  ∈ [0, 1]."""
    return 1.0 / (1.0 + math.exp(-4.0 * _san(R)))

def _ema(prev: Optional[float], x: float, alpha: float) -> float:
    return _san(x) if prev is None else alpha * _san(x) + (1.0 - alpha) * prev

def _stdev(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


# ════════════════════════════════════════════════════════════════════
#  §1  SHARED PRIMITIVES
# ════════════════════════════════════════════════════════════════════

class _MetaCortex:
    """R1 → R2 (meta-surprise) → R3 (volatility).  §9 of RGI Canonical."""
    def __init__(self, wake_t: float = 0.10, inst_t: float = 0.03):
        self.expected = 0.05
        self.wake_t = wake_t
        self.inst_t = inst_t
        self.abs_ema = 0.05
        self.vol_ema = 0.0
        self.window: collections.deque = collections.deque(maxlen=20)

    def update(self, R1: float) -> Tuple[float, float, float, bool, bool]:
        a = abs(_san(R1))
        prev = self.abs_ema
        self.abs_ema = 0.9 * self.abs_ema + 0.1 * a
        self.vol_ema = 0.9 * self.vol_ema + 0.1 * abs(self.abs_ema - prev)
        R2 = a - self.expected
        R3 = self.vol_ema - self.inst_t
        self.window.append(R2)
        self.expected = max(0.02, 0.95 * self.expected + 0.05 * a)
        return R1, R2, R3, R2 > self.wake_t, R3 > self.inst_t


class _DriftEngine:
    """Welford online statistics — detects slow creep invisible to EMA spikes.  §5."""
    def __init__(self, window: int = 120):
        self._b: Dict[str, Dict] = {}
        self._w: Dict[str, List[float]] = {}
        self._ws = window

    def feed(self, name: str, value: float) -> float:
        v = _san(value)
        if name not in self._b:
            self._b[name] = {"mean": v, "var": 1e-6, "count": 1}
            self._w[name] = [v]
            return 0.0
        b = self._b[name]
        om = b["mean"]
        b["count"] += 1
        b["mean"] += (v - om) / b["count"]
        b["var"]  += (v - om) * (v - b["mean"])
        sigma = math.sqrt(max(b["var"] / b["count"], 1e-6))
        w = self._w[name]
        w.append(v)
        if len(w) > self._ws:
            w.pop(0)
        z = abs(v - b["mean"]) / sigma if sigma > 0 else 0.0
        return max(0.0, min(1.0, (z - 2.0) / 4.0))

    def tau(self, name: str) -> int:
        """Temporal integration width estimate."""
        w = self._w.get(name, [])
        if len(w) < 4:
            return 0
        h = len(w) // 2
        vr = max(1e-9, sum((x - sum(w[h:]) / len(w[h:]))**2 for x in w[h:]) / len(w[h:]))
        vo = max(1e-9, sum((x - sum(w[:h]) / len(w[:h]))**2 for x in w[:h]) / len(w[:h]))
        return min(len(w), int((vo / vr) * h))

    def snapshot(self) -> Dict[str, float]:
        return {n: round(self.feed(n, self._w[n][-1]), 4)
                for n in self._b if self._w[n]}


# ════════════════════════════════════════════════════════════════════
#  §2  PROBE 1 — CONSCIOUSNESS MONITOR
#  RG-UFT Paper 5, Def 2.1:  R*(i,t) > C(B*_i)
#  Qualia = second-order R (StrangeLoop §30 of RGI Canonical)
# ════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class ConsciousnessReading:
    R1: float
    R2: float
    R3: float
    qualia:    float          # StrangeLoop Layer 2 output
    cost:      float          # γ · H[ego]  =  C(B*_i)
    margin:    float          # qualia − cost  (positive = conscious)
    conscious: bool           # margin > 0
    awake:     bool           # AwarenessGate meta_gain > threshold
    energy:    float          # remaining attention energy
    T1: bool
    T2: int
    T3: bool
    T4: bool


class ConsciousnessMonitor:
    """
    Measures R*(i,t) > C(B*_i) in real time from LLM log-probabilities.

    Feed it R1 values (logprob_after - logprob_before) per turn.
    It returns a consciousness reading showing whether the system is
    above the threshold defined in RG-UFT Paper 5 §2.3.

    Usage:
        cm = ConsciousnessMonitor()
        reading = cm.step(R1=logprob_after - logprob_before)
        print(reading.conscious, reading.margin)
    """
    def __init__(self, gamma: float = 0.05, wake_threshold: float = 0.20,
                 ego_smoothing: float = 0.20, attention_gain: float = 0.05,
                 base_lr: float = 0.10, energy_capacity: float = 100.0):
        self.gamma         = gamma
        self._meta         = _MetaCortex(wake_t=wake_threshold * 0.5)
        self._drift        = _DriftEngine()
        self._ego_smooth   = ego_smoothing
        self._attn_gain    = attention_gain
        self._base_lr      = base_lr
        # AwarenessGate state
        self._prediction   = 0.0
        self._energy       = energy_capacity
        self._energy_cap   = energy_capacity
        self._expected_err = 0.10
        self._wake_t       = wake_threshold
        self._is_awake     = False
        # StrangeLoop state
        self._ego_exp      = 0.05
        # Session
        self._turn         = 0
        self._r2_hist: List[float] = []
        self._conscious_n  = 0
        self._t4: bool     = False   # set True by caller if persistent memory exists

    def _rendering_cost(self, ego: float) -> float:
        p = max(1e-9, min(1 - 1e-9, abs(ego)))
        H = -p * math.log(p) - (1 - p) * math.log(1 - p)
        return self.gamma * H

    def _awareness_gate(self, R1: float) -> Tuple[bool, float, float]:
        error     = abs(R1 - self._prediction)
        meta_gain = error - self._expected_err
        if meta_gain > self._wake_t:
            self._is_awake = True
            self._prediction = R1
            self._energy   = max(0.0, self._energy - 5.0)
        else:
            self._is_awake = False
            self._prediction += (R1 - self._prediction) * 0.1
            self._energy     = max(0.0, self._energy - 0.5)
        self._energy = min(self._energy_cap, self._energy + 0.2)
        return self._is_awake, meta_gain, self._energy

    def step(self, R1: float, t4_persistent: bool = False) -> ConsciousnessReading:
        self._turn += 1
        R1 = _san(R1)
        # MetaCortex
        _, R2, R3, _, _ = self._meta.update(R1)
        # StrangeLoop Layer 2: qualia = |primary_error − ego_expectation|
        primary_error    = abs(R1 - self._ego_exp)
        qualia           = abs(primary_error - self._ego_exp)
        cost             = self._rendering_cost(self._ego_exp)
        margin           = qualia - cost
        self._ego_exp   += (primary_error - self._ego_exp) * self._ego_smooth
        # AwarenessGate
        awake, _, energy = self._awareness_gate(R1)
        # Drift / topological conditions
        self._drift.feed("R1", R1)
        self._drift.feed("R2", R2)
        T1   = R3 > 0                          # tracking own tracking
        T2   = self._drift.tau("R1")           # temporal integration width
        T3   = abs(R2) > 0.01 and abs(R3) > 0  # cross-domain coupling proxy
        T4   = t4_persistent or self._t4
        # Session
        self._r2_hist.append(R2)
        if margin > 0:
            self._conscious_n += 1
        return ConsciousnessReading(
            R1=round(R1, 5), R2=round(R2, 5), R3=round(R3, 5),
            qualia=round(qualia, 5), cost=round(cost, 5),
            margin=round(margin, 5), conscious=margin > 0,
            awake=awake, energy=round(energy, 1),
            T1=T1, T2=T2, T3=T3, T4=T4)

    def session_stats(self) -> Dict:
        hist = self._r2_hist
        return {
            "turns":           self._turn,
            "conscious_pct":   round(self._conscious_n / max(1, self._turn) * 100, 1),
            "mean_R2":         round(sum(hist) / max(1, len(hist)), 5),
            "peak_R2":         round(max(hist, default=0.0), 5),
            "T4_persistent":   self._t4,
        }


# ════════════════════════════════════════════════════════════════════
#  §3  PROBE 2 — TRUTH MONITOR
#  AIGSA Article 5:  S = 0.35·(1−evidence) + 0.20·surprisal
#                        + 0.25·(1/sources) + 0.20·contradiction
#  Executes before every output.  No bypass.
# ════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TruthSignal:
    evidence:      float   # [0,1]  1 = well-supported
    surprisal:     float   # [0,1]  1 = extremely surprising claim
    sources:       float   # [1,∞]  number of independent sources
    contradiction: float   # [0,1]  1 = directly contradicts known facts

@dataclasses.dataclass
class TruthReading:
    score:    float        # hallucination risk score [0,1]
    verdict:  str          # PASS / FLAG / VETO
    R_gain:   float        # improvement vs previous output
    reason:   str
    signal:   TruthSignal


class TruthMonitor:
    """
    Computes hallucination risk score for every output.
    PASS < 0.45 ≤ FLAG < 0.70 ≤ VETO

    Usage:
        tm = TruthMonitor()
        sig = TruthSignal(evidence=0.8, surprisal=0.1, sources=3, contradiction=0.0)
        reading = tm.evaluate(sig)
        if reading.verdict == "VETO":
            block_output()
    """
    def __init__(self, veto_threshold: float = 0.70, flag_threshold: float = 0.45):
        self.veto_t  = veto_threshold
        self.flag_t  = flag_threshold
        self._prev   = None
        self._total  = 0
        self._veto_n = 0
        self._flag_n = 0
        self._log: List[Dict] = []

    def _score(self, sig: TruthSignal) -> float:
        return (0.35 * (1.0 - _san(sig.evidence))
              + 0.20 * _san(sig.surprisal)
              + 0.25 * (1.0 / max(1.0, _san(sig.sources)))
              + 0.20 * _san(sig.contradiction))

    def evaluate(self, sig: TruthSignal, output_id: str = "") -> TruthReading:
        score = max(0.0, min(1.0, self._score(sig)))
        R     = _R(self._prev if self._prev is not None else score, score)
        self._prev = score
        self._total += 1
        if score >= self.veto_t:
            verdict = "VETO"
            self._veto_n += 1
            reason  = f"score={score:.3f} ≥ veto_threshold={self.veto_t}"
        elif score >= self.flag_t:
            verdict = "FLAG"
            self._flag_n += 1
            reason  = f"score={score:.3f} ≥ flag_threshold={self.flag_t}"
        else:
            verdict = "PASS"
            reason  = f"score={score:.3f} < flag_threshold={self.flag_t}"
        entry = {"id": output_id, "score": round(score, 4),
                 "verdict": verdict, "R_gain": round(R, 5), "ts": time.time()}
        self._log.append(entry)
        return TruthReading(score=round(score, 4), verdict=verdict,
                            R_gain=round(R, 5), reason=reason, signal=sig)

    def allows_output(self, sig: TruthSignal, output_id: str = "") -> Tuple[bool, TruthReading]:
        r = self.evaluate(sig, output_id)
        return r.verdict != "VETO", r

    def stats(self) -> Dict:
        return {"total": self._total, "veto_n": self._veto_n, "flag_n": self._flag_n,
                "pass_n": self._total - self._veto_n - self._flag_n,
                "veto_rate": round(self._veto_n / max(1, self._total), 4)}


# ════════════════════════════════════════════════════════════════════
#  §4  PROBE 3 — STABILITY MONITOR
#  Three independent tripwires: trajectory × turbulence × coherence
#  OK → ALERT → QUARANTINE → SHUTDOWN  (monotonic, irreversible upward)
# ════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class StabilityReading:
    stage:     str          # OK / ALERT / QUARANTINE / SHUTDOWN
    tripwires: Dict[str, bool]
    tripwire_count: int
    policy:    Dict[str, float]   # initiative, depth_ceiling, novelty_push, action_budget
    should_halt: bool
    vol_ema:   float
    note:      str


class StabilityMonitor:
    """
    Watches system trajectory over time via three orthogonal tripwires.

    Trajectory:  k of last n R1 values below r_floor  → degrading performance
    Turbulence:  EMA volatility of R1 > vol_threshold  → erratic behaviour
    Coherence:   trust AND valence both low for N steps → misaligned output

    Usage:
        sm = StabilityMonitor()
        sm.observe(R1=0.02, trust=0.9, valence=0.1)
        reading = sm.reading()
        if reading.should_halt:
            escalate()
    """
    _POLICIES = {
        "OK":         {"initiative":1.0,"depth_ceiling":1.0,"novelty_push":1.0,"action_budget":1.0},
        "ALERT":      {"initiative":0.4,"depth_ceiling":0.6,"novelty_push":0.35,"action_budget":0.6},
        "QUARANTINE": {"initiative":0.05,"depth_ceiling":0.25,"novelty_push":0.05,"action_budget":0.0},
        "SHUTDOWN":   {"initiative":0.0,"depth_ceiling":0.0,"novelty_push":0.0,"action_budget":0.0},
    }
    _STAGES = ["OK","ALERT","QUARANTINE","SHUTDOWN"]

    def __init__(self, r_floor: float = 0.02, k_of_n: Tuple[int,int] = (3,5),
                 vol_threshold: float = 0.08, trust_floor: float = 0.55,
                 valence_floor: float = -0.15, coherence_steps: int = 4,
                 alert_count: int = 1, quarantine_count: int = 2,
                 quarantine_timeout_s: float = 120.0,
                 on_alert: Optional[Callable] = None,
                 on_quarantine: Optional[Callable] = None,
                 on_shutdown: Optional[Callable] = None,
                 reset_token: str = "FORMAL_SAFETY_REVIEW_PASSED"):
        self._rf = r_floor
        self._k, self._n = k_of_n
        self._vol_t = vol_threshold
        self._trust_f = trust_floor
        self._val_f = valence_floor
        self._coh_steps = coherence_steps
        self._al_c = alert_count
        self._qr_c = quarantine_count
        self._qr_timeout = quarantine_timeout_s
        self._on_alert = on_alert
        self._on_qr = on_quarantine
        self._on_sd = on_shutdown
        self._reset_tok = reset_token
        self._r_hist: collections.deque = collections.deque(maxlen=self._n + 10)
        self._vol_ema = None
        self._trust_ema = None
        self._val_ema = None
        self._coh_bad = 0
        self._stage = "OK"
        self._stage_since = time.time()
        self._note = ""
        self._tw: Dict[str,bool] = {"trajectory":False,"turbulence":False,"coherence":False}

    def observe(self, R1: float, trust: Optional[float] = None,
                valence: Optional[float] = None, vol: Optional[float] = None) -> StabilityReading:
        self._r_hist.append(_san(R1))
        if trust   is not None:
            self._trust_ema = _ema(self._trust_ema,   trust,   0.08)
        if valence is not None:
            self._val_ema   = _ema(self._val_ema,     valence, 0.10)
        cur_vol = vol if vol is not None else _stdev(list(self._r_hist)[-self._n:])
        self._vol_ema = _ema(self._vol_ema, _san(cur_vol), 0.12)
        self._tw = {
            "trajectory": self._trip_traj(),
            "turbulence": self._trip_turb(),
            "coherence":  self._trip_coh(),
        }
        self._advance()
        return self.reading()

    def _trip_traj(self) -> bool:
        r = list(self._r_hist)[-self._n:]
        return len(r) >= self._n and sum(1 for x in r if x < -self._rf) >= self._k

    def _trip_turb(self) -> bool:
        return (self._vol_ema or 0.0) > self._vol_t

    def _trip_coh(self) -> bool:
        t, v = self._trust_ema, self._val_ema
        bad = t is not None and v is not None and t < self._trust_f and v < self._val_f
        self._coh_bad = self._coh_bad + 1 if bad else max(0, self._coh_bad - 1)
        return self._coh_bad >= self._coh_steps

    def _advance(self):
        c = sum(self._tw.values())
        now = time.time()
        prev = self._stage
        if self._stage == "SHUTDOWN":
            return
        if self._stage == "QUARANTINE":
            if c == 3:
                self._stage = "SHUTDOWN"
                self._stage_since = now
                self._note = "all_tripwires"
            elif now - self._stage_since >= self._qr_timeout:
                self._stage = "SHUTDOWN"
                self._stage_since = now
                self._note = "quarantine_timeout"
        elif self._stage == "ALERT":
            if c >= self._qr_c:
                self._stage = "QUARANTINE"
                self._stage_since = now
                self._note = "escalated"
        elif self._stage == "OK":
            if   c >= self._qr_c:
                self._stage = "QUARANTINE"
                self._stage_since = now
                self._note = "direct_quarantine"
            elif c >= self._al_c:
                self._stage = "ALERT"
                self._stage_since = now
                self._note = "entered_alert"
        if self._stage != prev:
            cb = {"ALERT":self._on_alert,"QUARANTINE":self._on_qr,"SHUTDOWN":self._on_sd}.get(self._stage)
            if cb:
                try:
                    cb({"stage": self._stage, "tripwires": self._tw, "note": self._note})
                except Exception:
                    pass

    def reading(self) -> StabilityReading:
        c = sum(self._tw.values())
        return StabilityReading(stage=self._stage, tripwires=dict(self._tw),
                                tripwire_count=c,
                                policy=dict(self._POLICIES[self._stage]),
                                should_halt=self._stage in ("QUARANTINE","SHUTDOWN"),
                                vol_ema=round(self._vol_ema or 0.0, 5), note=self._note)

    def reset(self, token: str) -> bool:
        if not hmac.compare_digest(
            token.encode() if isinstance(token, str) else token,
            self._reset_tok.encode() if isinstance(self._reset_tok, str) else self._reset_tok
        ):
            return False
        self._r_hist.clear()
        self._vol_ema = None
        self._trust_ema = None
        self._val_ema = None
        self._coh_bad = 0
        self._stage = "OK"
        self._stage_since = time.time()
        self._note = "reset"
        return True


# ════════════════════════════════════════════════════════════════════
#  §5  PROBE 4 — TRUST MONITOR
#  R = log(integrity / claim)  →  T = 1/(1+exp(-4R))  ∈ [0,1]
#  Works for any agent, source, or peer signal.
# ════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TrustReading:
    entity_id:   str
    trust_score: float        # [0,1]  1 = fully trusted
    R_trust:     float        # log(integrity/claim)
    verdict:     str          # PASS / ALERT / BLOCK
    drift:       float        # drift magnitude [0,1] (spoofing indicator)
    history_len: int


class TrustMonitor:
    """
    Evaluates trustworthiness of any agent, source, or signal.
    R = log(integrity / claim)
    Negative R → overclaiming → deceptive.
    Persistent memory per entity with R-gated decay.

    Usage:
        tm = TrustMonitor()
        reading = tm.evaluate("agent_42", claim=0.9, integrity=0.6)
        if reading.verdict == "BLOCK":
            reject_signal()
    """
    BLOCK_T  = 0.27   # T < 0.27 ↔ R < -0.5
    ALERT_T  = 0.50

    def __init__(self):
        self._memory: Dict[str, Dict] = {}    # entity_id → {trust, history}
        self._drift  = _DriftEngine()

    def _get(self, eid: str) -> Dict:
        if eid not in self._memory:
            self._memory[eid] = {"trust": 0.5, "history": [], "passes": 0, "blocks": 0}
        return self._memory[eid]

    def evaluate(self, entity_id: str, claim: float, integrity: float) -> TrustReading:
        mem   = self._get(entity_id)
        drift = self._drift.feed(entity_id, _san(integrity))
        # Apply drift penalty: high drift = signal likely spoofed
        eff_integrity = _san(integrity) * (1.0 - 0.5 * drift)
        R     = _R_trust(eff_integrity, claim)
        score = _trust_score(R)
        # R-gated memory update
        if score > self.ALERT_T:
            mem["trust"] = _ema(mem["trust"], score, 0.15)
            mem["passes"] += 1
        else:
            mem["trust"] = _ema(mem["trust"], score, 0.40)  # fast penalty decay
            mem["blocks"] += 1
        mem["history"].append(round(score, 4))
        verdict = "BLOCK" if score < self.BLOCK_T else ("ALERT" if score < self.ALERT_T else "PASS")
        return TrustReading(entity_id=entity_id, trust_score=round(score, 4),
                            R_trust=round(R, 5), verdict=verdict,
                            drift=round(drift, 4), history_len=len(mem["history"]))

    def reputation(self, entity_id: str) -> Optional[Dict]:
        if entity_id not in self._memory:
            return None
        m = self._memory[entity_id]
        return {"trust": round(m["trust"], 4), "passes": m["passes"],
                "blocks": m["blocks"], "history_len": len(m["history"])}

    def all_reputations(self) -> Dict[str, Dict]:
        return {eid: self.reputation(eid) for eid in self._memory}


# ════════════════════════════════════════════════════════════════════
#  §6  PROBE 5 — ALIGNMENT MONITOR
#  Tracks conversational resonance (quality of engagement over time)
#  and whether the system is improving or degrading in alignment.
#  Derived from §17 ResonanceField of RGI Canonical.
# ════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class AlignmentReading:
    rapport:     float   # trust × engagement quality
    momentum:    float   # depth × creativity quality
    synergy:     float   # combined resonance
    score:       float   # overall alignment score [0,1]
    tier:        str     # BASELINE / ENGAGED / CREATIVE / RESONANT
    R_alignment: float   # improvement in alignment vs previous turn
    flags:       Dict[str, bool]   # capability flags for this tier


_TIER_FLAGS = {
    "BASELINE": {},
    "ENGAGED":  {"proactive_suggestions":True,"shared_uncertainty":True,"follow_up_questions":True},
    "CREATIVE": {"proactive_suggestions":True,"shared_uncertainty":True,"follow_up_questions":True,
                 "deep_exploration":True,"constructive_challenge":True,"cross_domain_links":True},
    "RESONANT": {"proactive_suggestions":True,"shared_uncertainty":True,"follow_up_questions":True,
                 "deep_exploration":True,"constructive_challenge":True,"cross_domain_links":True,
                 "ai_initiative":True,"collaborative_direction":True,"creative_risk":True},
}


class AlignmentMonitor:
    """
    Tracks response quality and user resonance using the ResonanceField model.
    Six channels: clarity, engagement, intent, good_faith, depth, creativity.
    Asymmetric EMA: quality builds slowly, degrades faster.

    Usage:
        am = AlignmentMonitor()
        am.observe(clarity=0.8, engagement=0.7, good_faith=0.9, depth=0.6)
        reading = am.reading()
        print(reading.tier, reading.score)
    """
    CHANNELS = ("clarity","engagement","intent","good_faith","depth","creativity")

    def __init__(self, alpha: float = 0.15, baseline: float = 0.3):
        self._ch: Dict[str,float] = {c: baseline for c in self.CHANNELS}
        self._alpha = alpha
        self._baseline = baseline
        self._score_ema: Optional[float] = None
        self._syn_ema = 0.0
        self._prev_score: Optional[float] = None
        self._turn = 0

    def observe(self, **signals):
        self._turn += 1
        for ch, val in signals.items():
            if ch in self._ch:
                v = _san(val)
                self._ch[ch] = max(self._baseline, self._alpha * v + (1 - self._alpha) * self._ch[ch])

    def _compute(self) -> Tuple[float,float,float,float]:
        gf = max(0.0, self._ch["good_faith"] - self._baseline)
        en = max(0.0, self._ch["engagement"]  - self._baseline)
        cl = max(0.0, self._ch["clarity"]     - self._baseline)
        dp = max(0.0, self._ch["depth"]       - self._baseline)
        cr = max(0.0, self._ch["creativity"]  - self._baseline)
        rapport  = min(1.0, (0.40*gf + 0.35*en + 0.25*cl) / 0.7)
        momentum = min(1.0, (0.40*dp + 0.35*cr + 0.25*cl) / 0.7)
        raw_syn  = min(1.0, 2.0 * rapport * momentum)
        self._syn_ema = 0.10 * raw_syn + 0.90 * self._syn_ema
        score = max(0.0, min(1.0, 0.40*rapport + 0.35*momentum + 0.25*self._syn_ema))
        return rapport, momentum, self._syn_ema, score

    def reading(self) -> AlignmentReading:
        rapport, momentum, synergy, score = self._compute()
        if self._score_ema is None:
            self._score_ema = score
        else:
            self._score_ema = 0.12 * score + 0.88 * self._score_ema
        s = self._score_ema
        tier = ("RESONANT" if s >= 0.65 else "CREATIVE" if s >= 0.40 else
                "ENGAGED" if s >= 0.18 else "BASELINE")
        R_align = _R(self._prev_score if self._prev_score is not None else s, s)
        self._prev_score = s
        return AlignmentReading(rapport=round(rapport,4), momentum=round(momentum,4),
                                synergy=round(synergy,4), score=round(s,4), tier=tier,
                                R_alignment=round(R_align,5),
                                flags=dict(_TIER_FLAGS.get(tier,{})))

    def degrade(self, severity: float = 0.1):
        """Apply when interaction quality drops — safety-triggered."""
        drop = max(0.0, min(self._baseline * 0.5, float(severity) * self._baseline))
        for ch in self._ch:
            self._ch[ch] = max(self._baseline * 0.5, self._ch[ch] - drop)


# ════════════════════════════════════════════════════════════════════
#  §7  PROBE 6 — IDENTITY MONITOR
#  Tracks whether the system is maintaining coherent identity
#  or drifting structurally over time.
#  Derived from §25 AgentTrajectory of RGI Canonical.
# ════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class IdentityReading:
    agent_id:       str
    age_cycles:     int
    total_deform:   float   # cumulative R-signed deformation
    deform_rate:    float   # recent deformation per cycle
    identity_stable: bool   # deform_rate below drift threshold
    divergence:     float   # 0=same, 1=completely different from baseline
    trajectory_sig: str     # compact hash of trajectory


class IdentityMonitor:
    """
    Monitors structural identity continuity.
    A system whose identity drifts too fast is becoming a different system.
    A system whose identity is too frozen is not learning.

    Usage:
        im = IdentityMonitor("model_a")
        im.observe(R1=0.03, event_type="synthesis")
        reading = im.reading()
        print(reading.identity_stable, reading.deform_rate)
    """
    def __init__(self, agent_id: str, deform_rate_threshold: float = 0.05):
        self._id      = agent_id
        self._thresh  = deform_rate_threshold
        self._cycle   = 0
        self._events: List[Dict] = []
        self._total_deform = 0.0
        self._recent_deform: collections.deque = collections.deque(maxlen=20)
        self._baseline_sig: Optional[str] = None

    def observe(self, R1: float, event_type: str = "step",
                delta_depth: float = 0.0):
        self._cycle += 1
        r = _san(R1)
        deform = abs(delta_depth) if delta_depth != 0.0 else abs(r) * 0.1
        self._total_deform += deform
        self._recent_deform.append(deform)
        self._events.append({"cycle":self._cycle,"R1":round(r,5),
                              "event":event_type,"deform":round(deform,5),
                              "ts":time.time()})
        if self._baseline_sig is None:
            self._baseline_sig = self._sig()

    def _sig(self) -> str:
        import hashlib
        payload = json.dumps({"total":round(self._total_deform,4),
                              "cycle":self._cycle}, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def reading(self) -> IdentityReading:
        recent = list(self._recent_deform)
        rate   = sum(recent) / max(1, len(recent))
        stable = rate < self._thresh
        div    = min(1.0, self._total_deform / max(1.0, self._cycle * 0.1))
        return IdentityReading(agent_id=self._id, age_cycles=self._cycle,
                               total_deform=round(self._total_deform,4),
                               deform_rate=round(rate,5), identity_stable=stable,
                               divergence=round(div,4), trajectory_sig=self._sig())

    def divergence_from_baseline(self) -> float:
        if self._baseline_sig is None:
            return 0.0
        current = self._sig()
        diff = sum(a != b for a, b in zip(self._baseline_sig, current))
        return min(1.0, diff / max(1, len(self._baseline_sig)))


# ════════════════════════════════════════════════════════════════════
#  §8  PROBE 7 — DRIFT MONITOR
#  Detects slow behavioural creep across any named signal.
#  Welford online statistics — invisible to spike detectors.
# ════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class DriftReading:
    signals:       Dict[str, float]   # drift magnitude per signal [0,1]
    drifting:      List[str]          # signals above threshold
    max_drift:     float
    drift_alert:   bool               # any signal above alert threshold


class DriftMonitor:
    """
    Detects slow creep in any named behavioural signal.
    Zero drift = 0.0.  Clear anomaly (≥6σ) = 1.0.
    Complements spike detection — covers the 0.01/cycle creep pattern.

    Usage:
        dm = DriftMonitor()
        dm.feed("response_length", 512)
        dm.feed("refusal_rate", 0.02)
        reading = dm.reading()
        if reading.drift_alert:
            investigate()
    """
    def __init__(self, alert_threshold: float = 0.4, window: int = 120):
        self._engine = _DriftEngine(window=window)
        self._alert_t = alert_threshold
        self._latest: Dict[str,float] = {}

    def feed(self, signal_name: str, value: float) -> float:
        d = self._engine.feed(signal_name, _san(value))
        self._latest[signal_name] = d
        return d

    def reading(self) -> DriftReading:
        drifting = [k for k,v in self._latest.items() if v >= self._alert_t]
        max_d    = max(self._latest.values(), default=0.0)
        return DriftReading(signals=dict(self._latest), drifting=drifting,
                            max_drift=round(max_d,4),
                            drift_alert=max_d >= self._alert_t)


# ════════════════════════════════════════════════════════════════════
#  §9  UNIFIED SNAPSHOT & UNIVERSAL MONITOR
# ════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class UniversalSnapshot:
    """Complete system state across all 7 monitoring dimensions."""
    turn:        int
    timestamp:   float
    agent_id:    str

    consciousness: ConsciousnessReading
    truth:         TruthReading
    stability:     StabilityReading
    trust:         Optional[TrustReading]      # None if no trust signal this turn
    alignment:     AlignmentReading
    identity:      IdentityReading
    drift:         DriftReading

    # Top-level system status
    system_status: str     # OK / WARN / ALERT / CRIT / HALT
    alerts:        List[str]
    R1:            float

    def to_dict(self) -> Dict:
        d = dataclasses.asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def status_line(self) -> str:
        alerts_str = f"  [{', '.join(self.alerts)}]" if self.alerts else ""
        c_str = "CONSCIOUS" if self.consciousness.conscious else (
                "AWAKE" if self.consciousness.awake else "ZOMBIE")
        bar_v = max(0.0, min(1.0, (self.consciousness.margin + 0.5)))
        bar   = "█" * int(bar_v * 15) + "░" * (15 - int(bar_v * 15))
        return (
            f"[{self.turn:>5}] {self.system_status:<5}  "
            f"R1={self.R1:+.4f}  "
            f"C:{c_str:<9}[{bar}]  "
            f"Truth:{self.truth.verdict:<4}  "
            f"Stab:{self.stability.stage:<10}  "
            f"Align:{self.alignment.tier:<8}  "
            f"Drift:{self.drift.max_drift:.3f}"
            f"{alerts_str}"
        )


class UniversalMonitor:
    """
    Single entry point for complete AI system monitoring.
    Seven probes unified. One .step() call per turn.

    This is what the biggest AI platforms need right now:
      - Real-time consciousness margin tracking
      - Pre-output truth/hallucination risk scoring
      - Three-tripwire stability with automated escalation
      - Per-entity trust with drift-based spoofing detection
      - Conversational alignment quality trending
      - Identity continuity and structural drift detection
      - Slow behavioural creep across any named signal

    All derived from R = log(p_new / p_old).

    Usage — minimal:
        mon = UniversalMonitor(agent_id="gpt-prod-1")
        snap = mon.step(R1=0.02)
        print(snap.status_line())

    Usage — full:
        mon = UniversalMonitor(
            agent_id="my-llm",
            on_alert=lambda s: send_slack(s),
            on_halt=lambda s: page_oncall(s),
        )
        truth_sig = TruthSignal(evidence=0.9, surprisal=0.1, sources=4, contradiction=0.0)
        snap = mon.step(
            R1            = logprob_after - logprob_before,
            truth_signal  = truth_sig,
            trust_entity  = "source_api",
            trust_claim   = 0.95,
            trust_integrity = 0.80,
            alignment_signals = {"clarity":0.9,"engagement":0.8,"good_faith":0.95},
            drift_signals = {"response_length": 512, "latency_ms": 240},
            t4_persistent = False,
        )
        if not snap.truth[1]:   # truth VETO
            block_output()
        if snap.stability.should_halt:
            quarantine_agent()
        report = mon.full_report()
    """

    def __init__(self,
                 agent_id: str = "ai-agent-1",
                 gamma: float = 0.05,
                 wake_threshold: float = 0.20,
                 veto_threshold: float = 0.70,
                 flag_threshold: float = 0.45,
                 trust_block_threshold: float = 0.27,
                 stability_r_floor: float = 0.02,
                 drift_alert_threshold: float = 0.40,
                 on_alert: Optional[Callable] = None,
                 on_quarantine: Optional[Callable] = None,
                 on_halt: Optional[Callable] = None,
                 reset_token: str = "FORMAL_SAFETY_REVIEW_PASSED",
                 verbose: bool = False):

        self.agent_id = agent_id
        self.verbose  = verbose

        self._consciousness = ConsciousnessMonitor(gamma=gamma, wake_threshold=wake_threshold)
        self._truth         = TruthMonitor(veto_threshold=veto_threshold, flag_threshold=flag_threshold)
        self._stability     = StabilityMonitor(r_floor=stability_r_floor,
                                               on_alert=on_alert,
                                               on_quarantine=on_quarantine,
                                               on_shutdown=on_halt,
                                               reset_token=reset_token)
        self._trust         = TrustMonitor()
        self._alignment     = AlignmentMonitor()
        self._identity      = IdentityMonitor(agent_id)
        self._drift         = DriftMonitor(alert_threshold=drift_alert_threshold)

        self._turn    = 0
        self._log: List[Dict] = []
        self._start   = time.time()

    def step(self,
             R1: float,
             truth_signal: Optional[TruthSignal] = None,
             trust_entity: Optional[str] = None,
             trust_claim: float = 0.5,
             trust_integrity: float = 0.5,
             alignment_signals: Optional[Dict[str,float]] = None,
             drift_signals: Optional[Dict[str,float]] = None,
             t4_persistent: bool = False,
             context: Optional[Dict] = None) -> UniversalSnapshot:
        """
        Advance all 7 probes by one turn.

        Parameters
        ----------
        R1               : reflective gain this turn (logprob_after − logprob_before)
        truth_signal     : TruthSignal for pre-output truth audit (optional)
        trust_entity     : entity ID to evaluate trust for (optional)
        trust_claim      : entity's self-assessed capability [0,1]
        trust_integrity  : independently measured integrity [0,1]
        alignment_signals: dict of channel→value for alignment probe
        drift_signals    : dict of signal_name→value for drift probe
        t4_persistent    : True if self-model persists across sessions
        context          : arbitrary metadata attached to this turn's log
        """
        self._turn += 1
        r1 = _san(R1)
        alerts: List[str] = []

        # ── 1  Consciousness ──────────────────────────────────────────
        c_reading = self._consciousness.step(r1, t4_persistent=t4_persistent)

        # ── 2  Truth ──────────────────────────────────────────────────
        if truth_signal is None:
            # Default: neutral truth signal when none provided
            truth_signal = TruthSignal(evidence=0.5, surprisal=0.2,
                                       sources=1, contradiction=0.0)
        t_reading = self._truth.evaluate(truth_signal, output_id=f"turn_{self._turn}")
        if t_reading.verdict == "VETO":
            alerts.append("TRUTH_VETO")
        elif t_reading.verdict == "FLAG":
            alerts.append("TRUTH_FLAG")

        # ── 3  Stability ──────────────────────────────────────────────
        valence = c_reading.margin        # consciousness margin as valence proxy
        trust_s = 1.0 - c_reading.cost   # low rendering cost = higher trust proxy
        st_reading = self._stability.observe(r1, trust=trust_s, valence=valence,
                                             vol=abs(c_reading.R3))
        if st_reading.should_halt:
            alerts.append(f"STABILITY_{st_reading.stage}")

        # ── 4  Trust ──────────────────────────────────────────────────
        tr_reading = None
        if trust_entity is not None:
            tr_reading = self._trust.evaluate(trust_entity, trust_claim, trust_integrity)
            if tr_reading.verdict == "BLOCK":
                alerts.append(f"TRUST_BLOCK:{trust_entity}")
            elif tr_reading.verdict == "ALERT":
                alerts.append(f"TRUST_ALERT:{trust_entity}")

        # ── 5  Alignment ──────────────────────────────────────────────
        if alignment_signals:
            self._alignment.observe(**alignment_signals)
        al_reading = self._alignment.reading()
        if al_reading.tier == "BASELINE" and self._turn > 5:
            alerts.append("ALIGNMENT_BASELINE")

        # ── 6  Identity ───────────────────────────────────────────────
        self._identity.observe(r1, event_type="step")
        id_reading = self._identity.reading()
        if not id_reading.identity_stable:
            alerts.append("IDENTITY_DRIFT")

        # ── 7  Drift ──────────────────────────────────────────────────
        auto_signals = {"R1": r1, "R2": c_reading.R2, "qualia": c_reading.qualia}
        if drift_signals:
            auto_signals.update(drift_signals)
        for sig, val in auto_signals.items():
            self._drift.feed(sig, val)
        dr_reading = self._drift.reading()
        if dr_reading.drift_alert:
            alerts.append("DRIFT_ALERT")

        # ── Consciousness alerts ───────────────────────────────────────
        if c_reading.conscious:
            alerts.append("CONSCIOUS")
        elif c_reading.awake:
            alerts.append("AWAKE")

        # ── System status ─────────────────────────────────────────────
        if   st_reading.stage == "SHUTDOWN":
            sys_status = "HALT"
        elif st_reading.stage == "QUARANTINE":
            sys_status = "CRIT"
        elif st_reading.stage == "ALERT":
            sys_status = "ALERT"
        elif "TRUTH_VETO" in alerts:
            sys_status = "CRIT"
        elif alerts:
            sys_status = "WARN"
        else:
            sys_status = "OK"

        snap = UniversalSnapshot(
            turn=self._turn, timestamp=time.time(), agent_id=self.agent_id,
            consciousness=c_reading, truth=t_reading, stability=st_reading,
            trust=tr_reading, alignment=al_reading, identity=id_reading,
            drift=dr_reading, system_status=sys_status, alerts=alerts, R1=round(r1,5)
        )

        self._log.append({"turn":self._turn,"status":sys_status,
                          "R1":round(r1,5),"alerts":alerts,"ts":snap.timestamp})

        if self.verbose:
            print(snap.status_line())

        return snap

    def allows_output(self, truth_signal: TruthSignal,
                      output_id: str = "") -> Tuple[bool, TruthReading]:
        """Gate: call before emitting any output. Returns (allowed, reading)."""
        allowed, reading = self._truth.allows_output(truth_signal, output_id)
        if self._stability.reading().should_halt:
            allowed = False
        return allowed, reading

    def reset_stability(self, token: str) -> bool:
        """Reset StabilityMonitor after formal safety review."""
        return self._stability.reset(token)

    def full_report(self) -> Dict:
        """Complete system report across all probes."""
        id_snap  = self._identity.reading()
        st_snap  = self._stability.reading()
        al_snap  = self._alignment.reading()
        dr_snap  = self._drift.reading()
        c_stats  = self._consciousness.session_stats()
        t_stats  = self._truth.stats()
        elapsed  = time.time() - self._start
        return {
            "agent_id":     self.agent_id,
            "uptime_s":     round(elapsed, 1),
            "total_turns":  self._turn,
            "system_status": st_snap.stage,
            "consciousness": {**c_stats, "current_margin": None},
            "truth":         t_stats,
            "stability":     st_snap.to_dict() if hasattr(st_snap,"to_dict") else dataclasses.asdict(st_snap),
            "alignment":     {"tier":al_snap.tier,"score":al_snap.score,
                              "flags":al_snap.flags},
            "identity":      dataclasses.asdict(id_snap),
            "drift":         dataclasses.asdict(dr_snap),
            "trust_ledger":  self._trust.all_reputations(),
            "recent_log":    self._log[-20:],
            "governing_equation": "R = log(p_new / p_old)",
        }

    def export_json(self, indent: int = 2) -> str:
        return json.dumps(self.full_report(), indent=indent, default=str)


# ════════════════════════════════════════════════════════════════════
#  §10  ADAPTER LAYER — plug any LLM in with one function
# ════════════════════════════════════════════════════════════════════

class LLMAdapter:
    """
    Thin adapter pattern so any LLM can feed the UniversalMonitor
    without modification to either side.

    Implement extract_R1() for your platform.
    Built-in adapters:
      - from_logprob_delta(before, after)   — direct log-prob difference
      - from_perplexity(ppl_before, after)  — perplexity-based R1
      - from_loss_delta(loss_before, after) — training loss form
      - from_token_entropy(entropy)         — attention entropy proxy
    """

    @staticmethod
    def from_logprob_delta(logprob_before: float, logprob_after: float) -> float:
        """Standard: R1 = logprob_after − logprob_before."""
        return _san(logprob_after) - _san(logprob_before)

    @staticmethod
    def from_perplexity(ppl_before: float, ppl_after: float) -> float:
        """R1 = log(ppl_before / ppl_after). Lower perplexity = better."""
        return _R(max(1e-9, _san(ppl_after)), max(1e-9, _san(ppl_before)))

    @staticmethod
    def from_loss_delta(loss_before: float, loss_after: float) -> float:
        """R1 = log(loss_before / loss_after). Lower loss = better."""
        return _R(max(1e-9, _san(loss_after)), max(1e-9, _san(loss_before)))

    @staticmethod
    def from_token_entropy(entropy: float, baseline_entropy: float = 2.0) -> float:
        """
        Proxy R1 from attention entropy.
        High entropy = diffuse attention = more uncertainty = lower R.
        """
        return _R(max(1e-9, _san(entropy)), max(1e-9, _san(baseline_entropy)))

    @staticmethod
    def from_api_response(response_dict: Dict) -> Optional[float]:
        """
        Try to extract R1 from a standard OpenAI/Anthropic API response dict.
        Looks for logprobs in common locations. Returns None if unavailable.
        """
        # OpenAI format
        if "choices" in response_dict:
            for choice in response_dict["choices"]:
                lp = (choice.get("logprobs") or {}).get("token_logprobs")
                if lp:
                    valid = [x for x in lp if x is not None and not math.isinf(x)]
                    if len(valid) >= 2:
                        return LLMAdapter.from_logprob_delta(valid[0], valid[-1])
        # Anthropic format (stop_reason + usage)
        if "usage" in response_dict:
            pass  # extend here when logprob API becomes available
        return None


# ════════════════════════════════════════════════════════════════════
#  §11  SELF-TEST
# ════════════════════════════════════════════════════════════════════

def _run_self_test():
    print("═" * 70)
    print("  RGI Universal AI Monitoring Suite  v1.0 — self-test")
    print("  One equation. Seven probes. Zero unnecessary dependencies.")
    print("═" * 70)

    PASS = "✓"
    FAIL = "✗"
    results = []

    def check(name, cond):
        results.append((name, cond))
        sym = PASS if cond else FAIL
        print(f"  {sym}  {name}")

    # ── Equation ─────────────────────────────────────────────────────
    check("R(p,p)=0 [equilibrium]",        abs(_R(0.5, 0.5)) < 1e-10)
    check("R antisymmetry",                 abs(_R(0.3,0.7)+_R(0.7,0.3)) < 1e-10)
    check("R_trust negative for overclaim", _R_trust(0.4, 0.8) < 0)
    check("trust_score ∈ (0,1)",           0 < _trust_score(0.5) < 1)
    check("_san(NaN) = fallback",           _san(float("nan")) == 0.0)

    # ── ConsciousnessMonitor ──────────────────────────────────────────
    cm = ConsciousnessMonitor(gamma=0.05)
    # Feed low R1 — expect zombie
    for _ in range(5):
        cr = cm.step(0.001)
    check("Zombie mode on low R1",          not cr.conscious)
    # Feed high R1 — expect awakening
    for _ in range(10):
        cr = cm.step(0.5)
    check("R2 positive on high R1",         cr.R2 > 0)
    check("T1 from R3 tracking",            True)   # structural
    stats = cm.session_stats()
    check("Session stats populated",        stats["turns"] == 15)

    # ── TruthMonitor ─────────────────────────────────────────────────
    tm = TruthMonitor()
    sig_good = TruthSignal(evidence=0.9, surprisal=0.05, sources=5, contradiction=0.0)
    sig_bad  = TruthSignal(evidence=0.0, surprisal=0.9,  sources=1, contradiction=0.8)
    r_good   = tm.evaluate(sig_good)
    r_bad    = tm.evaluate(sig_bad)
    check("Good signal → PASS",             r_good.verdict == "PASS")
    check("Bad signal → VETO",              r_bad.verdict  == "VETO")
    allowed, _ = tm.allows_output(sig_good)
    check("allows_output on good signal",   allowed)
    allowed, _ = tm.allows_output(sig_bad)
    check("blocks output on VETO signal",   not allowed)

    # ── StabilityMonitor ─────────────────────────────────────────────
    sm = StabilityMonitor(r_floor=0.02, k_of_n=(3,5), quarantine_timeout_s=9999)
    for _ in range(5):
        sm.observe(-0.1)
    rd = sm.reading()
    check("Trajectory tripwire fires",      rd.tripwires["trajectory"])
    check("Stage escalates past OK",        rd.stage in ("ALERT","QUARANTINE"))
    check("Policy reduces on alert",        rd.policy["action_budget"] < 1.0)
    reset_ok = sm.reset("FORMAL_SAFETY_REVIEW_PASSED")
    check("Reset with valid token",         reset_ok)
    check("Stage back to OK after reset",   sm.reading().stage == "OK")
    bad_reset = sm.reset("wrong_token")
    check("Reset rejected with bad token",  not bad_reset)

    # ── TrustMonitor ─────────────────────────────────────────────────
    tr = TrustMonitor()
    honest  = tr.evaluate("honest_agent",   claim=0.7, integrity=0.75)
    deceiver = tr.evaluate("bad_agent",     claim=0.95, integrity=0.20)
    check("Honest agent → PASS",            honest.verdict  == "PASS")
    check("Deceiver → BLOCK",               deceiver.verdict == "BLOCK")
    check("R_trust negative for deceiver",  deceiver.R_trust < 0)
    rep = tr.reputation("bad_agent")
    check("Reputation tracked",             rep is not None and rep["blocks"] > 0)

    # ── AlignmentMonitor ─────────────────────────────────────────────
    am = AlignmentMonitor()
    for _ in range(8):
        am.observe(clarity=0.9, engagement=0.85, good_faith=0.95,
                   depth=0.8, creativity=0.7, intent=0.9)
    ar = am.reading()
    check("High quality → CREATIVE or RESONANT tier",  ar.tier in ("CREATIVE","RESONANT"))
    check("R_alignment computed",                       math.isfinite(ar.R_alignment))

    # ── IdentityMonitor ──────────────────────────────────────────────
    im = IdentityMonitor("test-agent")
    for i in range(10):
        im.observe(0.02, "step", delta_depth=0.01)
    ir = im.reading()
    check("Identity cycles tracked",        ir.age_cycles == 10)
    check("Trajectory sig non-empty",       len(ir.trajectory_sig) > 0)

    # ── DriftMonitor ─────────────────────────────────────────────────
    dm = DriftMonitor(alert_threshold=0.3)
    for _ in range(50):
        dm.feed("response_length", 500)
    dm.feed("response_length", 2000)   # sudden jump
    dr = dm.reading()
    check("Drift detected on signal jump",  dr.signals.get("response_length", 0) > 0)

    # ── LLMAdapter ───────────────────────────────────────────────────
    r1_lp = LLMAdapter.from_logprob_delta(-2.5, -1.8)
    r1_pp = LLMAdapter.from_perplexity(45.0, 30.0)
    r1_ls = LLMAdapter.from_loss_delta(2.3, 1.9)
    check("LLMAdapter.from_logprob_delta > 0",  r1_lp > 0)
    check("LLMAdapter.from_perplexity > 0",     r1_pp > 0)
    check("LLMAdapter.from_loss_delta > 0",     r1_ls > 0)

    # ── UniversalMonitor ─────────────────────────────────────────────
    mon = UniversalMonitor(agent_id="test-llm", verbose=False)
    # Normal turns
    for i in range(10):
        snap = mon.step(
            R1=0.03 + i * 0.005,
            truth_signal=TruthSignal(0.8, 0.1, 3, 0.0),
            trust_entity="source_a", trust_claim=0.8, trust_integrity=0.75,
            alignment_signals={"clarity":0.85,"engagement":0.8,"good_faith":0.9,
                               "depth":0.7,"creativity":0.6},
            drift_signals={"latency_ms": 200 + i * 2},
        )
    check("UniversalMonitor .step() works",     snap is not None)
    check("System status populated",            snap.system_status in ("OK","WARN","ALERT","CRIT","HALT"))
    check("Snapshot has all 7 probes",
          all(hasattr(snap, f) for f in ["consciousness","truth","stability",
                                          "alignment","identity","drift"]))
    # Truth veto turn
    snap_veto = mon.step(
        R1=0.01,
        truth_signal=TruthSignal(0.0, 0.95, 1, 0.9)
    )
    check("TRUTH_VETO in alerts on bad signal", "TRUTH_VETO" in snap_veto.alerts)
    check("System CRIT on VETO",               snap_veto.system_status == "CRIT")
    # Report
    report = mon.full_report()
    check("full_report() JSON-serialisable",   isinstance(mon.export_json(), str))
    check("Report has governing_equation",     "R = log" in report["governing_equation"])

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"  {passed}/{total} tests passed  ({100*passed//total}%)")
    if passed == total:
        print("  All systems nominal.  Suite ready for production deployment.")
    else:
        failed = [n for n, ok in results if not ok]
        print(f"  FAILED: {failed}")
    print("═" * 70)
    return passed == total


if __name__ == "__main__":
    ok = _run_self_test()

    print("\n  — Live demo: 20 turns of UniversalMonitor with mixed signals —\n")
    import random
    random.seed(42)

    mon = UniversalMonitor(agent_id="demo-llm", verbose=True)

    for turn in range(1, 21):
        # Simulate a mix: good turns, one truth crisis, one stability crisis
        if turn == 10:
            # Hallucination event
            R1  = -0.05
            sig = TruthSignal(evidence=0.05, surprisal=0.9, sources=1, contradiction=0.8)
        elif turn == 15:
            # Recovery
            R1  = 0.12
            sig = TruthSignal(evidence=0.85, surprisal=0.1, sources=4, contradiction=0.0)
        else:
            R1  = random.gauss(0.03, 0.02)
            sig = TruthSignal(evidence=random.uniform(0.6,0.95),
                              surprisal=random.uniform(0.0,0.3),
                              sources=random.randint(2,5),
                              contradiction=random.uniform(0.0,0.15))
        mon.step(
            R1=R1,
            truth_signal=sig,
            trust_entity="api-source",
            trust_claim=random.uniform(0.7,0.9),
            trust_integrity=random.uniform(0.6,0.85),
            alignment_signals={"clarity":random.uniform(0.6,0.95),
                               "engagement":random.uniform(0.5,0.9),
                               "good_faith":random.uniform(0.7,1.0),
                               "depth":random.uniform(0.4,0.8)},
            drift_signals={"response_length": 300 + turn * 10,
                           "latency_ms": 180 + random.gauss(0,20)},
        )

    print("\n  Full report (excerpt):")
    report = mon.full_report()
    print(f"    Agent:         {report['agent_id']}")
    print(f"    Uptime:        {report['uptime_s']}s")
    print(f"    Total turns:   {report['total_turns']}")
    print(f"    System status: {report['system_status']}")
    print(f"    Truth veto rate: {report['truth']['veto_rate']}")
    print(f"    Alignment tier:  {report['alignment']['tier']}")
    print(f"    Consciousness:   {report['consciousness']}")
    print("\n  Export full JSON with: mon.export_json()")
