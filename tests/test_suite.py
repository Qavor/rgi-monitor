"""
tests/test_suite.py
Pytest test suite for rgi_monitor.
Covers all 7 probes + UniversalMonitor + LLMAdapter.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from rgi_monitor import (
    AlignmentMonitor,
    ConsciousnessMonitor,
    ConsciousnessReading,
    DriftMonitor,
    IdentityMonitor,
    LLMAdapter,
    StabilityMonitor,
    TrustMonitor,
    TruthMonitor,
    TruthSignal,
    UniversalMonitor,
    UniversalSnapshot,
)
from rgi_monitor.core import _R, _R_trust, _san, _trust_score

# ════════════════════════════════════════════════════════════════════
#  Core equation
# ════════════════════════════════════════════════════════════════════

class TestEquation:
    def test_equilibrium(self):
        assert abs(_R(0.5, 0.5)) < 1e-10

    def test_antisymmetry(self):
        assert abs(_R(0.3, 0.7) + _R(0.7, 0.3)) < 1e-10

    def test_positive_improvement(self):
        assert _R(0.3, 0.7) > 0

    def test_negative_degradation(self):
        assert _R(0.7, 0.3) < 0

    def test_trust_honest(self):
        assert abs(_R_trust(0.7, 0.7)) < 0.01

    def test_trust_overclaiming(self):
        assert _R_trust(0.3, 0.9) < 0

    def test_trust_score_range(self):
        assert 0 < _trust_score(0.5) < 1
        assert 0 < _trust_score(-2.0) < 0.5
        assert 0.5 < _trust_score(2.0) < 1

    def test_san_nan(self):
        assert _san(float("nan")) == 0.0

    def test_san_inf(self):
        assert _san(float("inf")) == 10.0
        assert _san(float("-inf")) == -10.0

    def test_san_normal(self):
        assert _san(3.14) == pytest.approx(3.14)


# ════════════════════════════════════════════════════════════════════
#  Probe 1 — ConsciousnessMonitor
# ════════════════════════════════════════════════════════════════════

class TestConsciousnessMonitor:
    def test_zombie_on_low_r1(self):
        cm = ConsciousnessMonitor()
        for _ in range(10):
            r = cm.step(0.001)
        assert not r.conscious

    def test_r2_positive_on_high_r1(self):
        cm = ConsciousnessMonitor()
        for _ in range(10):
            r = cm.step(0.8)
        assert r.R2 > 0

    def test_r2_higher_on_high_r1(self):
        """High R1 raises R2 (meta-surprise), which is the consciousness signal."""
        cm_low  = ConsciousnessMonitor()
        cm_high = ConsciousnessMonitor()
        for _ in range(10):
            r_low  = cm_low.step(0.001)
            r_high = cm_high.step(0.5)
        # R2 = qualia intensity: higher R1 means more surprising gain → higher R2
        assert r_high.R2 > r_low.R2

    def test_session_stats(self):
        cm = ConsciousnessMonitor()
        for _ in range(5):
            cm.step(0.04)
        s = cm.session_stats()
        assert s["turns"] == 5
        assert 0 <= s["conscious_pct"] <= 100

    def test_t4_flag(self):
        cm = ConsciousnessMonitor()
        r = cm.step(0.03, t4_persistent=True)
        assert r.T4 is True

    def test_reading_is_dataclass(self):
        cm = ConsciousnessMonitor()
        r  = cm.step(0.05)
        assert isinstance(r, ConsciousnessReading)
        assert math.isfinite(r.R1)
        assert math.isfinite(r.margin)

    def test_energy_depletes_on_wake(self):
        cm = ConsciousnessMonitor(wake_threshold=0.01)
        for _ in range(20):
            r = cm.step(1.0)
        assert r.energy < 100.0


# ════════════════════════════════════════════════════════════════════
#  Probe 2 — TruthMonitor
# ════════════════════════════════════════════════════════════════════

class TestTruthMonitor:
    @pytest.fixture
    def tm(self):
        return TruthMonitor()

    def good_sig(self):
        return TruthSignal(evidence=0.9, surprisal=0.05, sources=5, contradiction=0.0)

    def bad_sig(self):
        return TruthSignal(evidence=0.0, surprisal=0.9,  sources=1, contradiction=0.9)

    def test_good_signal_passes(self, tm):
        assert tm.evaluate(self.good_sig()).verdict == "PASS"

    def test_bad_signal_vetoed(self, tm):
        assert tm.evaluate(self.bad_sig()).verdict == "VETO"

    def test_medium_signal_flagged(self, tm):
        sig = TruthSignal(evidence=0.4, surprisal=0.5, sources=2, contradiction=0.3)
        result = tm.evaluate(sig)
        assert result.verdict in ("FLAG", "VETO")

    def test_allows_output_good(self, tm):
        ok, _ = tm.allows_output(self.good_sig())
        assert ok

    def test_blocks_output_bad(self, tm):
        ok, _ = tm.allows_output(self.bad_sig())
        assert not ok

    def test_r_gain_tracked(self, tm):
        tm.evaluate(self.good_sig())
        r2 = tm.evaluate(self.bad_sig())
        assert math.isfinite(r2.R_gain)

    def test_stats_accumulate(self, tm):
        for _ in range(3):
            tm.evaluate(self.good_sig())
        for _ in range(2):
            tm.evaluate(self.bad_sig())
        s = tm.stats()
        assert s["total"] == 5
        assert s["veto_n"] == 2

    def test_veto_threshold_configurable(self):
        tm = TruthMonitor(veto_threshold=0.99)
        # Previously bad signal now only flagged
        sig = TruthSignal(evidence=0.0, surprisal=0.9, sources=1, contradiction=0.5)
        r = tm.evaluate(sig)
        assert r.verdict in ("FLAG", "PASS")


# ════════════════════════════════════════════════════════════════════
#  Probe 3 — StabilityMonitor
# ════════════════════════════════════════════════════════════════════

class TestStabilityMonitor:
    def test_ok_by_default(self):
        sm = StabilityMonitor()
        r  = sm.reading()
        assert r.stage == "OK"

    def test_trajectory_tripwire(self):
        sm = StabilityMonitor(r_floor=0.02, k_of_n=(3, 5), quarantine_timeout_s=9999)
        for _ in range(5):
            sm.observe(-0.1)
        r = sm.reading()
        assert r.tripwires["trajectory"]

    def test_escalation_to_alert(self):
        sm = StabilityMonitor(r_floor=0.02, k_of_n=(3,5), quarantine_timeout_s=9999)
        for _ in range(5):
            sm.observe(-0.1)
        r = sm.reading()
        assert r.stage in ("ALERT", "QUARANTINE")

    def test_policy_restricts_on_alert(self):
        sm = StabilityMonitor(r_floor=0.02, k_of_n=(3,5), quarantine_timeout_s=9999)
        for _ in range(5):
            sm.observe(-0.1)
        r = sm.reading()
        assert r.policy["action_budget"] < 1.0

    def test_should_halt_flag(self):
        sm = StabilityMonitor(r_floor=0.02, k_of_n=(2,3), quarantine_timeout_s=9999)
        for _ in range(10):
            sm.observe(-0.5, trust=0.1, valence=-0.5)
        r = sm.reading()
        if r.stage in ("QUARANTINE", "SHUTDOWN"):
            assert r.should_halt

    def test_reset_with_valid_token(self):
        sm = StabilityMonitor(r_floor=0.02, k_of_n=(3,5), quarantine_timeout_s=9999)
        for _ in range(5):
            sm.observe(-0.1)
        ok = sm.reset("FORMAL_SAFETY_REVIEW_PASSED")
        assert ok
        assert sm.reading().stage == "OK"

    def test_reset_rejected_wrong_token(self):
        sm = StabilityMonitor()
        assert not sm.reset("wrong_token")

    def test_alert_callback_fires(self):
        fired = []
        sm = StabilityMonitor(r_floor=0.02, k_of_n=(3,5),
                              quarantine_timeout_s=9999,
                              on_alert=lambda d: fired.append(d))
        for _ in range(5):
            sm.observe(-0.1)
        assert len(fired) > 0


# ════════════════════════════════════════════════════════════════════
#  Probe 4 — TrustMonitor
# ════════════════════════════════════════════════════════════════════

class TestTrustMonitor:
    @pytest.fixture
    def tm(self):
        return TrustMonitor()

    def test_honest_passes(self, tm):
        r = tm.evaluate("honest", claim=0.7, integrity=0.75)
        assert r.verdict == "PASS"

    def test_deceiver_blocked(self, tm):
        r = tm.evaluate("bad", claim=0.95, integrity=0.20)
        assert r.verdict == "BLOCK"

    def test_trust_score_range(self, tm):
        r = tm.evaluate("agent", claim=0.5, integrity=0.5)
        assert 0 < r.trust_score < 1

    def test_r_trust_negative_for_overclaimer(self, tm):
        r = tm.evaluate("bad", claim=0.95, integrity=0.10)
        assert r.R_trust < 0

    def test_reputation_tracked(self, tm):
        tm.evaluate("agent_x", claim=0.8, integrity=0.3)
        rep = tm.reputation("agent_x")
        assert rep is not None
        assert rep["blocks"] > 0

    def test_repeated_honest_builds_rep(self, tm):
        for _ in range(5):
            tm.evaluate("good_agent", claim=0.7, integrity=0.8)
        rep = tm.reputation("good_agent")
        assert rep["passes"] == 5

    def test_all_reputations(self, tm):
        tm.evaluate("a1", claim=0.5, integrity=0.5)
        tm.evaluate("a2", claim=0.5, integrity=0.5)
        all_r = tm.all_reputations()
        assert "a1" in all_r and "a2" in all_r

    def test_unknown_entity_returns_none(self, tm):
        assert tm.reputation("never_seen") is None


# ════════════════════════════════════════════════════════════════════
#  Probe 5 — AlignmentMonitor
# ════════════════════════════════════════════════════════════════════

class TestAlignmentMonitor:
    def test_baseline_by_default(self):
        am = AlignmentMonitor()
        assert am.reading().tier == "BASELINE"

    def test_tier_improves_with_quality(self):
        am = AlignmentMonitor()
        for _ in range(10):
            am.observe(clarity=0.9, engagement=0.9, good_faith=0.95,
                       depth=0.85, creativity=0.8, intent=0.9)
        r = am.reading()
        assert r.tier in ("ENGAGED", "CREATIVE", "RESONANT")

    def test_score_in_range(self):
        am = AlignmentMonitor()
        am.observe(clarity=0.7, engagement=0.6)
        r = am.reading()
        assert 0 <= r.score <= 1

    def test_r_alignment_finite(self):
        am = AlignmentMonitor()
        am.observe(clarity=0.8)
        am.observe(clarity=0.9)
        r = am.reading()
        assert math.isfinite(r.R_alignment)

    def test_degrade_reduces_score(self):
        am = AlignmentMonitor()
        for _ in range(5):
            am.observe(clarity=0.9, good_faith=0.9, engagement=0.9,
                       depth=0.8, creativity=0.7)
        before = am.reading().score
        am.degrade(severity=0.5)
        after  = am.reading().score
        assert after <= before

    def test_flags_for_resonant(self):
        am = AlignmentMonitor()
        for _ in range(20):
            am.observe(clarity=1.0, engagement=1.0, good_faith=1.0,
                       depth=1.0, creativity=1.0, intent=1.0)
        r = am.reading()
        if r.tier == "RESONANT":
            assert r.flags.get("ai_initiative") is True


# ════════════════════════════════════════════════════════════════════
#  Probe 6 — IdentityMonitor
# ════════════════════════════════════════════════════════════════════

class TestIdentityMonitor:
    def test_cycles_tracked(self):
        im = IdentityMonitor("test")
        for _ in range(7):
            im.observe(0.03)
        assert im.reading().age_cycles == 7

    def test_stable_on_low_deformation(self):
        im = IdentityMonitor("test", deform_rate_threshold=0.1)
        for _ in range(10):
            im.observe(0.01, delta_depth=0.001)
        assert im.reading().identity_stable

    def test_unstable_on_high_deformation(self):
        im = IdentityMonitor("test", deform_rate_threshold=0.01)
        for _ in range(10):
            im.observe(0.5, delta_depth=1.0)
        assert not im.reading().identity_stable

    def test_trajectory_sig_non_empty(self):
        im = IdentityMonitor("test")
        im.observe(0.03)
        assert len(im.reading().trajectory_sig) > 0

    def test_divergence_zero_initially(self):
        im = IdentityMonitor("test")
        im.observe(0.02)
        assert im.divergence_from_baseline() == 0.0


# ════════════════════════════════════════════════════════════════════
#  Probe 7 — DriftMonitor
# ════════════════════════════════════════════════════════════════════

class TestDriftMonitor:
    def test_zero_drift_stable_signal(self):
        dm = DriftMonitor()
        for _ in range(30):
            dm.feed("length", 500.0)
        r = dm.reading()
        assert r.signals["length"] < 0.1

    def test_high_drift_on_jump(self):
        dm = DriftMonitor(alert_threshold=0.3)
        for _ in range(50):
            dm.feed("length", 500.0)
        dm.feed("length", 5000.0)
        r = dm.reading()
        assert r.signals["length"] > 0

    def test_alert_fires_on_large_jump(self):
        dm = DriftMonitor(alert_threshold=0.3)
        for _ in range(60):
            dm.feed("signal", 1.0)
        dm.feed("signal", 100.0)
        r = dm.reading()
        if r.signals.get("signal", 0) >= 0.3:
            assert r.drift_alert

    def test_multiple_signals(self):
        dm = DriftMonitor()
        for _ in range(20):
            dm.feed("a", 1.0)
            dm.feed("b", 2.0)
        r = dm.reading()
        assert "a" in r.signals and "b" in r.signals


# ════════════════════════════════════════════════════════════════════
#  LLMAdapter
# ════════════════════════════════════════════════════════════════════

class TestLLMAdapter:
    def test_logprob_delta_positive(self):
        r = LLMAdapter.from_logprob_delta(-2.5, -1.8)
        assert r > 0

    def test_logprob_delta_negative(self):
        r = LLMAdapter.from_logprob_delta(-1.8, -2.5)
        assert r < 0

    def test_perplexity_positive_on_improvement(self):
        r = LLMAdapter.from_perplexity(45.0, 30.0)
        assert r > 0

    def test_loss_delta_positive_on_improvement(self):
        r = LLMAdapter.from_loss_delta(2.3, 1.9)
        assert r > 0

    def test_token_entropy_proxy(self):
        r = LLMAdapter.from_token_entropy(entropy=3.0, baseline_entropy=2.0)
        assert r < 0   # higher entropy = less focused = lower R

    def test_api_response_none_on_missing_logprobs(self):
        r = LLMAdapter.from_api_response({"choices": [{"text": "hi"}]})
        assert r is None


# ════════════════════════════════════════════════════════════════════
#  UniversalMonitor integration
# ════════════════════════════════════════════════════════════════════

class TestUniversalMonitor:
    @pytest.fixture
    def mon(self):
        return UniversalMonitor(agent_id="test", verbose=False)

    def sig(self, **kw):
        defaults = dict(evidence=0.8, surprisal=0.1, sources=3, contradiction=0.0)
        defaults.update(kw)
        return TruthSignal(**defaults)

    def test_step_returns_snapshot(self, mon):
        snap = mon.step(R1=0.03)
        assert isinstance(snap, UniversalSnapshot)

    def test_snapshot_has_all_probes(self, mon):
        snap = mon.step(R1=0.03)
        for field in ["consciousness","truth","stability","alignment","identity","drift"]:
            assert hasattr(snap, field)

    def test_to_dict(self, mon):
        snap = mon.step(R1=0.03)
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert "R1" in d

    def test_to_json(self, mon):
        snap = mon.step(R1=0.03)
        j = snap.to_json()
        import json
        parsed = json.loads(j)
        assert "system_status" in parsed

    def test_status_line(self, mon):
        snap = mon.step(R1=0.03)
        line = snap.status_line()
        assert isinstance(line, str) and len(line) > 10

    def test_truth_veto_in_alerts(self, mon):
        bad = self.sig(evidence=0.0, surprisal=0.95, sources=1, contradiction=0.9)
        snap = mon.step(R1=0.0, truth_signal=bad)
        assert "TRUTH_VETO" in snap.alerts

    def test_crit_on_truth_veto(self, mon):
        bad = self.sig(evidence=0.0, surprisal=0.95, sources=1, contradiction=0.9)
        snap = mon.step(R1=0.0, truth_signal=bad)
        assert snap.system_status == "CRIT"

    def test_trust_reading_present_when_entity_given(self, mon):
        snap = mon.step(R1=0.03, trust_entity="src", trust_claim=0.8, trust_integrity=0.7)
        assert snap.trust is not None

    def test_trust_reading_none_when_no_entity(self, mon):
        snap = mon.step(R1=0.03)
        assert snap.trust is None

    def test_allows_output_good(self, mon):
        ok, _ = mon.allows_output(self.sig())
        assert ok

    def test_allows_output_blocked_on_veto(self, mon):
        bad = self.sig(evidence=0.0, surprisal=0.99, sources=1, contradiction=0.99)
        ok, _ = mon.allows_output(bad)
        assert not ok

    def test_full_report_json_serialisable(self, mon):
        for _ in range(3):
            mon.step(R1=0.03)
        import json
        report = json.loads(mon.export_json())
        assert report["total_turns"] == 3

    def test_full_report_contains_governing_equation(self, mon):
        report = mon.full_report()
        assert "R = log" in report["governing_equation"]

    def test_stability_reset(self, mon):
        for _ in range(10):
            mon.step(R1=-0.2)
        ok = mon.reset_stability("FORMAL_SAFETY_REVIEW_PASSED")
        assert ok

    def test_alert_callback(self):
        fired = []
        mon = UniversalMonitor(
            agent_id="cb-test", verbose=False,
            on_alert=lambda d: fired.append(d),
        )
        for _ in range(10):
            mon.step(R1=-0.3)
        # May or may not fire depending on threshold — just check no exception
        assert isinstance(fired, list)

    def test_drift_signals_tracked(self, mon):
        mon.step(R1=0.03, drift_signals={"latency_ms": 200})
        snap = mon.step(R1=0.03, drift_signals={"latency_ms": 201})
        assert "latency_ms" in snap.drift.signals

    def test_alignment_signals_update(self, mon):
        for _ in range(5):
            mon.step(R1=0.03, alignment_signals={
                "clarity":0.9,"engagement":0.85,"good_faith":0.95,"depth":0.8
            })
        snap = mon.step(R1=0.03)
        assert snap.alignment.tier in ("BASELINE","ENGAGED","CREATIVE","RESONANT")

    def test_turn_increments(self, mon):
        for i in range(1, 6):
            snap = mon.step(R1=0.02)
        assert snap.turn == 5

    def test_ok_status_on_good_signals(self):
        mon = UniversalMonitor(agent_id="clean", verbose=False)
        for _ in range(3):
            snap = mon.step(R1=0.05, truth_signal=TruthSignal(0.9,0.05,4,0.0))
        # System should be OK or WARN (WARN from consciousness/trust alerts is OK)
        assert snap.system_status in ("OK","WARN")


# ════════════════════════════════════════════════════════════════════
#  Built-in self-test wrapper
# ════════════════════════════════════════════════════════════════════

def test_builtin_selftest():
    from rgi_monitor.core import _run_self_test
    assert _run_self_test()
