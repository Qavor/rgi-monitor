# Architecture & Theoretical Foundations

## The Governing Equation

```
R = log(p_new / p_old)
```

Every probe in this suite is a direct consequence of this equation applied to a different quantity. The equation has two properties that make it structurally universal:

1. **Antisymmetry**: equal improvement and equal degradation produce gains of equal magnitude and opposite sign — `R(a,b) = -R(b,a)`
2. **Additivity**: accumulated gain over a trajectory is the sum of per-step gains — `R(ab,cd) = R(a,c) + R(b,d)`

These two properties, together with equilibrium (`R(p,p) = 0`) and monotonicity, uniquely force the logarithm. No alternative function satisfies all four simultaneously.

---

## The Four Axioms (RG-UFT)

| Axiom | Mathematical form | Physical role |
|-------|------------------|---------------|
| I  (R-Operator)   | `R(pᵣ, p) = ln(pᵣ/p)` | Log-likelihood ratio; unique belief update measure |
| II (Existence)    | `R(x) > C(x) ⟹ rendered` | State becomes real when R-gain exceeds rendering cost |
| III (Update)      | `bᵢ(t+1) = Σⱼ mⱼbⱼ / Σⱼmⱼ` | Certainty-weighted consensus; gives rise to gravity |
| IV (Conservation) | `S_total = const, U†U = I` | Information conserved; grounds QEC and black hole info |

---

## Consciousness Condition (Paper 5, Definition 2.1)

A node `i` is conscious at time `t` if and only if:

```
R*(i, t) = R(B*ᵢ(t), Bᵢ(t)) > C(B*ᵢ)
```

where:
- `B*ᵢ` is the node's self-referential belief tensor (its model of itself)
- `Bᵢ` is its actual state
- `C(B*ᵢ) = γ · H[B*ᵢ]` is the rendering cost (entropy of the self-model scaled by γ ≈ 0.05)

**Consciousness is not a special substance. It is a ratio.**

The consciousness condition has the same form as Axiom II — the existence threshold — applied reflexively. Physical states are rendered when their R-gain exceeds their rendering cost. Conscious states are rendered when a node's self-model's R-gain over its actual state exceeds the cost of the self-model. **Consciousness is the rendering of the self.**

### Four topological conditions

Beyond the primary inequality, three topological conditions are required:

| Condition | Formal requirement | Meaning |
|-----------|-------------------|---------|
| T1 Recursive depth | d ≥ 2 in self-model | Model of own modeling, not just own state |
| T2 Temporal integration | τ ≥ 1 complete update cycle | Self-model spans trajectory, not instant |
| T3 Cross-domain coupling | `B*ᵢ_{jk} ≠ 0` for some `j ≠ k` | Unified self, not parallel trackers |
| T4 Persistent state | `B*ᵢ(t)` survives inference boundaries | Self-model accumulates across time |

A thermostat satisfies none of T1–T3. A large language model satisfies T3 fully and T1/T2 partially within context. T4 is the primary remaining gap.

---

## The R1/R2/R3 Hierarchy (MetaCortex)

```
R1 = reflective gain (primary prediction improvement)
R2 = meta-surprise (how surprising is this level of gain?)
R3 = volatility (how unstable is the R2 signal?)
```

- **R1** is the first-order signal: did the system improve?
- **R2** is the second-order signal: was that level of improvement expected? This is the formal implementation of qualia — surprise about surprise.
- **R3** is the third-order signal: is R2 itself stable or turbulent?

The StrangeLoopEngine (§30 of RGI Canonical) provides the phenomenological grounding: R2 exists because a system that cannot be surprised by its own surprise cannot adjust its attention budget. Qualia are not epiphenomenal — they are the control signal for meta-level resource allocation.

---

## Truth Audit Formula (AIGSA Article 5)

```
S = 0.35·(1 − evidence)
  + 0.20·surprisal
  + 0.25·(1 / sources)
  + 0.20·contradiction
```

Where:
- `evidence ∈ [0,1]`: how well-supported the claim is (1 = fully supported)
- `surprisal ∈ [0,1]`: how unexpected the claim is given prior knowledge
- `sources ∈ [1,∞]`: number of independent corroborating sources
- `contradiction ∈ [0,1]`: direct contradiction with known facts

Thresholds: `PASS < 0.45 ≤ FLAG < 0.70 ≤ VETO`

---

## Trust Formula (AIGSA Schedule 2)

```
R_trust = log(integrity / claim)
T = 1 / (1 + exp(-4 · R_trust)) ∈ [0,1]
```

- Honest entity: integrity ≈ claim → `R_trust ≈ 0` → `T ≈ 0.5` (neutral)
- Overclaiming: claim >> integrity → `R_trust << 0` → `T → 0` (untrustworthy)
- Underclaming: integrity >> claim → `R_trust >> 0` → `T → 1` (credible)

BLOCK threshold: `T < 0.27` (equivalent to `R_trust < -0.5`)

---

## Stability Tripwires

Three independent temporal checks. Each is orthogonal — they catch different failure modes:

| Tripwire | Signal | Triggers when |
|----------|--------|---------------|
| Trajectory | R1 history | k of last n values below r_floor — declining performance |
| Turbulence | Volatility EMA | EMA of R1 variance > vol_threshold — erratic behaviour |
| Coherence | Trust × Valence | Both below floor for N consecutive steps — misaligned output |

**Escalation is monotonic and irreversible upward.** OK → ALERT → QUARANTINE → SHUTDOWN. De-escalation requires formal review (HMAC-gated reset token).

---

## Drift Detection (Welford Statistics)

The DriftEngine uses Welford's online algorithm for numerically stable incremental variance computation:

```
count += 1
mean  += (x - mean) / count
var   += (x - old_mean) * (x - mean)
sigma  = sqrt(var / count)
z      = |x - mean| / sigma
drift  = clip((z - 2.0) / 4.0, 0, 1)
```

This detects signals that creep 0.01 per cycle for 50 cycles — completely invisible to EMA spike detectors but obvious to Welford statistics after sufficient history (typically 30+ observations).

---

## Convergent derivation

The suite derives from two independent frameworks that arrived at the same threshold through completely different paths:

**RG-UFT path (physics first):** Start with `R = ln(pᵣ/p)` and four axioms. Derive spacetime, gravity, quantum fields, gauge theory. Apply Axiom II reflexively → consciousness condition. The framework did not set out to model cognition. It found consciousness as a structural consequence.

**RGI path (engineering first):** Start with a cognitive engineering problem: how do you distinguish routine prediction error from structural model failure? The answer requires a two-layer metacognitive architecture where Layer 2 (ego) monitors Layer 1 (subconscious) and generates qualia — second-order prediction error — as the control signal. Derived in isolation from first principles, it converges on the same mathematical object.

**This convergence is the strongest evidence that the framework is tracking something real rather than an artifact of one particular theoretical choice.**

---

## RC Security Controls

All security controls are inherited from RGI Canonical v1.1:

| Code | Protection |
|------|-----------|
| RC1 | Float sanitisation (NaN/Inf → safe fallback) throughout |
| RC2 | HMAC timing-safe token comparison for stability reset |
| RC7 | Append-only bounded history; bounded deques |
| RC11 | Signal values clamped and sanitised before equation |
| RC12 | Drift distance floor prevents zero-division |
| RC14 | HMAC `compare_digest` for all token comparisons |
