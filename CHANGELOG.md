# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2025

### Added
- **ConsciousnessMonitor** — RG-UFT consciousness condition `R*(i,t) > C(B*ᵢ)` with four topological conditions (T1–T4)
- **TruthMonitor** — pre-output hallucination risk gate, AIGSA Article 5 formula, PASS/FLAG/VETO verdicts
- **StabilityMonitor** — three independent tripwires (trajectory × turbulence × coherence), monotonic OK → ALERT → QUARANTINE → SHUTDOWN, HMAC-gated reset
- **TrustMonitor** — per-entity reputation, `R = log(integrity/claim)`, drift-based spoofing detection
- **AlignmentMonitor** — six-channel conversational resonance, BASELINE/ENGAGED/CREATIVE/RESONANT tiers
- **IdentityMonitor** — structural identity continuity, deformation rate, trajectory signature
- **DriftMonitor** — Welford online statistics for slow behavioural creep
- **UniversalMonitor** — all 7 probes unified, single `.step()`, alert callbacks, JSON export
- **LLMAdapter** — logprob delta, perplexity, loss delta, attention entropy, OpenAI API response parsing
- **CLI** — `rgi-monitor selftest`, `rgi-monitor demo`, `rgi-monitor probe`
- **38/38 self-tests** — zero external dependencies
- **RC security controls** — RC1 float sanitisation, RC2/RC14 HMAC timing-safe comparison, RC7 bounded append-only history
