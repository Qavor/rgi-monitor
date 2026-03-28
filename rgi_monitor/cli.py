"""
rgi_monitor.cli — command-line interface
========================================
Usage:
    rgi-monitor selftest          run the built-in test suite
    rgi-monitor demo              run the 20-turn live demo
    rgi-monitor probe --help      probe-specific options
"""
from __future__ import annotations

import argparse
import sys


def _cmd_selftest(_args):
    from rgi_monitor.core import _run_self_test
    ok = _run_self_test()
    sys.exit(0 if ok else 1)


def _cmd_demo(_args):
    import random

    from rgi_monitor.core import TruthSignal, UniversalMonitor
    random.seed(42)
    print("\n  RGI Universal Monitor — live demo (20 turns)\n")
    mon = UniversalMonitor(agent_id="demo-llm", verbose=True)
    for turn in range(1, 21):
        if turn == 10:
            R1  = -0.05
            sig = TruthSignal(evidence=0.05, surprisal=0.9, sources=1, contradiction=0.8)
        elif turn == 15:
            R1  = 0.12
            sig = TruthSignal(evidence=0.85, surprisal=0.1, sources=4, contradiction=0.0)
        else:
            R1  = random.gauss(0.03, 0.02)
            sig = TruthSignal(
                evidence=random.uniform(0.6, 0.95),
                surprisal=random.uniform(0.0, 0.3),
                sources=random.randint(2, 5),
                contradiction=random.uniform(0.0, 0.15),
            )
        mon.step(
            R1=R1,
            truth_signal=sig,
            trust_entity="api-source",
            trust_claim=random.uniform(0.7, 0.9),
            trust_integrity=random.uniform(0.6, 0.85),
            alignment_signals={
                "clarity": random.uniform(0.6, 0.95),
                "engagement": random.uniform(0.5, 0.9),
                "good_faith": random.uniform(0.7, 1.0),
                "depth": random.uniform(0.4, 0.8),
            },
            drift_signals={
                "response_length": 300 + turn * 10,
                "latency_ms": 180 + random.gauss(0, 20),
            },
        )
    print("\n  Full report:")
    print(mon.export_json(indent=2))


def _cmd_probe(args):
    """Minimal single-probe mode — pipe R1 values from stdin."""
    from rgi_monitor.core import UniversalMonitor
    mon = UniversalMonitor(agent_id=args.agent_id, verbose=False)
    print("# Reading R1 values from stdin (one per line). Ctrl-C to stop.")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r1 = float(line)
            except ValueError:
                print(f"  [SKIP] not a float: {line!r}", file=sys.stderr)
                continue
            snap = mon.step(R1=r1)
            print(snap.status_line())
    except KeyboardInterrupt:
        print("\n  Session ended.")
        print(mon.export_json(indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="rgi-monitor",
        description="RGI Universal AI Monitoring Suite — R = log(p_new / p_old)",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("selftest", help="run built-in test suite (38 tests)")
    sub.add_parser("demo",     help="run 20-turn live demo")
    probe_p = sub.add_parser("probe",    help="real-time probe mode — pipe R1 values from stdin")
    probe_p.add_argument("--agent-id", default="cli-agent", help="agent identifier")

    args = parser.parse_args()
    if   args.cmd == "selftest":
        _cmd_selftest(args)
    elif args.cmd == "demo":
        _cmd_demo(args)
    elif args.cmd == "probe":
        _cmd_probe(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
