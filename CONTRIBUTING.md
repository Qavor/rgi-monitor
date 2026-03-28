# Contributing

Thank you for your interest in contributing to the RGI Universal AI Monitoring Suite.

## Principles

Every contribution must respect the founding constraint: **every probe is a direct consequence of `R = log(p_new / p_old)` applied to a specific quantity.** New probes that introduce independent governing equations will not be accepted. The power of the suite is that everything derives from one equation.

## Setup

```bash
git clone https://github.com/denisq/rgi-monitor
cd rgi-monitor
pip install -e ".[dev]"
pytest tests/ -v
```

## Adding a probe

1. Implement your probe in `rgi_monitor/core.py` following the pattern of existing probes (dataclass for reading, class for monitor, `step()` method returns reading)
2. Export it from `rgi_monitor/__init__.py`
3. Add tests in `tests/test_suite.py` covering at least: normal operation, edge cases, RC safety
4. Add a usage example in `examples/individual_probes.py`
5. Document the governing equation derivation in `docs/architecture.md`

## RC controls

All numerical inputs must go through `_san()` before use in the governing equation (RC1). Any authentication or reset mechanism must use `hmac.compare_digest` (RC2/RC14). History structures must use bounded `deque` or equivalent (RC7).

## Tests

All 38 existing tests must continue to pass. New probes require minimum 5 new tests.

```bash
pytest tests/ -v --cov=rgi_monitor --cov-report=term-missing
```

## Pull request checklist

- [ ] All 38 existing tests pass
- [ ] New probe derives from `R = log(p_new / p_old)`
- [ ] RC1 float sanitisation applied to all inputs
- [ ] Dataclass reading type with `to_dict()` or `dataclasses.asdict()` compatible
- [ ] CLI updated if relevant
- [ ] `CHANGELOG.md` updated
