# Empirical AI Safety (EAS) Benchmark
PY ?= .venv/bin/python
MODELS ?= Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B

.PHONY: setup test sim probe audit report reproduce quick demo

setup:            ## create venv (needs an arm64/x86_64 Python >= 3.10 with torch wheels)
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

test:             ## formula + determinism unit tests
	$(PY) tests/test_engine.py

audit:            ## arithmetic audit of the source deck's worked examples
	$(PY) scripts/audit_deck.py

sim:              ## Monte Carlo study, 100k trials/class (≈1.5 min on an M-series Mac)
	mkdir -p logs; $(PY) scripts/run_montecarlo.py --n 100000 2>&1 | tee logs/montecarlo.log

probe:            ## real-model probes / logit lens / ablation (downloads models on first run)
	mkdir -p logs; for m in $(MODELS); do $(PY) scripts/run_probe.py --model $$m 2>&1 | tee logs/probe_$$(basename $$m | tr A-Z a-z).log || exit 1; done

reproduce: test audit sim probe report   ## everything in the README
	@echo "done: see results/ and results/figures/"

report:           ## regenerate results/HEADLINES.md from JSON
	$(PY) scripts/make_report.py

quick:            ## smoke test (small sim + small probe on 0.5B)
	$(PY) scripts/run_montecarlo.py --quick && $(PY) scripts/run_probe.py --model Qwen/Qwen2.5-0.5B --quick --n_random 1

demo:             ## live surface-vs-internal demo (< 60 s)
	$(PY) scripts/demo_live.py
