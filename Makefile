.PHONY: test experiment-smoke activity demo

PYTHON ?= python3
ACTIVITY_VCD ?= ../self-repairing-systolic-accelerator/build/systolic.vcd
ACTIVITY_SAIF ?= build/activity.saif

test:
	$(PYTHON) -m py_compile experiments/mnist_fault_experiment.py tools/vcd_to_saif.py

experiment-smoke:
	$(PYTHON) experiments/mnist_fault_experiment.py \
	  --synthetic --limit 128 --output build/fault-experiment

activity:
	@test -f "$(ACTIVITY_VCD)" || (echo "VCD not found: $(ACTIVITY_VCD)" >&2; exit 2)
	$(PYTHON) tools/vcd_to_saif.py "$(ACTIVITY_VCD)" "$(ACTIVITY_SAIF)"

demo:
	$(PYTHON) -m http.server 8000 --directory demo
