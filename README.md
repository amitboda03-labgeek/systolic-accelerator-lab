# Systolic accelerator lab

This repository is the experiment, evidence, and demonstration layer for the
[self-repairing systolic accelerator](https://github.com/amitboda03-labgeek/self-repairing-systolic-accelerator).
It is intentionally separate from the synthesizable RTL and the physical
design sweep so that the workload, activity, and demo artifacts can evolve
without changing the hardware baseline.

## What is here

- `demo/` — a browser walkthrough of PE fault injection, ABFT detection,
  column localisation, spare-column repair, and degraded mode.
- `experiments/` — a NumPy-only quantized workload experiment. It applies four
  3×3 ReLU filters to MNIST-style 8×8 inputs, trains an int8 eight-class head,
  tiles the 144 features into eighteen 8×8 matrix operations, and sweeps PE
  faults.
- `tools/vcd_to_saif.py` — converts a representative RTL VCD into a SAIF
  switching-activity artifact for a real power run.
- `docs/` — an implementation matrix separating tested evidence from the
  physical-flow boundary.

## Quick start

Run the dependency-light checks and generate the deterministic smoke artifact:

```sh
make test
make experiment-smoke
```

Use official MNIST IDX files when network access is available:

```sh
python3 experiments/mnist_fault_experiment.py --download --limit 512
```

To convert an RTL waveform produced by the accelerator checkout:

```sh
make activity \
  ACTIVITY_VCD=../self-repairing-systolic-accelerator/build/systolic.vcd
```

`activity.saif` contains switching activity, not fabricated power numbers. A
real area/timing/power result still requires OpenLane 2 and the Sky130 PDK;
the physical-flow sweep lives in the separate
[`systolic-ppa-sweep`](https://github.com/amitboda03-labgeek/systolic-ppa-sweep)
repository.

## Browser demo

```sh
python3 -m http.server 8000 --directory demo
```

Open <http://localhost:8000>, choose a PE row, logical column, and fault mode,
then watch the checksum mismatch, diagnostic signature, spare-column copy, and
post-repair result.

## Scope

The lab repository does not pretend that a JavaScript walkthrough is a
gate-level simulator or that a structural PPA fixture is a Sky130 measurement.
The authoritative RTL, cocotb tests, OpenLane configuration, and actual sweep
entry point remain linked above; this repository makes the evidence easy to
run, inspect, and present.
