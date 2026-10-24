# Report implementation matrix

This is the current implementation boundary for the accelerator project. It
separates what is implemented and tested from what still requires an installed
physical-design environment or a larger ML workload.

| Report item | Status | Evidence |
| --- | --- | --- |
| Weight-stationary PE array with staggered activation wavefront | Implemented | `rtl/systolic_pe.sv`, `rtl/systolic_array.sv`, `make test` |
| Online checksum detection | Implemented | Extra checksum column stores row sums when weights load; RTL test injects permanent faults |
| Fault localisation | Implemented | All-ones diagnostic wavefront compares stored per-column signatures |
| Column-level spare repair | Implemented | One spare physical column, logical-to-physical remap, copy-and-retry path |
| Repair without corrupting logical outputs | Implemented | Icarus and cocotb regressions verify pre/post-repair outputs |
| Explicit operand isolation and accumulator clock enable | Implemented | Optimized wrapper and PE bypass path; lint and simulation cover zero-heavy activations |
| Detection and repair latency | Implemented | `build/rtl_metrics.json` records clean, detection, and repair cycles |
| Real workload and fault/accuracy sweep | Implemented as a reproducible experiment | `experiments/mnist_fault_experiment.py` supports official MNIST IDX files and a CI synthetic smoke test |
| Activity-annotated power input | Implemented as an artifact generator | `make activity` produces `build/activity.saif` from `build/systolic.vcd` |
| Sky130 synthesis, placement, routing, timing sign-off | Flow-ready, not run in this checkout | `openlane/config*.json`; run with OpenLane 2 and a Sky130 PDK |
| Area, timing, and power numbers | Not claimed yet | The PPA repository parses real reports and has a deterministic structural demo flow, but demo values are not silicon measurements |
| CNN-scale streaming benchmark | Not yet implemented | Current RTL transaction is a single 8×8 matrix-vector operation; the quantized experiment tiles a classifier around it |

The checksum is implemented as an extra **column**, which is the natural form
for a column-level repair design: each checksum-column weight is the sum of the
corresponding logical-column weights across rows. The default physical array is
therefore 8 logical columns + 1 checksum column + 1 spare column.
