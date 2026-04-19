# VBD per-color particle-body contact: experiment results

- **Branch:** `perf/vbd-per-color-contact-experiment`
- **Date:** 2026-04-19
- **Scene:** `newton.examples cloth_franka` on NVIDIA RTX 4090, Warp 1.12.1, CUDA 12.4

## What the change does

VBD parallelizes the cloth solve by graph-coloring particles: within one color,
particles share no force term, so their updates run in parallel. Each outer
iteration loops over colors and launches one kernel per (color, force-type) pair.

For particle–rigid (soft) contacts, the baseline kernel
`accumulate_particle_body_contact_force_and_hessian` is launched per color with
`dim = contacts.soft_contact_max` (= `particle_count × shape_count`, i.e.
**360 416** in this scene). Each thread reads the contact's particle, looks up
`particle_colors[pidx]`, and returns if it doesn't match the current color.

This patch replaces the per-color color-filter launches with a per-color compact
index list rebuilt once per step, then launches `accumulate_..._by_color` with
`dim = per-color capacity` (summing to `soft_contact_max`). Intent was to cut
accumulator thread-launches by `num_colors` (~5) and give each accumulator thread
coalesced loads from a compact buffer.

Changes:
- `newton/_src/solvers/vbd/particle_vbd_kernels.py` — new
  `fill_particle_body_contacts_per_color` and
  `accumulate_particle_body_contact_force_and_hessian_by_color` kernels.
- `newton/_src/solvers/vbd/solver_vbd.py` — per-color capacity/offset/count/index
  buffers, a single fill-kernel launch each step, and per-color accumulator
  launches taking capacity as a scalar.

## Result: net regression

### FPS (warm Warp cache, no nsys, 3 × 3-second benchmarks)

|                   | run 1 | run 2 | run 3 | mean        |
| ----------------- | ----- | ----- | ----- | ----------- |
| baseline (main)   | 61.1  | 60.7  | 61.5  | **61.1 FPS** |
| patched (cleaned) | 58.4  | 59.2  | 59.2  | **58.9 FPS** |

Δ = **−3.6 %**.

### GPU-kernel breakdown (graph-node nsys, ~170 frames)

| kernel                                             | baseline         | patched                  | Δ                 |
| -------------------------------------------------- | ---------------- | ------------------------ | ----------------- |
| `accumulate_particle_body_contact…` avg per call   | 5025 ns          | 7375 ns (`…_by_color`)   | **+47 % per call** |
| same kernel, total GPU time                        | 226.1 ms (8.3 %) | 320.8 ms (11.6 %)        | +94.7 ms          |
| `fill_particle_body_contacts_per_color`            | —                | 6.1 ms (0.2 %)           | +6 ms             |
| **total particle-body contact path**               | **226.1 ms**     | **326.9 ms**             | **+45 %**         |
| `accumulate_self_contact_force_and_hessian`        | 19.6 %           | 18.6 %                   | neutral           |
| `solve_elasticity_tile`                            | 14.3 %           | 13.8 %                   | neutral           |
| `apply_planar_truncation_parallel_by_collision`    | 12.4 %           | 12.4 %                   | neutral           |

## Why it regresses

The "wasted" threads in the baseline kernel aren't really waste — they're filler
that lets one big launch saturate the 4090's SMs. Per-thread work is tiny
(two loads + compare + return), and it runs in parallel with the minority of
threads doing real contact accumulation.

- Baseline launch: `dim = 360 416` threads per color — easily saturates ~260 k
  concurrent-thread GPU capacity.
- Patched launch: `dim ≈ 72 k` threads per color (that color's capacity) —
  under-occupies the SMs, and we now do 5 such launches instead of 5 big ones
  plus an extra 360 k-thread fill pass per step.

Per-call kernel time rose 47 % because the smaller launch achieves worse SM
occupancy. Launching on the *live* per-color count isn't an option — it lives
on device, so reading it as the launch dim would force a D→H sync every step
and break CUDA-graph capture.

The experiment was partly motivated by an earlier baseline reading that put the
particle-body contact kernel at ~19 % of GPU time. The graph-node capture shows
the true share is 8.3 %, comfortably below several other cloth kernels.

## Next targets (bigger VBD/cloth hotspots in this scene)

| kernel                                            | share  |
| ------------------------------------------------- | ------ |
| `accumulate_self_contact_force_and_hessian`       | 19.6 % |
| `edge_colliding_edges_detection_kernel`           | 14.7 % |
| `solve_elasticity_tile`                           | 14.3 % |
| `apply_planar_truncation_parallel_by_collision`   | 12.4 % |

## Artifacts in this commit

- `work/cloth_franka_baseline_graphnode.nsys-rep` — main, `--cuda-graph-trace=node`
- `work/cloth_franka_patched_graphnode.nsys-rep` — same flags, per-color patch applied

## Reproduce

```bash
nsys profile -o work/cloth_franka_<label>_graphnode \
  --force-overwrite true --trace=cuda,nvtx,osrt --cuda-graph-trace=node --stats=false \
  uv run -m newton.examples cloth_franka \
    --device cuda:0 --viewer null --benchmark 3 --num-frames 200
```

## Recommendation

**Do not merge.** Keep the branch as a negative-result record and pick the next
target from the list above.
