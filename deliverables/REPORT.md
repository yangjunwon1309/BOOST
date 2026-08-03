# SAM3 Pascal Environment Report

Status: stopped at Gate 2 per the work script.

Last updated: 2026-07-28 18:12 KST.

## Phase 0 Host Precheck

Gate: passed.

- `logs/phase0_nvidia_smi_final.log`: PASS. Host-visible GPU is
  `NVIDIA TITAN Xp`; driver is `535.288.01`; CUDA runtime reported by the
  driver is `12.2`.
- `logs/phase0_docker_version_final.log`: PASS. Docker CLI is installed:
  `Docker version 29.6.2`.
- `logs/phase0_sg_docker_info_final.log`: PASS. Docker daemon is reachable via
  `sg docker`; NVIDIA runtime and CDI GPU devices are present.
- `logs/phase0_sg_docker_run_ubuntu_nvidia_smi_final.log`: PASS. A Docker
  container launched with `--gpus all` can see the `NVIDIA TITAN Xp`.

Note: direct `docker info` from this Codex process used an inherited group list
that did not include `docker`. The work script gate is satisfied using
`sg docker`, which enters the Docker group for each Docker command without
changing the host driver or CUDA toolkit.

## Phase 1 Base Image

Gate: passed.

- `logs/phase1_docker_build.log`: PASS. Built `sam3-pascal:latest` from
  `python:3.12-slim`.
- PyTorch install followed the script constraint:
  `torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu126`.
- No `nvidia/cuda:12.6.*` base image was used.
- No host CUDA toolkit or NVIDIA driver changes were made.

## Phase 2 sm_61 Smoke Test

Gate: failed.

- `logs/phase2_smoke_sm61.log`: FAIL by scripted gate criteria.
- `torch`: `2.8.0+cu126`.
- `cuda(build)`: `12.6`.
- `arch_list`: `['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']`.
- `capability`: `(6, 1)`.
- `device`: `NVIDIA TITAN Xp`.
- `bf16_supported`: `True`.
- `matmul`: PASS.
- `conv`: PASS.
- SDPA:
  - `SDPBackend.FLASH_ATTENTION`: FAIL (`RuntimeError`), expected on sm_61.
  - `SDPBackend.EFFICIENT_ATTENTION`: OK.
  - `SDPBackend.MATH`: OK.

The work script requires all Gate 2 conditions to pass. Two conditions failed:
`arch_list` does not include `sm_61`, and `bf16_supported` is not `False`.
Therefore SAM3 installation and benchmarking were not started.

## Phase 3 SAM3 Install And Pascal Patch

Not run because Gate 2 failed.

## Phase 4 SAM3 Benchmark

Not run because Gate 2 failed.

## Phase 5 Decision

SAM3 is not approved for this environment under the attached work script because
the required Phase 2 smoke gate failed before SAM3 installation.

## Phase 6 Lightweight SAM Validation

Not run. The script says a Phase 2 failure makes later work meaningless, so no
fallback model was installed or benchmarked in this pass.
