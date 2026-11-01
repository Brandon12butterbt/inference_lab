# InferenceLab — Claude Code Project Instructions

You are assisting with a **learning-focused, local AI inference performance lab**. The owner is a master's CS student specializing in AI and wants to understand every engineering decision. Prefer incremental, testable implementation and concise explanations over broad code generation.

## Read first

- `docs/BENCHMARK_PROTOCOL.md` — authoritative measurement rules
- `docs/SPEC.md` — scope, MVP features and acceptance criteria; tracked open decisions (§8)
- `docs/ARCHITECTURE.md` — module responsibilities, contracts, lifecycle and error codes
- `docs/TASKS.md` — milestone order and acceptance tests

**Precedence for benchmarking requirements:** `BENCHMARK_PROTOCOL.md` > `SPEC.md` > `ARCHITECTURE.md` > `TASKS.md`. This file governs how you work, not what the benchmark numbers are; it deliberately does not restate protocol parameters. If documents still conflict after applying the precedence, **stop, explain the conflict and request owner guidance**. Do not silently invent requirements.

## Current status

- **Documentation revision r2 is approved by the owner. Milestone M1 is authorized, but only incrementally by work package** (M1.1 Core Configuration and Error Handling is authorized; later packages need explicit owner approval). Only modify documentation after explicit approval. Do not start M2 or later milestones without clear owner authorization.
- Project root contains `pyproject.toml` and `uv.lock`; Python **3.12** via uv. `uv.lock` is the authority for installed dependency versions. Commands should be executed at project root.
- Windows 11; Ryzen 9 9900X3D; 32 GB RAM; RTX 5070 Ti (16 GB VRAM).
- Working NVIDIA CUDA PyTorch build **2.11.0+cu128**; GPU matrix multiplication has already been verified. Transformers is **5.22.1**.
- **Pandas 2.2.3 is deliberately selected**: a newer version was blocked by Windows Smart App Control. Do **not** upgrade Pandas, change CUDA/PyTorch package sources, alter NVIDIA drivers or disable Windows security without explicit approval.
- `src/inference_lab/` is the correct Python package location. Do not create `src/inferencelab/`.

## Operating rules

1. **Read before editing.** Start each requested task by confirming scope, expected files, acceptance criteria and tests.
2. **Make only approved changes.** Do not add features, install packages, upgrade dependencies, edit `pyproject.toml` or `uv.lock`, run destructive commands, commit or push to GitHub without explicit permission.
3. **Explain decisions as you go.** Give the owner a short 'why', key trade-offs, and how to verify the work. Ask when a material technical assumption is unresolved (see `docs/SPEC.md` §8).
4. **Keep backend independent from UI.** No Streamlit imports in the benchmark engine or worker. No timing/math/storage/comparison logic buried inside Streamlit widgets. Cross-run comparison belongs to `service.py` only.
5. **Never fabricate benchmark data.** Distinguish unit-test fixture values from real measured results; label all units and timing scope precisely.
6. **Do correct CUDA timing.** CUDA is asynchronous; follow `docs/BENCHMARK_PROTOCOL.md` exactly (synchronization, warm-ups, trials, batch sequencing, controlled threads/precision/attention settings, untimed parity check, separate configuration-level memory pass). Do not alter iteration counts, tolerances or thresholds without an owner-approved protocol revision.
7. **Reproducible experiments.** Record everything the protocol lists: model revision, fixture manifest and hashes, seed, device, precision and attention settings, batch size, hardware, power plan, relevant package versions, iteration counts and methodology identifier.
8. **No silent fallback.** Missing CUDA must not be labeled as GPU; failures must not be saved as successful experiments; use the error codes in `docs/ARCHITECTURE.md` §6 and the lifecycle states `running`/`completed`/`failed`/`interrupted`.
9. **Safety.** Do not commit secrets, model weights, databases, `.venv`, data dumps or user identifiers. Keep security features enabled.
10. **Reviewable diffs.** Implement one task or milestone at a time; summarize files changed and tests actually run. Ask before Git commits/pushes.
11. **Staged execution model.** M1–M2 run the engine in-process. The one-subprocess-per-configuration worker is introduced only in M3, runs sequentially, and uses the standard library only. Do not add a job queue, API server, cloud service or new dependency.

## Project commands (after code exists)

- `uv sync` — synchronize the locked project environment (avoid unapproved package upgrades).
- `uv run python --version` — confirm project interpreter.
- `uv run pytest` — run tests.
- `uv run ruff check .` and `uv run ruff format --check .` — lint and format verification.
- `uv run streamlit run app.py` — launch local UI **after `app.py` has been implemented (M4)**.

Before invoking potentially expensive/remote operations (model download, long benchmarks, installs), briefly describe them and obtain permission. Tests requiring GPU or model assets are isolated with markers and skip appropriately when CUDA or assets are unavailable. On the reference machine, GPU tests must be executed and passing — not skipped — to count as GPU acceptance; report skipped tests as skipped.

## Before sending a completed task response

State what changed, why, tests executed/results (including skips), what still needs validation, and the proposed next milestone. Do not claim any GPU speedup without actual benchmark measurements and conditions.
