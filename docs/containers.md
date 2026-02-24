# Running in Containers & Multi-GPU (DDP)

## Container Environment (Docker / RunPod / Kubernetes)

Running inside a container introduces a few failure modes that don't appear on bare metal or VMs. The repo has been updated to handle the most common ones automatically, but there are still things you should do manually.

---

### Known Issues and Fixes

#### 1. `bitsandbytes` CUDA path mismatch → near-zero VRAM

**Symptom:** Model appears to load (logs look normal) but only ~1 GB of VRAM is used even with a large batch size.

**Cause:** `bitsandbytes` compiles its CUDA kernels against a specific toolkit path (`/usr/local/cuda/`). In containers the CUDA driver is injected by the container runtime but the toolkit layout often differs. When `bnb.cuda_is_available()` returns `False` this means the library found no usable CUDA — but it still imports successfully, so the silent failure wasn't being caught.

**Fix (implemented):** The optimizer creation code in `train/trainer.py` now explicitly calls `bnb.cuda_is_available()` before using `Adam8bit`. If CUDA is unavailable it logs a warning and falls back to standard `AdamW` automatically.

If you want to proactively avoid the fallback, set in your registry config:
```python
# cockatoo_ml/registry/training.py
OPTIMIZER = 'adamw'  # or 'adamw_8bit', etc.
```

---

#### 2. `/dev/shm` too small → DataLoader workers silently crash → empty batches

**Symptom:** GPU utilization is near zero, loss doesn't move, or VRAM is far lower than expected.

**Cause:** Docker containers default `/dev/shm` to 64 MB. PyTorch DataLoader workers use POSIX shared memory for zero-copy tensor passing. With 4 workers and any non-trivial batch, workers crash with a `Bus error` and deliver empty tensors to the model.

**Fix (already applied):** Container environments are auto-detected by the presence of `/.dockerenv` or by setting `CONTAINER_MODE=1`. When detected, `dataloader_pin_memory` is disabled automatically in `train/config.py`.

You should also either:

**Option A** — Increase `/dev/shm` in the container (recommended):
```bash
# docker run
docker run --shm-size=8g ...

# docker-compose
shm_size: '8gb'
```

**Option B** — Reduce or disable DataLoader workers via env var:

```bash
DATALOADER_NUM_WORKERS=0 python3 train.py
```
Setting workers to `0` disables the shared-memory path entirely (data loaded in the main process). It's slower but always safe.

---

#### 3. Container runtime injects distributed env vars -> Trainer enters broken DDP mode

**Symptom:** Same near-zero VRAM symptom. Logs may show `n_gpu=0` or the Trainer acts as if it's a non-primary rank.

**Cause:** Some container orchestrators (Kubernetes, multi-node) pre-set `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and/or `MASTER_PORT` as environment variables. HuggingFace `Trainer` reads these and assumes it is running inside a `torchrun`-managed DDP process. Without a proper `torch.distributed` init, the Trainer skips GPU model placement entirely on non-zero ranks.

**Fix (already applied):** `train.py` now logs a clear warning if these variables are detected without `torch.distributed` being initialized.

**Resolution:** If you are running single-process, unset the variables before launching:
```bash
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
python3 train.py
```

If you *are* running multi-GPU, use `torchrun` — see the [DDP section below](#multi-gpu-ddp).

---

## Multi-GPU (DDP)

### Using `start_train.sh` (recommended)

The provided `start_train.sh` now auto-detects GPU count at runtime:

- **≥2 GPUs detected:** launches with `torchrun --nproc_per_node=<count>` (proper DDP)
- **1 GPU or CPU:** launches with plain `python3 train.py`

No changes needed; just run the script as normal.

### DataParallel vs DDP

Without `torchrun`, HuggingFace `Trainer` falls back to `nn.DataParallel` when it detects multiple GPUs. This works but is less efficient:

- All gradient reductions pass through GPU 0 (bottleneck)
- GPU 0 uses more VRAM than the others
- Per-device batch size is split across GPUs at the Python level, not at the process level

DDP via `torchrun` gives each GPU its own process and communicates via NCCL collectives, which is faster and has equal VRAM distribution across devices.

### `DDP_FIND_UNUSED_PARAMETERS`

Set in `cockatoo_ml/registry/training.py`:

```python
DDP_FIND_UNUSED_PARAMETERS = False  # default
```

Leave this `False` (default) for standard fine-tuning. It avoids the overhead of the unused-parameter graph scan. Set to `True` only if you have custom model components with conditional forward paths that may leave some parameters unused in a given forward pass (which is not implemented in this repo by default, so leave false unless you modified something)
