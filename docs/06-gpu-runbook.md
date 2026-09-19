# GPU runbook

Last reviewed: 2026-09-18

The simulation is memory-bandwidth-bound, so speed tracks GPU memory bandwidth
almost linearly. On the Mac we measured ~18 GB/s effective against a roofline of
maybe 60-100; an A10G does ~600 GB/s.

| Hardware | Bandwidth | Per window | 30k-window arm |
|---|---|---|---|
| M3, 8 cores | ~18 GB/s achieved | 176 ms | ~95 min |
| A10G (g5.xlarge) | ~600 GB/s | expect 5-15 ms | expect 3-8 min |

## Instance choice

**g5.xlarge**, and nothing larger. Every size from g5.xlarge to g5.16xlarge has
exactly one A10G; bigger sizes add vCPUs and RAM we do not use. Only the
12xlarge (4 GPUs) and 48xlarge (8 GPUs) add hardware, which would need code
changes.

Do not use g6: the L4 has roughly half the memory bandwidth of the A10G, which
for this workload means roughly half the speed.

- AMI: `ami-0ea3ebd89344bea98` — Deep Learning OSS Nvidia Driver AMI GPU
  PyTorch 2.13 (Ubuntu 26.04), eu-west-2
- Storage: 60 GiB gp3 root. The 250 GB instance-store NVMe comes free and is
  wiped on termination; fine for scratch.
- Network: SSH from your IP only.
- **Advanced details -> Shutdown behaviour: Terminate.** With
  `shutdown -h now` at the end of a run, the box disposes of itself.
- Cost: $1.277/hr on-demand in London, so roughly $0.45 for a full sweep.

## Quota

GPU quota is zero on new *and* long-standing accounts until requested, and it is
per-region. Both were requested for eu-west-2 on 2026-09-18:

| Quota code | Name | Requested |
|---|---|---|
| `L-DB2E81BA` | Running On-Demand G and VT instances | 8 vCPU |
| `L-3819A6DF` | All G and VT Spot Instance Requests | 8 vCPU |

Check status:

```bash
AWS_PROFILE=your-profile aws service-quotas get-service-quota \
  --region eu-west-2 --service-code ec2 --quota-code L-DB2E81BA \
  --query 'Quota.Value' --output text
```

## Bring-up

```bash
# 1. code (private repo: token, gh auth, or just scp the tree — it is under 100 KB)
git clone https://github.com/KrystianDud/fly-trader.git fruit_fly && cd fruit_fly

# 2. market data from the laptop (22 MB; the connectome downloads itself)
scp -i KEY.pem data/market/usdjpy_1m.parquet ubuntu@HOST:~/fruit_fly/data/market/

# 3. drivers check, uv, CUDA torch, connectome from Google Cloud Storage
bash scripts/gpu_setup.sh

# 4. benchmark before committing to a long run
PYTHONPATH=. uv run python scripts/extract.py \
  --arm real --windows 256 --batch 256 --device cuda
```

## Batch size

State is roughly 17 tensors of `neurons x batch x 4 bytes`. At 165,836 neurons:

| Batch | VRAM | Fits 24 GB? |
|---|---|---|
| 256 | ~2.9 GB | yes, comfortably |
| 512 | ~5.8 GB | yes |
| 1024 | ~11.5 GB | yes, leave headroom for cuSPARSE workspace |

Start at 256 and raise it while throughput improves.

## Full sweep on GPU

```bash
for arm in real rewire0.1 rewire0.25 rewire0.5 rewire1.0 random; do
  PYTHONPATH=. uv run python scripts/extract.py \
    --arm "$arm" --windows 0 --batch 512 --device cuda --seed 1
done
sudo shutdown -h now     # with terminate-on-shutdown set, the box disposes of itself
```

`--windows 0` means the full history: 92,030 windows, 3.7 years.

Copy the activations back before shutdown — they are the expensive artefact and
everything downstream reads them:

```bash
scp -i KEY.pem -r ubuntu@HOST:~/fruit_fly/data/activations ./data/
```

## Gotchas

- Plain Amazon Linux or Ubuntu AMIs have no NVIDIA driver. Use a Deep Learning
  AMI or lose an hour.
- PyTorch has no sparse support on Apple Metal, which is why the Mac runs on CPU
  and why `--device mps` is not an option.
- Index tensors must live on the same device as what they index; `lif.py` moves
  them, but any new indexing code needs the same care.
- Forgetting to terminate costs about $30/day. Terminate-on-shutdown plus a
  billing alarm is the cheap insurance.
