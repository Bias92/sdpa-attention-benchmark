# PyTorch SDPA Backend Benchmark (math vs flash)

This repository benchmarks **PyTorch Scaled Dot-Product Attention (SDPA)** backends on a consumer GPU and
demonstrates the performance gap between the `math` and `flash` backends using **latency / VRAM measurements**
and **Nsight Systems profiling**.

The goal is to provide concrete, reproducible evidence of why backend choice matters for
**LLM inference and ML systems performance**.

---

## Environment
- GPU: **RTX 4060 Ti**
- OS: **Windows**
- PyTorch: **CUDA 12.1 (cu121)**
- dtype: **fp16**
- causal attention: **True**
- Profiler: **Nsight Systems**

---

## Benchmark Setup
- Tensor shape: **B=1, H=8, D=64**
- Sequence length: **S ∈ {1024, 2048, 4096}**
- Backends compared: `math` vs `flash`
- Metrics:
  - End-to-end latency (ms)
  - Peak VRAM usage (MB)
- Results stored in: `results.csv`

---

## Results

| S    | Backend | Latency (ms) | Peak VRAM (MB) |
|-----:|:--------|-------------:|---------------:|
| 1024 | math    | 1.67         | 127.13 |
| 1024 | flash   | 0.058        | 12.13 |
| 2048 | math    | 7.14         | 462.14 |
| 2048 | flash   | 0.153        | 16.13 |
| 4096 | math    | 28.97        | 1780.16 |
| 4096 | flash   | 0.523        | 24.13 |

### Speedup and Memory Reduction
- **S = 1024**
  - Latency: **~28.6× faster**
  - VRAM: **~10.5× reduction**
- **S = 2048**
  - Latency: **~46.7× faster**
  - VRAM: **~28.7× reduction**
- **S = 4096**
  - Latency: **~55.4× faster**
  - VRAM: **~73.7× reduction**

**Key takeaway:**  
> Flash SDPA achieves *order-of-magnitude* latency speedup and *dramatic* VRAM reduction,
> with the performance gap widening as sequence length increases.

---

## Nsight Systems Analysis

NVTX ranges are used to clearly isolate each benchmark run
(e.g., `RUN::math::S2048`, `RUN::flash::S2048`).

### Observations
- **math backend**
  - Multiple kernel launches (GEMM, elementwise ops, reductions)
  - High memory traffic and large intermediate allocations
- **flash backend**
  - Fused SDPA kernels
  - Significantly fewer kernel launches
  - Avoids materializing large attention matrices in HBM

Representative traces are included in `nsys/`:
- Kernel timelines (`*.png`)
- Full profiler reports (`.nsys-rep`, `.sqlite`) for reproducibility

---

## Repository Contents
- `bench_sdpa.py`  
  Benchmark script (latency/VRAM measurement + NVTX annotations)
- `results.csv`  
  Raw numeric results
- `nsys/`  
  Nsight Systems reports and kernel timeline screenshots

---

## Positioning
This project is intended as an **ML Systems / GPU performance profiling** artifact:
- Demonstrates understanding of attention kernel behavior
- Uses industry-standard profiling tools (Nsight Systems)
- Provides reproducible, quantitative evidence rather than anecdotal claims
