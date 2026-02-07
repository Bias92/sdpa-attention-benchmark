import time
import csv
import torch
from torch.nn.functional import scaled_dot_product_attention
import torch.cuda.nvtx as nvtx

# 1. 환경 설정 (PDF/스크린샷과 동일)
device = "cuda"
dtype = torch.float16 # Flash Attention 사용 시 필수

B = 1      # batch
H = 8      # heads
D = 64     # head dim

iters = 50
warmup = 10

S_list = [1024, 2048, 4096] # 실험할 시퀀스 길이
causal_list = [True]
modes = ["math", "flash"]   # 비교 대상

torch.manual_seed(0)

# 2. 백엔드 설정 함수 (스크린샷 로직 준수)
def set_backend(mode: str):
    if mode == "math":
        # Math만 켜고 나머지는 끈다
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    elif mode == "flash":
        # Flash만 켜고 나머지는 끈다
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def bench_once(S: int, causal: bool, mode: str):
    # 데이터 생성
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # ★ 핵심: PDF 방식대로 전역 설정 적용
    set_backend(mode)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    nvtx.range_push(f"WARMUP::{mode}")
    for _ in range(warmup):
        _ = scaled_dot_product_attention(q, k, v, is_causal=causal)
    nvtx.range_pop()

    torch.cuda.synchronize()

    # 시간 측정 (cuda.Event가 time.time()보다 GPU 측정엔 더 정확하므로 유지 권장)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    nvtx.range_push(f"RUN::{mode}::S{S}")
    starter.record()
    for _ in range(iters):
        _ = scaled_dot_product_attention(q, k, v, is_causal=causal)
    ender.record()
    nvtx.range_pop()

    torch.cuda.synchronize()
    
    total_ms = starter.elapsed_time(ender)
    avg_ms = total_ms / iters
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return avg_ms, peak_mb

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    
    out_csv = "results.csv"
    rows = []

    print(f"{'SeqLen':<8} {'Mode':<8} {'Latency (ms)':<15} {'VRAM (MB)':<10}")
    print("-" * 45)

    for S in S_list:
        for causal in causal_list:
            for mode in modes:
                try:
                    avg_ms, peak_mb = bench_once(S, causal, mode)
                    print(f"{S:<8} {mode:<8} {avg_ms:<15.3f} {peak_mb:<10.1f}")
                    rows.append([S, causal, mode, avg_ms, peak_mb])
                except RuntimeError as e:
                    print(f"{S:<8} {mode:<8} FAILED ({str(e)[:20]}...)")
                except ValueError as e:
                    print(f"Error setting backend: {e}")

    # CSV 저장
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["S", "causal", "mode", "latency_ms", "peak_vram_mb"])
        w.writerows(rows)
    print("\nBenchmark finished. Saved to results.csv")

if __name__ == "__main__":
    try:
        main()
    finally:
        # 종료 시 설정 원상복구 (선택 사항)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)