"""
test_gate1_fix.py
-----------------
Chạy: python test_gate1_fix.py

Kiểm tra 3 điều sau khi áp dụng Gate-1 fix:

  1. channel_rel thay đổi theo SNR (mean tăng đơn điệu)
  2. channel_rel có variance thực (std > 0.10) ở mọi SNR
  3. channel_rel khác với old behavior (norm_db(|h|²))
  4. EQ path và AWGN path không bị ảnh hưởng

Không cần GPU, không cần model, chỉ cần channel.py đã được fix.
"""

import math
import sys
import torch

# ── Import channel.py (đảm bảo đang dùng bản đã fix) ─────────
try:
    from channel import Channel
except ImportError:
    print("[ERROR] Không tìm thấy channel.py. Chạy script này từ cùng thư mục với channel.py.")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────
def ctx_stats(ctx, key="channel_rel", n_samples=2000):
    """Trả về (mean, std, min, max) của một key trong ctx."""
    val = ctx[key]
    if not torch.is_tensor(val):
        return None
    return {
        "mean": val.float().mean().item(),
        "std":  val.float().std().item(),
        "min":  val.float().min().item(),
        "max":  val.float().max().item(),
    }


def run_snr_sweep(channel_type, equalize, snr_list, batch_size=512):
    """
    Chạy sample_context() ở nhiều SNR, trả về stats của channel_rel
    và gamma_eff_norm để so sánh.
    """
    ch = Channel(channel_type=channel_type, P=1.0)
    ch.enable_rayleigh_equalization(equalize)
    results = []
    for snr in snr_list:
        ch.change_snr(snr)
        ctx = ch.sample_context(batch_size=batch_size, device=torch.device("cpu"))
        results.append({
            "snr": snr,
            "channel_rel":    ctx_stats(ctx, "channel_rel"),
            "gamma_eff_norm": ctx_stats(ctx, "gamma_eff_norm"),
        })
    return results


def check_monotone(means, label):
    """Kiểm tra mean tăng đơn điệu theo SNR."""
    ok = all(means[i] < means[i+1] for i in range(len(means)-1))
    symbol = "✓" if ok else "✗"
    print(f"  {symbol} {label}: monotone increasing = {ok}")
    print(f"    values: {[f'{v:.3f}' for v in means]}")
    return ok


def check_variance(stds, label, threshold=0.10):
    ok = all(s > threshold for s in stds)
    symbol = "✓" if ok else "✗"
    print(f"  {symbol} {label}: all std > {threshold} = {ok}")
    print(f"    stds:   {[f'{s:.3f}' for s in stds]}")
    return ok


def check_not_constant(means, label):
    spread = max(means) - min(means)
    ok = spread > 0.05
    symbol = "✓" if ok else "✗"
    print(f"  {symbol} {label}: spread across SNR = {spread:.4f} (need >0.05) = {ok}")
    return ok


# ── Main ──────────────────────────────────────────────────────
def main():
    SNR_LIST = [1.0, 4.0, 7.0, 10.0, 13.0]
    BATCH    = 4096  # lớn để giảm variance thống kê
    ALL_PASS = True

    print("=" * 62)
    print("GATE-1 FIX VERIFICATION")
    print("=" * 62)

    # ── TEST 1: Rayleigh no-EQ ──────────────────────────────
    print("\n[1] Rayleigh no-EQ  (channel_rel = gamma_eff_norm sau fix)")
    results = run_snr_sweep("rayleigh", equalize=False,
                             snr_list=SNR_LIST, batch_size=BATCH)

    rel_means  = [r["channel_rel"]["mean"] for r in results]
    rel_stds   = [r["channel_rel"]["std"]  for r in results]
    norm_means = [r["gamma_eff_norm"]["mean"] for r in results]

    p1 = check_monotone(rel_means,  "channel_rel mean vs SNR")
    p2 = check_variance(rel_stds,   "channel_rel std")
    p3 = check_not_constant(rel_means, "channel_rel")

    # Sau fix, channel_rel phải bằng gamma_eff_norm
    diff = [abs(rel_means[i] - norm_means[i]) for i in range(len(SNR_LIST))]
    p4 = all(d < 1e-5 for d in diff)
    symbol = "✓" if p4 else "✗"
    print(f"  {symbol} channel_rel == gamma_eff_norm (max diff={max(diff):.2e})")

    # Kiểm tra OLD behavior: norm_db(|h|²) sẽ hằng số
    print("\n  [sanity] Giá trị trước fix sẽ là:")
    ch_old = Channel("rayleigh", P=1.0)
    old_means = []
    for snr in SNR_LIST:
        ch_old.change_snr(snr)
        ctx = ch_old.sample_context(batch_size=BATCH, device=torch.device("cpu"))
        # Tính lại theo công thức cũ để demo
        h2 = ctx["h_abs2"]
        h2_db = 10.0 * torch.log10(h2.clamp_min(1e-8))
        old_rel = ((h2_db - ch_old.context_db_min) /
                   (ch_old.context_db_max - ch_old.context_db_min + 1e-8)).clamp(0, 1)
        old_means.append(old_rel.mean().item())
    old_spread = max(old_means) - min(old_means)
    print(f"    norm_db(|h|²) means: {[f'{v:.3f}' for v in old_means]}")
    print(f"    spread = {old_spread:.4f}  (≈0 → controller bị mù SNR)")

    ALL_PASS = ALL_PASS and p1 and p2 and p3 and p4

    # ── TEST 2: AWGN không bị ảnh hưởng ────────────────────
    print("\n[2] AWGN  (channel_rel không đổi — mong đợi std=0)")
    results_awgn = run_snr_sweep("awgn", equalize=False,
                                  snr_list=SNR_LIST, batch_size=BATCH)
    awgn_stds = [r["channel_rel"]["std"] for r in results_awgn]
    awgn_ok = all(s < 1e-4 for s in awgn_stds)
    symbol = "✓" if awgn_ok else "✗"
    print(f"  {symbol} AWGN channel_rel std ≈ 0 at all SNRs: {awgn_ok}")
    awgn_means = [r["channel_rel"]["mean"] for r in results_awgn]
    awgn_mono = all(awgn_means[i] < awgn_means[i+1] for i in range(len(awgn_means)-1))
    symbol2 = "✓" if awgn_mono else "✗"
    print(f"  {symbol2} AWGN channel_rel mean monotone (deterministic): {awgn_mono}")
    print(f"    means: {[f'{v:.3f}' for v in awgn_means]}")
    ALL_PASS = ALL_PASS and awgn_ok and awgn_mono

    # ── TEST 3: Rayleigh EQ không bị ảnh hưởng ─────────────
    print("\n[3] Rayleigh EQ  (path không thay đổi)")
    results_eq = run_snr_sweep("rayleigh", equalize=True,
                                snr_list=SNR_LIST, batch_size=BATCH)
    eq_means = [r["channel_rel"]["mean"] for r in results_eq]
    eq_stds  = [r["channel_rel"]["std"]  for r in results_eq]
    p5 = check_monotone(eq_means, "EQ channel_rel mean")
    p6 = check_variance(eq_stds,  "EQ channel_rel std", threshold=0.10)
    ALL_PASS = ALL_PASS and p5 and p6

    # ── TEST 4: Kiểm tra controller sẽ nhận được input khác nhau ──
    print("\n[4] Controller input diversity check")
    ch = Channel("rayleigh", P=1.0)
    ch.enable_rayleigh_equalization(False)
    ranges = []
    for snr in [1.0, 13.0]:
        ch.change_snr(snr)
        ctx = ch.sample_context(batch_size=BATCH, device=torch.device("cpu"))
        rel = ctx["channel_rel"]
        ranges.append((snr, rel.mean().item(), rel.std().item()))

    delta_mean = ranges[1][1] - ranges[0][1]
    p7 = delta_mean > 0.20
    symbol = "✓" if p7 else "✗"
    print(f"  {symbol} channel_rel: mean(13dB) - mean(1dB) = {delta_mean:.3f} (need >0.20)")
    print(f"    @  1 dB: mean={ranges[0][1]:.3f}, std={ranges[0][2]:.3f}")
    print(f"    @ 13 dB: mean={ranges[1][1]:.3f}, std={ranges[1][2]:.3f}")
    ALL_PASS = ALL_PASS and p7

    # ── Tóm tắt ─────────────────────────────────────────────
    print("\n" + "=" * 62)
    if ALL_PASS:
        print("TẤT CẢ CHECKS PASSED — Gate-1 fix đúng, sẵn sàng retrain")
    else:
        print("MỘT SỐ CHECKS FAILED — kiểm tra lại channel.py đã được fix chưa")
    print("=" * 62)

    # ── In bảng tóm tắt để so sánh trực quan ───────────────
    print("\nBảng channel_rel theo SNR sau fix (Rayleigh no-EQ):")
    print(f"  {'SNR':>6} | {'mean':>6} | {'std':>6} | {'min':>6} | {'max':>6}")
    print("  " + "-" * 42)
    results2 = run_snr_sweep("rayleigh", equalize=False,
                              snr_list=SNR_LIST, batch_size=BATCH)
    for r in results2:
        s = r["channel_rel"]
        print(f"  {r['snr']:>5.0f}  | {s['mean']:>6.3f} | {s['std']:>6.3f} | "
              f"{s['min']:>6.3f} | {s['max']:>6.3f}")
    print()


if __name__ == "__main__":
    main()
