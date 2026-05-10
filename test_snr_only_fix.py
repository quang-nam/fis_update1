"""
test_snr_only_fix.py
Chạy: python test_snr_only_fix.py
"""
import torch
from fis_modules import FIS_SpatialPowerController

print('=' * 60)
print('TEST: snr_only mode sau khi fix')
print('=' * 60)

ctrl = FIS_SpatialPowerController()
B, C, H, W = 4, 8, 8, 8
z = torch.randn(B, C, H, W)
torch.manual_seed(42)

# Test 1: snr_only chạy qua FIS pipeline thật
print('\n--- Test 1: snr_only đi qua FIS Layer-2 ---')
A_snr, info_snr = ctrl(z, snr_db=7.0, budget=1.0, mode='snr_only',
    channel_rel=torch.full((B,), 0.5), return_info=True)
assert 'rule2_strength' in info_snr
assert 'score_map' in info_snr
assert 'delta_map' in info_snr
print('[PASS 1] snr_only chạy qua FIS Layer-2 hoàn chỉnh')
print('  Keys:', sorted(k for k in info_snr.keys() if k != 'I'))

# Test 2: I=0.5 + channel_rel=scalar → A≈1 (đúng toán)
print('\n--- Test 2: I=const + channel_rel=scalar → A≈1 ---')
print(f'  A_mean: {A_snr.mean():.6f}')
print(f'  A_std:  {A_snr.std():.6f}')
assert abs(A_snr.mean().item() - 1.0) < 0.01
print('[PASS 2] A_mean ≈ 1.000')

# Test 3: snr_only vs full
print('\n--- Test 3: snr_only (I=const) vs full (I=from content) ---')
A_full, info_full = ctrl(z, snr_db=7.0, budget=1.0, mode='full',
    channel_rel=torch.full((B,), 0.5), return_info=True)
print(f'  snr_only A_std: {A_snr.std():.6f}')
print(f'  full     A_std: {A_full.std():.6f}')
assert A_full.std() > A_snr.std() * 10
print('[PASS 3] full A_std (%.4f) >> snr_only A_std (%.4f)' % (
    A_full.std().item(), A_snr.std().item()))

# Test 4: snr_only với 3 mức kênh → A vẫn ≈1
print('\n--- Test 4: snr_only với 3 mức kênh ---')
for rel_val, snr in [(0.1, 1.0), (0.5, 7.0), (0.9, 13.0)]:
    A_test, _ = ctrl(z, snr_db=snr, budget=1.0, mode='snr_only',
        channel_rel=torch.full((B,), rel_val), return_info=True)
    print(f'  rel={rel_val:.1f}, SNR={snr:4.1f}dB → A_mean={A_test.mean():.4f}, A_std={A_test.std():.6f}')
    assert abs(A_test.mean().item() - 1.0) < 0.02
print('[PASS 4] snr_only luôn ≈ baseline cho mọi kênh')

# Test 5: return_info=False
print('\n--- Test 5: return_info=False ---')
A_no_info = ctrl(z, snr_db=7.0, budget=1.0, mode='snr_only',
    channel_rel=torch.full((B,), 0.5), return_info=False)
assert A_no_info.shape == (B, H, W)
print('[PASS 5] return_info=False: shape', A_no_info.shape)

# Test 6: Rule activation (I=0.5 → medium rules active)
print('\n--- Test 6: Rule activation cho snr_only ---')
rs = info_snr['rule2_strength']
rs_avg = rs.mean(dim=(0, 2, 3))
rs_avg = rs_avg / rs_avg.sum()
print('  Rule activation:')
labels = ['iH+cL', 'iH+cM', 'iH+cH', 'iM+cL', 'iM+cMH', 'iL']
for i, lbl in enumerate(labels):
    print(f'    r{i} ({lbl:8s}): {rs_avg[i].item():.4f}')
assert rs_avg[3].item() > rs_avg[0].item()
print('[PASS 6] I=0.5 → medium-I rules active mạnh nhất')

print('\n' + '=' * 60)
print('TẤT CẢ 6 TEST PASS')
print('=' * 60)