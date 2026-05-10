"""
test_rule_consequents.py
Chạy: python test_rule_consequents.py
"""
import torch
from fis_modules import FIS_PowerAllocation

print('=' * 60)
print('TEST: Rule consequents sau khi fix')
print('=' * 60)

fis = FIS_PowerAllocation()
c = fis.c.data
print()
print('Rule consequents (khởi tạo, nn.Parameter):')
print(f'  r0 (high-I + low-rel):    {c[0].item():+.2f}')
print(f'  r1 (high-I + med-rel):    {c[1].item():+.2f}')
print(f'  r2 (high-I + high-rel):   {c[2].item():+.2f}')
print(f'  r3 (med-I  + low-rel):    {c[3].item():+.2f}')
print(f'  r4 (med-I  + med/high):   {c[4].item():+.2f}')
print(f'  r5 (low-I  + all):        {c[5].item():+.2f}')
print()

# --- Test 1: gradient giảm dần cho high-I ---
assert c[0] > c[1] > c[2], 'FAIL: high-I consequents không giảm dần!'
print('[PASS 1] Gradient giảm dần: r0(%.2f) > r1(%.2f) > r2(%.2f)' % (c[0].item(), c[1].item(), c[2].item()))

# --- Test 2: r5 âm ---
assert c[5] < 0, 'FAIL: r5 phải âm'
print('[PASS 2] Low-I consequent âm: r5(%.2f)' % c[5].item())

# --- Test 3: consequents là nn.Parameter ---
assert isinstance(fis.c, torch.nn.Parameter), 'FAIL: c phải là nn.Parameter'
print('[PASS 3] consequents là nn.Parameter (trainable)')

print()
print('--- Kiểm tra hành vi FIS với input mẫu ---')
torch.manual_seed(42)
B, H, W = 4, 8, 8
I = torch.rand(B, H, W)

# 3 kịch bản channel
A_low,  _, _, _ = fis(I, snr_db=1.0,  budget=1.0, channel_rel=torch.full((B,), 0.1), return_rules=True)
A_med,  _, _, _ = fis(I, snr_db=7.0,  budget=1.0, channel_rel=torch.full((B,), 0.5), return_rules=True)
A_high, _, _, _ = fis(I, snr_db=13.0, budget=1.0, channel_rel=torch.full((B,), 0.9), return_rules=True)

print(f'  A_mean (kênh tệ,  rel=0.1): {A_low.mean():.4f}  |  A_std: {A_low.std():.4f}')
print(f'  A_mean (kênh TB,  rel=0.5): {A_med.mean():.4f}  |  A_std: {A_med.std():.4f}')
print(f'  A_mean (kênh tốt, rel=0.9): {A_high.mean():.4f}  |  A_std: {A_high.std():.4f}')
print()

# --- Test 4: kênh tệ → dispersion cao hơn ---
assert A_low.std() > A_high.std(), 'FAIL: kênh tệ phải có A_std cao hơn'
print('[PASS 4] Kênh tệ → dispersion cao hơn (%.4f > %.4f)' % (A_low.std().item(), A_high.std().item()))

print()
print('TẤT CẢ TEST PASS')
print('=' * 60)