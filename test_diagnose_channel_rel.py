"""
test_diagnose_channel_rel.py
Chạy: python test_diagnose_channel_rel.py
"""
import torch
from channel import Channel
from fis_modules import FIS_SpatialPowerController

print('=' * 60)
print('TEST: diagnose_controller.py dung dung channel_rel')
print('=' * 60)

B, C, H, W = 4, 8, 8, 8
snr_db = 7.0
torch.manual_seed(42)

# Test 1: gamma_eff_norm vs channel_rel khac nhau cho fading
print('\n--- So sanh gamma_eff_norm vs channel_rel (Rayleigh) ---')
ch = Channel(channel_type='rayleigh', snr_db=snr_db)
ctx = ch.sample_context(B, torch.device('cpu'), torch.float32)
gen = ctx['gamma_eff_norm']
crel = ctx['channel_rel']
print(f'  gamma_eff_norm: mean={gen.mean():.4f}, std={gen.std():.4f}')
print(f'  channel_rel:    mean={crel.mean():.4f}, std={crel.std():.4f}')
diff = (gen - crel).abs().mean().item()
print(f'  |diff|: {diff:.6f}')
assert diff > 1e-6
print('[PASS 1] channel_rel != gamma_eff_norm cho Rayleigh')

# Test 2: AWGN thi bang nhau
print('\n--- AWGN: gamma_eff_norm == channel_rel ---')
ch_aw = Channel(channel_type='awgn', snr_db=snr_db)
ctx_aw = ch_aw.sample_context(B, torch.device('cpu'), torch.float32)
d_aw = (ctx_aw['gamma_eff_norm'] - ctx_aw['channel_rel']).abs().mean().item()
print(f'  |diff|: {d_aw:.6f}')
assert d_aw < 1e-6
print('[PASS 2] AWGN: channel_rel == gamma_eff_norm')

# Test 3: A khac nhau khi dung sai vs dung
print('\n--- A khi dung sai (gamma_eff_norm) vs dung (channel_rel) ---')
z = torch.randn(B, C, H, W)
ctrl = FIS_SpatialPowerController()
A_wrong, _ = ctrl(z, snr_db=snr_db, budget=1.0, mode='full',
    channel_rel=ctx['gamma_eff_norm'], return_info=True)
A_right, _ = ctrl(z, snr_db=snr_db, budget=1.0, mode='full',
    channel_rel=ctx['channel_rel'], return_info=True)
dA = (A_wrong - A_right).abs().mean().item()
print(f'  |A_wrong - A_right| mean: {dA:.6f}')
if dA > 1e-6:
    print('[PASS 3] A khac nhau -> dung sai channel_rel anh huong ket qua!')

# Test 4: Verify file
print('\n--- Verify file diagnose_controller.py ---')
with open('diagnose_controller.py', 'r') as f:
    for i, line in enumerate(f, 1):
        if 'channel_rel=channel_ctx' in line:
            print(f'  Dong {i}: {line.rstrip()}')
            assert "channel_ctx['channel_rel']" in line
            print('[PASS 4] File da dung channel_ctx["channel_rel"]')
            break

print('\n' + '=' * 60)
print('TAT CA 4 TEST PASS')
print('=' * 60)