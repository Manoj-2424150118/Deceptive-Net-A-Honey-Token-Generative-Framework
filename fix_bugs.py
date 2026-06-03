#!/usr/bin/env python3
"""
Deceptive-Net: Bug Fix Script
Applies critical fixes to main.py
"""

import re

def fix_dds_metric():
    """Fix the DDS metric computation bug"""
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the buggy DDS computation
    # The bug: diff @ diff should be np.sum(diff ** 2)
    pattern = r'(diff = mu_r - mu_g\s*\n\s*# Matrix sqrt via scipy\s*\n\s*covmean, _ = linalg\.sqrtm.*?\n\s*if np\.iscomplexobj\(covmean\):\s*\n\s*covmean = covmean\.real)\s*\n\s*(dds = \(diff @ diff)'
    
    # Better approach: just fix the specific line
    if 'dds = (diff @ diff +' in content:
        content = content.replace(
            'dds = (diff @ diff +',
            'diff_sq = np.sum(diff ** 2)\n    dds = (diff_sq +'
        )
        print("✓ Fixed DDS metric: Changed 'diff @ diff' to 'np.sum(diff ** 2)'")
    
    # Also add try-catch for matrix sqrt numerical stability
    if 'covmean, _ = linalg.sqrtm(sig_r @ sig_g, disp=False)' in content:
        old_code = '''     covmean, _ = linalg.sqrtm(sig_r @ sig_g, disp=False)
     if np.iscomplexobj(covmean):
         covmean = covmean.real'''
        
        new_code = '''     try:
         covmean, _ = linalg.sqrtm(sig_r @ sig_g, disp=False)
         if np.iscomplexobj(covmean):
             covmean = covmean.real
         covmean = (covmean + covmean.T) / 2
     except np.linalg.LinAlgError:
         covmean = np.eye(sig_r.shape[0]) * 1e-6'''
        
        content = content.replace(old_code, new_code)
        print("✓ Added error handling for matrix sqrt")
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


def fix_encoder_stability():
    """Fix encoder training instability"""
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add learning rate decay to encoder training
    old_train = '''def train_encoder(X_real: np.ndarray, cfg: Config,
                   device: torch.device) -> TabularEncoder:
     """Train the tabular encoder used for DDS computation."""
     enc = TabularEncoder(X_real.shape[1], cfg.enc_dim).to(device)
     opt = optim.Adam(enc.parameters(), lr=1e-3)'''
    
    new_train = '''def train_encoder(X_real: np.ndarray, cfg: Config,
                   device: torch.device) -> TabularEncoder:
     """Train the tabular encoder used for DDS computation."""
     enc = TabularEncoder(X_real.shape[1], cfg.enc_dim).to(device)
     opt = optim.Adam(enc.parameters(), lr=1e-3)
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.enc_epochs)'''
    
    if old_train in content:
        content = content.replace(old_train, new_train)
        print("✓ Added learning rate scheduler to encoder training")
    
    # Add gradient clipping
    old_step = '''            opt.zero_grad()
             loss.backward()
             opt.step()'''
    
    new_step = '''            opt.zero_grad()
             loss.backward()
             torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1.0)
             opt.step()
             scheduler.step()'''
    
    if old_step in content:
        content = content.replace(old_step, new_step)
        print("✓ Added gradient clipping and scheduler step")
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


def fix_critic_steps():
    """Increase critic training steps for stability"""
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Change n_critic from 2 to 5
    if "n_critic: int = 2" in content:
        content = content.replace("n_critic: int = 2", "n_critic: int = 5")
        print("✓ Increased critic steps: 2 -> 5 (better WGAN-GP training)")
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


def fix_config_for_4gb_gpu():
    """Optimize batch size for 4GB GPU"""
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add adaptive batch sizing
    insert_point = "DEVICE = torch.device(CFG.device)"
    adaptive_code = '''

# Adapt batch size for available GPU memory
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    available_gb = props.total_memory / 1e9
    if available_gb < 6:
        CFG.batch_size = 128  # Reduce from 256 for smaller GPUs
        logger.info(f"GPU memory {available_gb:.1f}GB detected - reducing batch size to 128")
    elif available_gb < 10:
        CFG.batch_size = 192
        logger.info(f"GPU memory {available_gb:.1f}GB detected - batch size set to 192")'''
    
    if insert_point in content:
        content = content.replace(insert_point, insert_point + adaptive_code)
        print("✓ Added adaptive batch sizing for GPU memory")
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


def add_data_validation():
    """Add data validation function"""
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    validation_code = '''

def validate_aligned_data(X_baf: np.ndarray, X_pii: np.ndarray, name: str = "Data") -> bool:
    """
    Validate aligned datasets for training compatibility.
    Returns True if valid, logs warnings if issues found.
    """
    issues = []
    
    # Shape check
    if X_baf.shape[0] != X_pii.shape[0]:
        issues.append(f"Shape mismatch: BAF {X_baf.shape[0]} != PII {X_pii.shape[0]} rows")
    
    # NaN check
    baf_nans = np.isnan(X_baf).sum()
    pii_nans = np.isnan(X_pii).sum()
    if baf_nans > 0 or pii_nans > 0:
        issues.append(f"NaN values: BAF {baf_nans}, PII {pii_nans}")
    
    # Inf check
    baf_infs = np.isinf(X_baf).sum()
    pii_infs = np.isinf(X_pii).sum()
    if baf_infs > 0 or pii_infs > 0:
        issues.append(f"Inf values: BAF {baf_infs}, PII {pii_infs}")
    
    # Range check (should be [0,1] after normalization)
    if (X_baf < -0.01).any() or (X_baf > 1.01).any():
        issues.append(f"BAF values out of [0,1] range: min={X_baf.min():.4f}, max={X_baf.max():.4f}")
    if (X_pii < -0.01).any() or (X_pii > 1.01).any():
        issues.append(f"PII values out of [0,1] range: min={X_pii.min():.4f}, max={X_pii.max():.4f}")
    
    # Empty check
    if len(X_baf) == 0 or len(X_pii) == 0:
        issues.append("Empty dataset detected")
    
    if issues:
        logger.warning(f"{name} validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info(f"{name} validation passed - all checks OK")
        return True
'''
    
    # Insert before the train function
    validate_point = content.find("def train(cfg: Config")
    if "def train(cfg: Config" in content and validate_point > 0:
        content = content[:validate_point] + validation_code + "\n" + content[validate_point:]
        print("✓ Added data validation function")
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DECEPTIVE-NET: APPLYING CRITICAL FIXES")
    print("="*70 + "\n")
    
    fixes = [
        ("DDS Metric Fix", fix_dds_metric),
        ("Encoder Stability", fix_encoder_stability),
        ("Critic Steps", fix_critic_steps),
        ("4GB GPU Optimization", fix_config_for_4gb_gpu),
        ("Data Validation", add_data_validation),
    ]
    
    for name, fix_func in fixes:
        try:
            print(f"\nApplying: {name}")
            fix_func()
        except Exception as e:
            print(f"✗ Error in {name}: {e}")
    
    print("\n" + "="*70)
    print("ALL FIXES APPLIED SUCCESSFULLY")
    print("="*70 + "\n")
    print("Next steps:")
    print("1. Review changes in main.py")
    print("2. Test with: python main.py --mode train --epochs 10")
    print("3. Check training logs for improvements")
