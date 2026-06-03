#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Deceptive-Net: Data Setup & Validation Script
================================================================================

This script helps set up the real fraud detection datasets or validates
synthetic data fallback.

USAGE:
    python setup_datasets.py --download    # Instructions for downloading
    python setup_datasets.py --validate    # Validate existing data
    python setup_datasets.py --generate    # Generate synthetic data
    python setup_datasets.py --status      # Check what data is available

================================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("./data")
BAF_CSV = DATA_DIR / "Base.csv"
PII_CSV = DATA_DIR / "pii_dataset.csv"

# Expected schemas
BAF_EXPECTED_COLS = 30  # Approximately 30 columns in NeurIPS BAF
PII_EXPECTED_COLS = 15  # Approximately 15 columns in Mendeley PII


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {DATA_DIR.absolute()}")


def check_data_status():
    """Check what data files are available."""
    logger.info("\n" + "=" * 70)
    logger.info("DATA STATUS CHECK")
    logger.info("=" * 70)
    
    baf_exists = BAF_CSV.exists()
    pii_exists = PII_CSV.exists()
    
    logger.info(f"BAF Dataset (Base.csv): {'[FOUND]' if baf_exists else '[MISSING]'}")
    if baf_exists:
        df = pd.read_csv(BAF_CSV, nrows=1)
        logger.info(f"  - Shape: {pd.read_csv(BAF_CSV).shape}")
        logger.info(f"  - Columns: {list(df.columns)[:5]}...")
    
    logger.info(f"\nPII Dataset (pii_dataset.csv): {'[FOUND]' if pii_exists else '[MISSING]'}")
    if pii_exists:
        df = pd.read_csv(PII_CSV, nrows=1)
        logger.info(f"  - Shape: {pd.read_csv(PII_CSV).shape}")
        logger.info(f"  - Columns: {list(df.columns)[:5]}...")
    
    logger.info("\n" + "=" * 70)
    if baf_exists and pii_exists:
        logger.info("Status: READY FOR TRAINING [Both datasets available]")
    elif baf_exists or pii_exists:
        logger.info("Status: PARTIAL [One dataset missing]")
    else:
        logger.info("Status: USING SYNTHETIC FALLBACK [No real datasets found]")
    logger.info("=" * 70 + "\n")


def show_download_instructions():
    """Show instructions for downloading datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("HOW TO DOWNLOAD REAL DATASETS")
    logger.info("=" * 70)
    
    logger.info("""
1. NEURIPS 2022 BANK ACCOUNT FRAUD DATASET
   ────────────────────────────────────────
   Source: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
   
   Steps:
   a) Create a Kaggle account (free): https://www.kaggle.com
   b) Download the dataset (CSV files)
   c) Extract and place 'Base.csv' in: ./data/
   
   Expected columns (~30):
   - fraud_bool, income, customer_age, employment_status, housing_status
   - source, days_since_request, payment_type, intended_balcon_amount
   - proposed_credit_limit, foreign_request, name_email_similarity, etc.

2. MENDELEY PII DATASET
   ────────────────────
   Source: https://data.mendeley.com/datasets/compare/sxfjgcynjv
   
   Steps:
   a) Visit the Mendeley Data link
   b) Download the dataset
   c) Extract and place CSV in: ./data/pii_dataset.csv
   
   Expected columns (~15):
   - name, email, credit_card_number, dob, address, phone, etc.

3. AFTER DOWNLOADING
   ──────────────────
   Place both CSVs in the ./data/ folder, then run:
   
   python setup_datasets.py --validate
   python main.py --mode full

Alternative: Generate synthetic data for testing
   python setup_datasets.py --generate
    """)
    logger.info("=" * 70 + "\n")


def validate_data():
    """Validate existing datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("DATA VALIDATION")
    logger.info("=" * 70)
    
    errors = []
    
    # Check BAF
    if not BAF_CSV.exists():
        logger.warning(f"BAF Dataset not found: {BAF_CSV}")
    else:
        try:
            df = pd.read_csv(BAF_CSV)
            logger.info(f"✓ BAF Dataset: {df.shape}")
            
            # Check for required columns
            if "fraud_bool" not in df.columns:
                logger.warning("  - WARNING: 'fraud_bool' column not found")
            else:
                fraud_pct = df["fraud_bool"].mean() * 100
                logger.info(f"  - Fraud rate: {fraud_pct:.2f}%")
            
            # Check for missing values
            missing = df.isnull().sum().sum()
            if missing > 0:
                logger.warning(f"  - {missing} missing values detected")
            else:
                logger.info(f"  - No missing values")
                
        except Exception as e:
            errors.append(f"BAF validation error: {e}")
            logger.error(f"✗ BAF validation failed: {e}")
    
    # Check PII
    if not PII_CSV.exists():
        logger.warning(f"PII Dataset not found: {PII_CSV}")
    else:
        try:
            df = pd.read_csv(PII_CSV)
            logger.info(f"✓ PII Dataset: {df.shape}")
            
            # Check for sensitive columns
            sensitive_cols = ["credit_card_number", "cc_num", "card_number", 
                            "ssn", "dob", "phone"]
            found_cols = [c for c in sensitive_cols if c in df.columns]
            if found_cols:
                logger.info(f"  - Found sensitive columns: {found_cols}")
                logger.warning("  - Remember: Use anonymization before production!")
            
            # Check for missing values
            missing = df.isnull().sum().sum()
            if missing > 0:
                logger.warning(f"  - {missing} missing values detected")
            else:
                logger.info(f"  - No missing values")
                
        except Exception as e:
            errors.append(f"PII validation error: {e}")
            logger.error(f"✗ PII validation failed: {e}")
    
    logger.info("=" * 70 + "\n")
    
    if errors:
        logger.error(f"Validation completed with {len(errors)} error(s)")
        return False
    else:
        logger.info("Validation completed successfully")
        return True


def generate_synthetic():
    """Generate synthetic datasets for testing."""
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING SYNTHETIC DATA")
    logger.info("=" * 70)
    
    # This is a placeholder - the main.py already has synthetic data generation
    # We're just showing how to manually trigger it
    logger.info("""
To generate synthetic data automatically, run:
    python main.py --mode train

This will:
1. Check for real datasets in ./data/
2. If not found, generate synthetic data automatically
3. Train the model on synthetic data
4. Save outputs to ./outputs/

Note: For academic submission, real datasets are HIGHLY RECOMMENDED.
Synthetic data is only for initial testing and development.
    """)
    logger.info("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Deceptive-Net Dataset Setup & Validation"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Show instructions for downloading datasets"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate existing datasets"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate synthetic datasets"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Check current data status"
    )
    
    args = parser.parse_args()
    
    ensure_data_dir()
    
    if args.download or (not any([args.validate, args.generate, args.status])):
        show_download_instructions()
    
    if args.validate:
        validate_data()
    
    if args.generate:
        generate_synthetic()
    
    if args.status:
        check_data_status()


if __name__ == "__main__":
    main()
