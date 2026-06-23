"""
Deceptive-Net – Deception & Honey Token Attribution Module
===========================================================
This module:
1. Reconstructs realistic honey tokens (credit cards, bank accounts, identities, credentials)
   from the outputs of the conditional GAN (cWGAN-GP) trained on synthetic BAF/PII datasets.
2. Enforces Luhn checksum compliance on generated credit card numbers.
3. Manages the Honey Token Registry, watermarking exfiltrated CSV files.
4. Captures and attributes hacker activity logs back to the original breach source.
"""

import os
import csv
import uuid
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Constants for realistic token generation
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Melissa", "George", "Deborah",
    "Timothy", "Stephanie", "Ronald", "Rebecca", "Edward", "Sharon", "Jason", "Laura",
    "Jeffrey", "Cynthia", "Ryan", "Kathleen", "Jacob", "Amy", "Gary", "Angela",
    "Nicholas", "Shirley", "Eric", "Brenda", "Stephen", "Emma", "Jonathan", "Anna",
    "Ronald", "Pamela", "Timothy", "Nicole", "George", "Samantha", "Jeffrey", "Katherine",
    "Gregory", "Christine", "Raymond", "Helen", "Dennis", "Debra", "Jerry", "Rachel",
    "Tyler", "Carolyn", "Aaron", "Janet", "Jose", "Maria", "Adam", "Heather", "Douglas", "Diane"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
    "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza",
    "Ruiz", "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers"
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "proton.me", "icloud.com"]

COMMON_PASSWORDS = [
    "password123", "qwerty12345", "admin123!", "welcome1", "letmein123",
    "charlie01", "shadow99", "superman88", "football2026", "dragonpass",
    "masterkey", "spring2026", "securepass!", "network99", "cyberguard"
]

BANK_NAMES = [
    "Apex Global Bank", "Summit Trust", "Sentinel Commerce", "Vanguard Federal",
    "Horizon Credit Bank", "Meridian Trust", "Pinnacle Financial"
]

def _luhn_check_digit(digits_15: List[int]) -> int:
    """Return the Luhn check digit for a 15-digit prefix."""
    total = 0
    for i, d in enumerate(reversed(digits_15)):
        if i % 2 == 0:
            d2 = d * 2
            total += d2 - 9 if d2 > 9 else d2
        else:
            total += d
    return (10 - total % 10) % 10

class HoneyTokenRegistry:
    def __init__(self):
        # Maps token string (e.g. credit card number or username) -> watermark metadata dict
        self._registry: Dict[str, dict] = {}
        # List of exfiltration session logs
        self._exports: List[dict] = []
        # List of caught attacker logs
        self._caught_hackers: List[dict] = []
        
        # Load pre-trained GAN tokens at startup
        self.honey_tokens: List[dict] = self._load_and_decode_tokens()

    def _load_and_decode_tokens(self) -> List[dict]:
        """Loads outputs/honey_tokens_10k.csv and decodes float vectors into text fields."""
        tokens = []
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "honey_tokens_10k.csv")
        
        # Fallback to random generation if GAN output is missing
        if not os.path.exists(csv_path):
            logger.warning(f"GAN honey tokens file not found at {csv_path}. Generating realistic fallback tokens.")
            return self._generate_fallback_tokens(500)

        try:
            df = pd.read_csv(csv_path, nrows=1000) # Load first 1000 for server memory efficiency
            pii_cols = [c for c in df.columns if c.startswith("pii_")]
            baf_cols = [c for c in df.columns if c.startswith("baf_")]
            
            # Luhn CC columns (last 16 columns of PII features)
            cc_cols = pii_cols[-16:] if len(pii_cols) >= 16 else []
            
            for idx, row in df.iterrows():
                # Reconstruct credit card digits from GAN output
                if len(cc_cols) == 16:
                    cc_vals = row[cc_cols].values
                    digits = np.clip(np.round(cc_vals * 9), 0, 9).astype(int).tolist()
                    
                    # Ensure first digit is realistic for Visa (4) or Mastercard (5)
                    if digits[0] not in [3, 4, 5, 6]:
                        digits[0] = 4 if digits[1] % 2 == 0 else 5
                        
                    # Calculate correct check digit for Luhn compliance
                    check_digit = _luhn_check_digit(digits[:15])
                    digits[15] = check_digit
                    card_num = "".join(map(str, digits))
                else:
                    # Random fallback if columns mismatch
                    card_num = "4" + "".join(map(str, np.random.randint(0, 10, 14)))
                    card_num = card_num + str(_luhn_check_digit([int(d) for d in card_num]))

                # Deterministically decode name from first 8 name-hash columns
                name_hash_sum = int(abs(sum(row[pii_cols[:8]].values)) * 1000)
                first_name = FIRST_NAMES[name_hash_sum % len(FIRST_NAMES)]
                last_name = LAST_NAMES[(name_hash_sum // 7) % len(LAST_NAMES)]
                
                cardholder = f"{first_name} {last_name}"
                email = f"{first_name.lower()}.{last_name.lower()}{name_hash_sum % 99}@{EMAIL_DOMAINS[name_hash_sum % len(EMAIL_DOMAINS)]}"
                
                # Deterministically decode credentials
                username = f"{first_name.lower()[:3]}_{last_name.lower()[:3]}{name_hash_sum % 89 + 10}"
                password = COMMON_PASSWORDS[name_hash_sum % len(COMMON_PASSWORDS)]
                
                # Deterministically decode CVV (normalized 0-1 float at pii_f40 proxy)
                cvv_val = row[pii_cols[min(40, len(pii_cols)-1)]]
                cvv = int(cvv_val * 899) + 100
                
                # Deterministically decode expiry
                exp_month = int(row[pii_cols[min(41, len(pii_cols)-1)]] * 11) + 1
                exp_year = int(row[pii_cols[min(42, len(pii_cols)-1)]] * 5) + 26 # 2026 to 2031
                expiry = f"{exp_month:02d}/{exp_year}"

                # Decode BAF Account features
                baf_val = int(abs(sum(row[baf_cols[:5]].values)) * 100000)
                account_num = f"ACT-{baf_val:08d}"
                bank_name = BANK_NAMES[baf_val % len(BANK_NAMES)]
                balance = round(row[baf_cols[min(10, len(baf_cols)-1)]] * 50000 + 1200, 2)
                monthly_income = round(row[baf_cols[min(1, len(baf_cols)-1)]] * 12000 + 1500, 2)

                tokens.append({
                    "id": f"HTK-{idx:04d}",
                    "card_number": card_num,
                    "cardholder": cardholder,
                    "email": email,
                    "cvv": str(cvv),
                    "expiry": expiry,
                    "username": username,
                    "password": password,
                    "account_number": account_num,
                    "bank_name": bank_name,
                    "balance": balance,
                    "monthly_income": monthly_income,
                    "is_honey": True
                })
            
            logger.info(f"Successfully loaded and decoded {len(tokens)} GAN honey tokens.")
            return tokens
        except Exception as e:
            logger.error(f"Error decoding GAN honey tokens: {e}. Generating fallback tokens.")
            return self._generate_fallback_tokens(500)

    def _generate_fallback_tokens(self, count: int) -> List[dict]:
        """Generates realistic-looking fallback honey tokens if CSV is unavailable."""
        tokens = []
        rng = np.random.default_rng(42)
        for i in range(count):
            first_name = rng.choice(FIRST_NAMES)
            last_name = rng.choice(LAST_NAMES)
            cardholder = f"{first_name} {last_name}"
            email = f"{first_name.lower()}.{last_name.lower()}{rng.integers(10,99)}@{rng.choice(EMAIL_DOMAINS)}"
            
            # Luhn card
            prefix = [4] + rng.integers(0, 10, 14).tolist()
            check = _luhn_check_digit(prefix)
            card_num = "".join(map(str, prefix + [check]))
            
            username = f"{first_name.lower()[:3]}_{last_name.lower()[:3]}{rng.integers(10,99)}"
            password = rng.choice(COMMON_PASSWORDS)
            cvv = rng.integers(100, 1000)
            expiry = f"{rng.integers(1,13):02d}/{rng.integers(26,32)}"
            account_num = f"ACT-{rng.integers(10000000, 99999999)}"
            bank_name = rng.choice(BANK_NAMES)
            balance = round(float(rng.uniform(1500, 25000)), 2)
            monthly_income = round(float(rng.uniform(2000, 12000)), 2)
            
            tokens.append({
                "id": f"HTK-{i:04d}",
                "card_number": card_num,
                "cardholder": cardholder,
                "email": email,
                "cvv": str(cvv),
                "expiry": expiry,
                "username": username,
                "password": password,
                "account_number": account_num,
                "bank_name": bank_name,
                "balance": balance,
                "monthly_income": monthly_income,
                "is_honey": True
            })
        return tokens

    def register_export(self, watermark_id: str, tokens: List[dict], metadata: dict) -> None:
        """Registers a set of tokens as 'exported' under a specific watermark."""
        timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S IST")
        export_record = {
            "watermark_id": watermark_id,
            "timestamp": timestamp,
            "actor": metadata.get("actor", "unknown"),
            "ip": metadata.get("ip", "unknown"),
            "token_count": len(tokens),
            "tokens": [t["id"] for t in tokens]
        }
        self._exports.append(export_record)
        
        # Link individual cards/usernames back to this export record
        for t in tokens:
            self._registry[t["card_number"]] = export_record
            self._registry[t["username"]] = export_record

    def check_token(self, token_val: str) -> Optional[dict]:
        """Checks if a value matches a registered Honey Token, returning breach metadata."""
        # Strip any formatting spaces/hyphens
        clean_val = token_val.replace(" ", "").replace("-", "").strip()
        return self._registry.get(clean_val)

    def log_caught_hacker(self, attacker_ip: str, user_agent: str, token_used: str, 
                          token_type: str, attribution: dict, action: str, request_payload: dict = None) -> dict:
        """Logs a caught attacker with reverse-engineered attribution back to the breach source."""
        timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S IST")
        
        # Derive country/location based on simulated IP mapping (for UI richness)
        simulated_locations = {
            "185.": "Moscow, RU (via TOR Exit Node)",
            "109.": "Amsterdam, NL (VPN proxy)",
            "195.": "Saint Petersburg, RU (de-cloaked proxy)",
            "45.": "Frankfurt, DE",
            "91.": "Kiev, UA",
            "127.": "Localhost Sandbox",
            "192.": "Local LAN Network"
        }
        location = "Unknown Cyber-Threat Source"
        for prefix, loc in simulated_locations.items():
            if attacker_ip.startswith(prefix):
                location = loc
                break
                
        hacker_record = {
            "id": f"HAK-{len(self._caught_hackers)+1:03d}",
            "ts": timestamp,
            "attacker_ip": attacker_ip,
            "user_agent": user_agent,
            "location": location,
            "token_used": token_used,
            "token_type": token_type,
            "action": action,
            "request_payload": request_payload or {},
            "attribution": {
                "watermark_id": attribution["watermark_id"],
                "leak_timestamp": attribution["timestamp"],
                "leak_actor": attribution["actor"],
                "leak_ip": attribution["ip"]
            }
        }
        self._caught_hackers.append(hacker_record)
        return hacker_record

    def get_exports(self) -> List[dict]:
        return self._exports

    def get_caught_hackers(self) -> List[dict]:
        return self._caught_hackers

# Single global instance
deception_registry = HoneyTokenRegistry()
