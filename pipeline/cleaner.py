"""
Data Cleaner Module for FraudGuard Pipeline.

Handles all data cleaning operations including parsing dirty amounts,
timestamps, normalizing cities/categories, validating IPs, and
producing a comprehensive data quality report.
"""

import re
import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# ─── City normalization map ───────────────────────────────────────────────
CITY_MAP = {
    # Mumbai variants
    "mumbai": "Mumbai", "MUMBAI": "Mumbai", "mUMBAI": "Mumbai",
    "BOM": "Mumbai", "Bombay": "Mumbai", "bombay": "Mumbai",
    "MUMB...": "Mumbai", "MUMB": "Mumbai", "Mumbai": "Mumbai",
    # Delhi variants
    "delhi": "Delhi", "DELHI": "Delhi", "DEL": "Delhi",
    "New Delhi": "Delhi", "New delhi": "Delhi", "new delhi": "Delhi",
    # Bengaluru variants
    "bangalore": "Bengaluru", "Bangalore": "Bengaluru",
    "BENGALURU": "Bengaluru", "BLR": "Bengaluru",
    "Bengaluru": "Bengaluru", "bengaluru": "Bengaluru",
    "Ba": "Bengaluru", "Beng...": "Bengaluru",
    # Chennai variants
    "chennai": "Chennai", "CHENNAI": "Chennai", "MAA": "Chennai",
    "Madras": "Chennai", "Chenna#": "Chennai", "Chennai": "Chennai",
    "Madr#": "Chennai", "Madr": "Chennai", "madras": "Chennai",
    # Hyderabad variants
    "hyderabad": "Hyderabad", "HYDERABAD": "Hyderabad",
    "HYD": "Hyderabad", "HYDERAB#": "Hyderabad",
    "Hyder...": "Hyderabad", "Hyderabad": "Hyderabad",
    # Kolkata variants
    "kolkata": "Kolkata", "Kolkata": "Kolkata", "CCU": "Kolkata",
    "Calcutta": "Kolkata", "calcutta": "Kolkata",
    # Pune variants
    "pune": "Pune", "Pune": "Pune", "PNQ": "Pune",
    # Jaipur variants
    "jaipur": "Jaipur", "Jaipur": "Jaipur", "JAI": "Jaipur",
    "Jai??": "Jaipur", "jai": "Jaipur",
    # Lucknow variants
    "lucknow": "Lucknow", "Lucknow": "Lucknow", "LKO": "Lucknow",
    "Luckn": "Lucknow", "luc#": "Lucknow", "lu#": "Lucknow",
    "L??": "Lucknow",
    # Ahmedabad variants
    "ahmedabad": "Ahmedabad", "Ahmedabad": "Ahmedabad", "AMD": "Ahmedabad",
    # International
    "Dubai": "Dubai", "Singapore": "Singapore",
    "Bangkok": "Bangkok", "New York": "New York",
    "New Yor#": "New York",
}

# ─── Category normalization map ───────────────────────────────────────────
CATEGORY_MAP = {
    "travel": "Travel", "Travel": "Travel", "Tra": "Travel",
    "T#": "Travel", "T??": "Travel",
    "food & dining": "Food & Dining", "Food & Dining": "Food & Dining",
    "Food & Di#": "Food & Dining", "Food & Di...": "Food & Dining",
    "Food & Di??": "Food & Dining", "Food ??": "Food & Dining",
    "food": "Food & Dining",
    "electronics": "Electronics", "Electronics": "Electronics",
    "clothing": "Clothing", "Clothing": "Clothing",
    "Clothin??": "Clothing", "Cl??": "Clothing", "Cl": "Clothing",
    "C#": "Clothing",
    "grocery": "Grocery", "Grocery": "Grocery",
    "Groce#": "Grocery", "Gr...": "Grocery", "Gr": "Grocery",
    "fuel": "Fuel", "Fuel": "Fuel", "Fu??": "Fuel",
    "utilities": "Utilities", "Utilities": "Utilities",
    "Ut...": "Utilities", "Utili": "Utilities",
    "entertainment": "Entertainment", "Entertainment": "Entertainment",
    "Enterta#": "Entertainment", "Ent": "Entertainment",
    "Ent#": "Entertainment",
    "education": "Education", "Education": "Education",
    "healthcare": "Healthcare", "Healthcare": "Healthcare",
    "Healthca??": "Healthcare",
}

# Valid canonical categories for fuzzy matching
VALID_CATEGORIES = [
    "Travel", "Food & Dining", "Electronics", "Clothing",
    "Grocery", "Fuel", "Utilities", "Entertainment",
    "Education", "Healthcare"
]

VALID_CITIES = list(set(CITY_MAP.values()))


class DataCleaner:
    """
    Cleans raw transaction DataFrames by parsing amounts, timestamps,
    normalizing cities/categories, validating IPs, and removing duplicates.
    """

    def __init__(self):
        """Initialize the DataCleaner."""
        self.quality_report: Dict[str, Any] = {}

    @staticmethod
    def parse_amount(val) -> Optional[float]:
        """
        Parse transaction amount from various dirty formats.

        Handles: plain float, ₹ prefix, Rs prefix, INR suffix,
        null/empty values.

        Args:
            val: Raw amount value (str, float, int, or None).

        Returns:
            Cleaned float rounded to 2 decimal places, or None on failure.
        """
        if val is None:
            return None
        if isinstance(val, (int, float)):
            if pd.isna(val):
                return None
            return round(float(val), 2)

        val_str = str(val).strip()
        if val_str == '' or val_str.lower() in ('nan', 'none', 'null', 'na', 'n/a', ''):
            return None

        try:
            # Remove currency symbols and text
            cleaned = val_str
            cleaned = cleaned.replace('₹', '').replace('Rs', '').replace('INR', '')
            cleaned = cleaned.replace(',', '').strip()
            return round(float(cleaned), 2)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def parse_timestamp(val) -> Optional[pd.Timestamp]:
        """
        Parse timestamp from 7+ formats found in the data.

        Handles: ISO 8601, Unix epoch (10-digit), compact 14-digit,
        DD/MM/YYYY HH:MM:SS, Month DD YYYY HH:MM AM/PM,
        DD-Mon-YYYY, MM-DD-YYYY or DD-MM-YYYY.

        Args:
            val: Raw timestamp value.

        Returns:
            pd.Timestamp or None on failure.
        """
        if val is None:
            return None
        if isinstance(val, pd.Timestamp):
            return val
        val_str = str(val).strip()
        if val_str == '' or val_str.lower() in ('nan', 'none', 'null', 'na', 'n/a'):
            return None

        # Try parsing as a numeric value (Unix epoch or compact)
        try:
            numeric_val = float(val_str)
            if len(val_str.split('.')[0]) == 10:
                # Unix epoch (10-digit integer)
                return pd.Timestamp.fromtimestamp(numeric_val)
            elif len(val_str) == 14 and val_str.isdigit():
                # Compact 14-digit: YYYYMMDDHHmmss
                return pd.Timestamp(
                    year=int(val_str[:4]), month=int(val_str[4:6]),
                    day=int(val_str[6:8]), hour=int(val_str[8:10]),
                    minute=int(val_str[10:12]), second=int(val_str[12:14])
                )
        except (ValueError, TypeError, OSError):
            pass

        # Check for compact 14-digit (may have been missed above)
        if len(val_str) == 14 and val_str.isdigit():
            try:
                return pd.Timestamp(
                    year=int(val_str[:4]), month=int(val_str[4:6]),
                    day=int(val_str[6:8]), hour=int(val_str[8:10]),
                    minute=int(val_str[10:12]), second=int(val_str[12:14])
                )
            except (ValueError, TypeError):
                pass

        # Try various string format parsers
        formats_to_try = [
            # ISO 8601
            None,  # pd.to_datetime default
            # DD/MM/YYYY HH:MM:SS
            "%d/%m/%Y %H:%M:%S",
            # MM/DD/YYYY HH:MM:SS
            "%m/%d/%Y %H:%M:%S",
            # Month DD, YYYY HH:MM AM/PM
            "%B %d, %Y %I:%M %p",
            # DD-Mon-YYYY
            "%d-%b-%Y",
            # MM-DD-YYYY HH:MM
            "%m-%d-%Y %H:%M",
            # DD-MM-YYYY HH:MM
            "%d-%m-%Y %H:%M",
        ]

        for fmt in formats_to_try:
            try:
                if fmt is None:
                    result = pd.to_datetime(val_str)
                else:
                    result = pd.to_datetime(val_str, format=fmt)
                return result
            except (ValueError, TypeError):
                continue

        # Last resort: dateutil parser
        try:
            return pd.to_datetime(val_str, dayfirst=True)
        except (ValueError, TypeError):
            pass

        return None

    @staticmethod
    def normalize_city(val) -> str:
        """
        Normalize city name to canonical form using lookup + fuzzy fallback.

        Args:
            val: Raw city string.

        Returns:
            Canonical city name or 'Unknown'.
        """
        if val is None:
            return "Unknown"
        val_str = str(val).strip()
        if val_str == '' or val_str.lower() in ('nan', 'none', 'null', 'na', 'n/a'):
            return "Unknown"

        # Exact match
        if val_str in CITY_MAP:
            return CITY_MAP[val_str]

        # Case-insensitive match
        val_lower = val_str.lower()
        for key, canonical in CITY_MAP.items():
            if key.lower() == val_lower:
                return canonical

        # Partial / fuzzy match — strip trailing special chars
        cleaned = re.sub(r'[#?.…]+$', '', val_str).strip()
        if cleaned in CITY_MAP:
            return CITY_MAP[cleaned]
        for key, canonical in CITY_MAP.items():
            if key.lower() == cleaned.lower():
                return canonical

        # Substring match
        for city in VALID_CITIES:
            if city.lower() in val_lower or val_lower in city.lower():
                return city

        return "Unknown"

    @staticmethod
    def normalize_category(val) -> str:
        """
        Normalize merchant category to canonical form.

        Args:
            val: Raw category string.

        Returns:
            Canonical category name or 'Unknown'.
        """
        if val is None:
            return "Unknown"
        val_str = str(val).strip()
        if val_str == '' or val_str.lower() in ('nan', 'none', 'null', 'na', 'n/a'):
            return "Unknown"

        # Exact match
        if val_str in CATEGORY_MAP:
            return CATEGORY_MAP[val_str]

        # Case-insensitive match
        val_lower = val_str.lower()
        for key, canonical in CATEGORY_MAP.items():
            if key.lower() == val_lower:
                return canonical

        # Strip trailing special chars and retry
        cleaned = re.sub(r'[#?.…]+$', '', val_str).strip()
        if cleaned in CATEGORY_MAP:
            return CATEGORY_MAP[cleaned]
        for key, canonical in CATEGORY_MAP.items():
            if key.lower() == cleaned.lower():
                return canonical

        # Substring match
        for cat in VALID_CATEGORIES:
            if cat.lower().startswith(cleaned.lower()) or cleaned.lower().startswith(cat.lower()[:3]):
                return cat

        return "Unknown"

    @staticmethod
    def validate_ip(val) -> bool:
        """
        Validate whether an IP address has valid format (4 octets, 0-255).

        Args:
            val: Raw IP string.

        Returns:
            True if valid IPv4, False otherwise.
        """
        if val is None:
            return False
        val_str = str(val).strip()
        if val_str == '' or val_str.lower() in ('nan', 'none', 'null', 'na', 'n/a', 'not_an_ip'):
            return False
        parts = val_str.split('.')
        if len(parts) != 4:
            return False
        for part in parts:
            try:
                num = int(part)
                if num < 0 or num > 255:
                    return False
            except (ValueError, TypeError):
                return False
        return True

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Master cleaning method that runs all cleaning operations.

        Args:
            df: Raw transaction DataFrame.

        Returns:
            Tuple of (cleaned DataFrame, data quality report dict).
        """
        report = {
            "total_rows": len(df),
            "duplicate_rows_removed": 0,
            "duplicate_transaction_ids": 0,
            "missing_amount_filled_from_amt": 0,
            "amount_parse_failures": 0,
            "timestamp_parse_failures": 0,
            "city_normalizations": 0,
            "category_normalizations": 0,
            "invalid_ips": 0,
            "missing_per_column": {},
            "zero_balance_rows": 0,
        }

        df = df.copy()

        # ── 1. Remove exact duplicate rows ──
        initial_len = len(df)
        df = df.drop_duplicates()
        report["duplicate_rows_removed"] = initial_len - len(df)

        # ── 2. Identify duplicate transaction IDs ──
        if 'transaction_id' in df.columns:
            dup_txn = df['transaction_id'].duplicated(keep=False)
            report["duplicate_transaction_ids"] = int(dup_txn.sum())

        # ── 3. Parse amounts (with amt column fallback) ──
        if 'transaction_amount' in df.columns:
            # Fill from 'amt' shadow column where amount is missing
            if 'amt' in df.columns:
                mask = df['transaction_amount'].isna() | (df['transaction_amount'].astype(str).str.strip() == '')
                filled_count = 0
                for idx in df[mask].index:
                    amt_val = df.at[idx, 'amt']
                    if pd.notna(amt_val) and str(amt_val).strip() != '':
                        df.at[idx, 'transaction_amount'] = amt_val
                        filled_count += 1
                report["missing_amount_filled_from_amt"] = filled_count

            df['clean_amount'] = df['transaction_amount'].apply(self.parse_amount)
            report["amount_parse_failures"] = int(df['clean_amount'].isna().sum())
        else:
            df['clean_amount'] = np.nan

        # ── 4. Parse timestamps ──
        if 'transaction_timestamp' in df.columns:
            df['clean_timestamp'] = df['transaction_timestamp'].apply(self.parse_timestamp)
            report["timestamp_parse_failures"] = int(df['clean_timestamp'].isna().sum())
        else:
            df['clean_timestamp'] = pd.NaT

        # ── 5. Normalize cities ──
        city_norm_count = 0
        for col in ['user_location', 'merchant_location']:
            if col in df.columns:
                original = df[col].copy()
                canonical_col = f"{col}_canonical" if col == 'user_location' else 'merchant_city_canonical'
                if col == 'user_location':
                    canonical_col = 'user_city_canonical'
                df[canonical_col] = df[col].apply(self.normalize_city)
                changed = (original.astype(str) != df[canonical_col].astype(str))
                city_norm_count += int(changed.sum())
        report["city_normalizations"] = city_norm_count

        # ── 6. Normalize categories ──
        if 'merchant_category' in df.columns:
            original_cat = df['merchant_category'].copy()
            df['clean_category'] = df['merchant_category'].apply(self.normalize_category)
            changed_cat = (original_cat.astype(str) != df['clean_category'].astype(str))
            report["category_normalizations"] = int(changed_cat.sum())
        else:
            df['clean_category'] = "Unknown"

        # ── 7. Validate IPs ──
        if 'ip_address' in df.columns:
            df['ip_valid'] = df['ip_address'].apply(self.validate_ip)
            report["invalid_ips"] = int((~df['ip_valid']).sum())
        else:
            df['ip_valid'] = False

        # ── 8. Missing per column ──
        for col in df.columns:
            if col not in ['clean_amount', 'clean_timestamp', 'ip_valid',
                           'user_city_canonical', 'merchant_city_canonical',
                           'clean_category']:
                missing = int(df[col].isna().sum() + (df[col].astype(str).str.strip().isin(
                    ['', 'nan', 'None', 'null', 'NA', 'N/A']
                )).sum())
                if missing > 0:
                    report["missing_per_column"][col] = missing

        # ── 9. Zero balance rows ──
        if 'account_balance' in df.columns:
            balance = pd.to_numeric(df['account_balance'], errors='coerce')
            df['clean_balance'] = balance
            report["zero_balance_rows"] = int((balance == 0).sum())
        else:
            df['clean_balance'] = np.nan

        # ── 10. Clean payment method ──
        if 'payment_method' in df.columns:
            df['clean_payment_method'] = df['payment_method'].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x).strip().lower()
                not in ('nan', 'none', 'null', 'na', 'n/a', '') else 'Unknown'
            )
        else:
            df['clean_payment_method'] = 'Unknown'

        # ── 11. Clean device type ──
        if 'device_type' in df.columns:
            df['clean_device_type'] = df['device_type'].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x).strip().lower()
                not in ('nan', 'none', 'null', 'na', 'n/a', '') else 'Unknown'
            )
        else:
            df['clean_device_type'] = 'Unknown'

        # ── 12. Clean transaction status ──
        if 'transaction_status' in df.columns:
            df['clean_status'] = df['transaction_status'].apply(
                lambda x: str(x).strip().lower() if pd.notna(x) and str(x).strip().lower()
                not in ('nan', 'none', 'null', 'na', 'n/a', '') else 'unknown'
            )
        else:
            df['clean_status'] = 'unknown'

        self.quality_report = report
        logger.info(f"Data cleaning complete. {report['total_rows']} rows processed.")
        return df, report
