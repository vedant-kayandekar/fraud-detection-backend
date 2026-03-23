"""
Feature Engineering Module for FraudGuard Pipeline.

Computes 20 fraud-signal features from cleaned transaction data,
including per-user baselines, temporal features, amount anomalies,
location mismatches, device signals, and velocity features.
All operations are vectorized — no .iterrows().
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

INTERNATIONAL_CITIES = {"Dubai", "Singapore", "Bangkok", "New York"}


class FeatureEngineer:
    """
    Engineers fraud-detection features from cleaned transaction DataFrames.
    All operations use vectorized pandas operations for performance.
    """

    def __init__(self):
        """Initialize FeatureEngineer with empty user baselines."""
        self.user_baselines: Optional[pd.DataFrame] = None

    def compute_user_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-user baseline statistics from their transaction history.

        Args:
            df: Cleaned DataFrame with 'user_id' and 'clean_amount'.

        Returns:
            DataFrame with user-level aggregations (mean, std, mode city/device/payment).
        """
        baselines = df.groupby('user_id').agg(
            user_avg_amount=('clean_amount', 'mean'),
            user_std_amount=('clean_amount', 'std'),
        ).reset_index()

        # Most frequent city per user
        if 'user_city_canonical' in df.columns:
            home_city = df.groupby('user_id')['user_city_canonical'].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            ).reset_index()
            home_city.columns = ['user_id', 'user_home_city']
            baselines = baselines.merge(home_city, on='user_id', how='left')
        else:
            baselines['user_home_city'] = 'Unknown'

        # Most frequent device per user
        if 'device_id' in df.columns:
            usual_device = df.groupby('user_id')['device_id'].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            ).reset_index()
            usual_device.columns = ['user_id', 'user_usual_device']
            baselines = baselines.merge(usual_device, on='user_id', how='left')
        else:
            baselines['user_usual_device'] = 'Unknown'

        # Most frequent payment method per user
        if 'clean_payment_method' in df.columns:
            usual_payment = df.groupby('user_id')['clean_payment_method'].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            ).reset_index()
            usual_payment.columns = ['user_id', 'user_usual_payment']
            baselines = baselines.merge(usual_payment, on='user_id', how='left')
        else:
            baselines['user_usual_payment'] = 'Unknown'

        # Fill NaN std with 0 (single-transaction users)
        baselines['user_std_amount'] = baselines['user_std_amount'].fillna(0)

        self.user_baselines = baselines
        return baselines

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all 20 fraud-signal features. Must be called after clean().

        Args:
            df: Cleaned DataFrame from DataCleaner.clean().

        Returns:
            DataFrame with 20 new feature columns appended.
        """
        df = df.copy()

        # ── Compute user baselines ──
        baselines = self.compute_user_baselines(df)
        df = df.merge(baselines, on='user_id', how='left')

        # ── 1-3. Amount z-score features ──
        df['amount_zscore'] = np.where(
            df['user_std_amount'] > 0,
            (df['clean_amount'] - df['user_avg_amount']) / df['user_std_amount'],
            0.0
        )

        # ── 4-5. Temporal features ──
        df['hour_of_day'] = df['clean_timestamp'].dt.hour.fillna(12).astype(int)
        df['day_of_week'] = df['clean_timestamp'].dt.dayofweek.fillna(0).astype(int)

        # ── 6-7. Time-based binary flags ──
        df['is_night'] = ((df['hour_of_day'] >= 2) & (df['hour_of_day'] <= 5)).astype(int)
        df['is_weekend'] = ((df['day_of_week'] >= 5)).astype(int)

        # ── 8. Outlier amount ──
        df['is_outlier_amount'] = (df['amount_zscore'].abs() > 3).astype(int)

        # ── 9-10. Micro / zero amount ──
        df['is_micro_transaction'] = (
            (df['clean_amount'] > 0) & (df['clean_amount'] < 10)
        ).astype(int)
        df['is_zero_amount'] = (df['clean_amount'] == 0).astype(int)

        # ── 11. Zero balance + success ──
        df['is_zero_balance_success'] = (
            (df['clean_balance'] == 0) & (df['clean_status'] == 'success')
        ).astype(int)

        # ── 12. Amount vs user avg ratio ──
        df['amount_vs_user_avg_ratio'] = np.where(
            df['user_avg_amount'] > 0,
            (df['clean_amount'] / df['user_avg_amount']).clip(upper=20),
            0.0
        )

        # ── 13. Location mismatch ──
        if 'user_city_canonical' in df.columns and 'merchant_city_canonical' in df.columns:
            df['is_location_mismatch'] = (
                df['user_city_canonical'] != df['merchant_city_canonical']
            ).astype(int)
        else:
            df['is_location_mismatch'] = 0

        # ── 14. International transaction ──
        if 'user_city_canonical' in df.columns:
            df['is_international'] = df['user_city_canonical'].isin(
                INTERNATIONAL_CITIES
            ).astype(int)
        else:
            df['is_international'] = 0

        # ── 15-16. Device prefix flags ──
        if 'device_id' in df.columns:
            device_str = df['device_id'].astype(str).fillna('')
            df['is_new_device_prefix'] = device_str.str.startswith('NEW-').astype(int)
            df['is_cnp_device'] = device_str.str.startswith('CNP-').astype(int)
        else:
            df['is_new_device_prefix'] = 0
            df['is_cnp_device'] = 0

        # ── 17. Device new for user (first appearance) ──
        if 'device_id' in df.columns and 'clean_timestamp' in df.columns:
            df_sorted = df.sort_values(['user_id', 'clean_timestamp'])
            df_sorted['_dev_cumcount'] = df_sorted.groupby(
                ['user_id', 'device_id']
            ).cumcount()
            df['device_is_new_for_user'] = (
                df_sorted['_dev_cumcount'] == 0
            ).astype(int).values
            # Re-align to original index
            df['device_is_new_for_user'] = df_sorted.set_index(df.index)['_dev_cumcount'].eq(0).astype(int)
        else:
            df['device_is_new_for_user'] = 0

        # ── 18-19. Transaction velocity ──
        if 'clean_timestamp' in df.columns:
            df = df.sort_values(['user_id', 'clean_timestamp']).reset_index(drop=True)

            # Velocity: count of same-user transactions in previous 1hr and 24hr
            velocity_1hr = []
            velocity_24hr = []
            for uid in df['user_id'].unique():
                mask = df['user_id'] == uid
                user_df = df.loc[mask].copy()
                ts = user_df['clean_timestamp']
                v1 = []
                v24 = []
                for i, row_ts in enumerate(ts):
                    if pd.isna(row_ts):
                        v1.append(0)
                        v24.append(0)
                        continue
                    prev = ts.iloc[:i]
                    prev_valid = prev.dropna()
                    if len(prev_valid) == 0:
                        v1.append(0)
                        v24.append(0)
                    else:
                        delta = row_ts - prev_valid
                        v1.append(int((delta <= pd.Timedelta(hours=1)).sum()))
                        v24.append(int((delta <= pd.Timedelta(hours=24)).sum()))
                velocity_1hr.extend(v1)
                velocity_24hr.extend(v24)

            df['user_txn_velocity_1hr'] = velocity_1hr
            df['user_txn_velocity_24hr'] = velocity_24hr
        else:
            df['user_txn_velocity_1hr'] = 0
            df['user_txn_velocity_24hr'] = 0

        # ── 20. Failed high amount ──
        df['is_failed_high_amount'] = (
            (df['clean_status'] == 'failed') & (df['clean_amount'] > 5000)
        ).astype(int)

        # ── 21. Device used by multiple users ──
        if 'device_id' in df.columns:
            dev_user_count = df.groupby('device_id')['user_id'].nunique().reset_index()
            dev_user_count.columns = ['device_id', '_dev_user_count']
            df = df.merge(dev_user_count, on='device_id', how='left')
            df['device_multi_user'] = (df['_dev_user_count'] > 1).astype(int)
            df.drop(columns=['_dev_user_count'], inplace=True)
        else:
            df['device_multi_user'] = 0

        logger.info(f"Feature engineering complete. {len(df)} rows, features appended.")
        return df
