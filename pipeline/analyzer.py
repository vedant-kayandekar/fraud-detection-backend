"""
EDA Analyzer Module for FraudGuard Pipeline.

Produces summary statistics and chart-ready data for the frontend
dashboard. All methods return JSON-serializable Python dicts/lists.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """
    Analyzes cleaned and fraud-labeled DataFrames to produce
    summary statistics and chart data for the dashboard.
    """

    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute summary statistics from the cleaned DataFrame.

        Args:
            df: Cleaned DataFrame with clean_amount, clean_status, etc.

        Returns:
            Dict with amount_stats, status counts, payment/device/category
            distributions, unique counts, and date range.
        """
        amount = df['clean_amount'].dropna()
        stats = {
            "amount_stats": {
                "mean": round(float(amount.mean()), 2) if len(amount) > 0 else 0,
                "median": round(float(amount.median()), 2) if len(amount) > 0 else 0,
                "std": round(float(amount.std()), 2) if len(amount) > 0 else 0,
                "min": round(float(amount.min()), 2) if len(amount) > 0 else 0,
                "max": round(float(amount.max()), 2) if len(amount) > 0 else 0,
                "p95": round(float(amount.quantile(0.95)), 2) if len(amount) > 0 else 0,
                "p99": round(float(amount.quantile(0.99)), 2) if len(amount) > 0 else 0,
            },
            "transaction_status_counts": {},
            "payment_method_counts": {},
            "device_type_counts": {},
            "category_counts": {},
            "unique_users": 0,
            "unique_devices": 0,
            "date_range": {"start": "", "end": ""},
        }

        # Status counts
        if 'clean_status' in df.columns:
            status_counts = df['clean_status'].value_counts()
            stats["transaction_status_counts"] = {
                str(k): int(v) for k, v in status_counts.items()
            }

        # Payment method counts
        if 'clean_payment_method' in df.columns:
            pay_counts = df['clean_payment_method'].value_counts()
            stats["payment_method_counts"] = {
                str(k): int(v) for k, v in pay_counts.items()
            }

        # Device type counts
        if 'clean_device_type' in df.columns:
            dev_counts = df['clean_device_type'].value_counts()
            stats["device_type_counts"] = {
                str(k): int(v) for k, v in dev_counts.items()
            }

        # Category counts
        if 'clean_category' in df.columns:
            cat_counts = df['clean_category'].value_counts()
            stats["category_counts"] = {
                str(k): int(v) for k, v in cat_counts.items()
            }

        # Unique users
        if 'user_id' in df.columns:
            stats["unique_users"] = int(df['user_id'].nunique())

        # Unique devices
        if 'device_id' in df.columns:
            stats["unique_devices"] = int(df['device_id'].nunique())

        # Date range
        if 'clean_timestamp' in df.columns:
            valid_ts = df['clean_timestamp'].dropna()
            if len(valid_ts) > 0:
                stats["date_range"] = {
                    "start": str(valid_ts.min()),
                    "end": str(valid_ts.max()),
                }

        return stats

    def get_chart_data(self, df_clean: pd.DataFrame,
                       df_with_fraud: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate chart-ready data for the frontend dashboard.

        Args:
            df_clean: Cleaned DataFrame (without fraud labels).
            df_with_fraud: DataFrame with predicted_fraud column.

        Returns:
            Dict with fraud_by_category, fraud_by_hour, fraud_by_payment_method,
            fraud_by_device_type, fraud_by_city, amount_distribution,
            daily_fraud_trend, top_fraud_users.
        """
        df = df_with_fraud.copy()
        charts = {}

        # ── Fraud by category ──
        if 'clean_category' in df.columns and 'predicted_fraud' in df.columns:
            cat_fraud = df.groupby('clean_category').agg(
                fraud_count=('predicted_fraud', 'sum'),
                total=('predicted_fraud', 'count')
            ).reset_index()
            cat_fraud['fraud_rate'] = (
                cat_fraud['fraud_count'] / cat_fraud['total'] * 100
            ).round(2)
            charts["fraud_by_category"] = [
                {
                    "category": str(row['clean_category']),
                    "fraud_count": int(row['fraud_count']),
                    "total": int(row['total']),
                    "fraud_rate": float(row['fraud_rate'])
                }
                for _, row in cat_fraud.iterrows()
            ]
        else:
            charts["fraud_by_category"] = []

        # ── Fraud by hour ──
        if 'hour_of_day' in df.columns and 'predicted_fraud' in df.columns:
            hour_data = df.groupby('hour_of_day').agg(
                fraud_count=('predicted_fraud', 'sum'),
                total=('predicted_fraud', 'count')
            ).reindex(range(24), fill_value=0).reset_index()
            hour_data.columns = ['hour', 'fraud_count', 'total']
            charts["fraud_by_hour"] = [
                {
                    "hour": int(row['hour']),
                    "fraud_count": int(row['fraud_count']),
                    "total": int(row['total'])
                }
                for _, row in hour_data.iterrows()
            ]
        else:
            charts["fraud_by_hour"] = []

        # ── Fraud by payment method ──
        if 'clean_payment_method' in df.columns and 'predicted_fraud' in df.columns:
            pay_fraud = df.groupby('clean_payment_method').agg(
                fraud_count=('predicted_fraud', 'sum'),
                total=('predicted_fraud', 'count')
            ).reset_index()
            charts["fraud_by_payment_method"] = [
                {
                    "method": str(row['clean_payment_method']),
                    "fraud_count": int(row['fraud_count']),
                    "total": int(row['total'])
                }
                for _, row in pay_fraud.iterrows()
            ]
        else:
            charts["fraud_by_payment_method"] = []

        # ── Fraud by device type ──
        if 'clean_device_type' in df.columns and 'predicted_fraud' in df.columns:
            dev_fraud = df.groupby('clean_device_type').agg(
                fraud_count=('predicted_fraud', 'sum'),
                total=('predicted_fraud', 'count')
            ).reset_index()
            charts["fraud_by_device_type"] = [
                {
                    "type": str(row['clean_device_type']),
                    "fraud_count": int(row['fraud_count']),
                    "total": int(row['total'])
                }
                for _, row in dev_fraud.iterrows()
            ]
        else:
            charts["fraud_by_device_type"] = []

        # ── Fraud by city ──
        if 'user_city_canonical' in df.columns and 'predicted_fraud' in df.columns:
            city_fraud = df.groupby('user_city_canonical').agg(
                fraud_count=('predicted_fraud', 'sum'),
                total=('predicted_fraud', 'count')
            ).reset_index()
            charts["fraud_by_city"] = [
                {
                    "city": str(row['user_city_canonical']),
                    "fraud_count": int(row['fraud_count']),
                    "total": int(row['total'])
                }
                for _, row in city_fraud.iterrows()
            ]
        else:
            charts["fraud_by_city"] = []

        # ── Amount distribution (histogram buckets) ──
        if 'clean_amount' in df.columns:
            amounts = df['clean_amount'].dropna()
            if len(amounts) > 0:
                buckets = [0, 100, 500, 1000, 2000, 5000, 10000, 50000, 100000, float('inf')]
                labels = [
                    '0-100', '100-500', '500-1K', '1K-2K', '2K-5K',
                    '5K-10K', '10K-50K', '50K-100K', '100K+'
                ]
                hist = pd.cut(amounts, bins=buckets, labels=labels, right=False)
                hist_counts = hist.value_counts().sort_index()
                charts["amount_distribution"] = [
                    {"bucket": str(k), "count": int(v)}
                    for k, v in hist_counts.items()
                ]
            else:
                charts["amount_distribution"] = []
        else:
            charts["amount_distribution"] = []

        # ── Daily fraud trend ──
        if 'clean_timestamp' in df.columns and 'predicted_fraud' in df.columns:
            df_ts = df.dropna(subset=['clean_timestamp']).copy()
            if len(df_ts) > 0:
                df_ts['date'] = df_ts['clean_timestamp'].dt.date
                daily = df_ts.groupby('date').agg(
                    fraud_count=('predicted_fraud', 'sum'),
                    total=('predicted_fraud', 'count')
                ).reset_index()
                daily = daily.sort_values('date')
                charts["daily_fraud_trend"] = [
                    {
                        "date": str(row['date']),
                        "fraud_count": int(row['fraud_count']),
                        "total": int(row['total'])
                    }
                    for _, row in daily.iterrows()
                ]
            else:
                charts["daily_fraud_trend"] = []
        else:
            charts["daily_fraud_trend"] = []

        # ── Top fraud users ──
        if 'user_id' in df.columns and 'predicted_fraud' in df.columns:
            user_fraud = df[df['predicted_fraud'] == 1].groupby('user_id').agg(
                fraud_count=('predicted_fraud', 'sum'),
                total_amount=('clean_amount', 'sum')
            ).reset_index()
            user_fraud = user_fraud.nlargest(10, 'fraud_count')
            charts["top_fraud_users"] = [
                {
                    "user_id": str(row['user_id']),
                    "fraud_count": int(row['fraud_count']),
                    "total_amount": round(float(row['total_amount'] or 0), 2)
                }
                for _, row in user_fraud.iterrows()
            ]
        else:
            charts["top_fraud_users"] = []

        return charts
