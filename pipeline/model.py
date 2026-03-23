"""
ML Model Module for FraudGuard Pipeline.

Two-stage fraud detection: IsolationForest (unsupervised pseudo-labels)
→ XGBoost (supervised classification) + SHAP explainability.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

# Features used by IsolationForest
IF_FEATURES = [
    'clean_amount', 'amount_zscore', 'hour_of_day', 'is_night',
    'is_zero_balance_success', 'is_micro_transaction',
    'is_new_device_prefix', 'is_cnp_device', 'is_location_mismatch',
    'is_international', 'user_txn_velocity_1hr', 'amount_vs_user_avg_ratio',
]

# All 20 engineered features used by XGBoost
XGB_FEATURES = [
    'clean_amount', 'amount_zscore', 'user_avg_amount', 'user_std_amount',
    'hour_of_day', 'day_of_week', 'is_night', 'is_weekend',
    'is_outlier_amount', 'is_micro_transaction', 'is_zero_amount',
    'is_zero_balance_success', 'amount_vs_user_avg_ratio',
    'is_location_mismatch', 'is_international',
    'is_new_device_prefix', 'is_cnp_device', 'device_is_new_for_user',
    'user_txn_velocity_1hr', 'user_txn_velocity_24hr',
    'is_failed_high_amount', 'device_multi_user',
]


class FraudDetector:
    """
    Two-stage fraud detector using IsolationForest + XGBoost + SHAP.

    Stage 1: IsolationForest generates pseudo-labels (unsupervised).
    Stage 2: XGBoost trains on pseudo-labels (supervised) for probabilities.
    SHAP: Explains top fraud predictions with feature importance.
    """

    def __init__(self):
        """Initialize FraudDetector with no trained model."""
        self.xgb_model = None
        self.iso_model = None
        self.feature_importance: Dict[str, float] = {}
        self.last_df_features: Optional[pd.DataFrame] = None

    def _prepare_features(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """
        Prepare feature matrix by selecting columns and filling NaN.

        Args:
            df: DataFrame with engineered features.
            feature_list: List of column names to use.

        Returns:
            Clean feature DataFrame with no NaN.
        """
        available = [f for f in feature_list if f in df.columns]
        X = df[available].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        return X

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the full two-stage fraud detection pipeline.

        Args:
            df: DataFrame with all engineered features (from FeatureEngineer).

        Returns:
            Dict with fraud_count, fraud_rate, total_processed, precision,
            recall, f1_score, fraud_rows (top 500), feature_importance,
            and shap_summary.
        """
        import xgboost as xgb

        self.last_df_features = df.copy()
        X_iso = self._prepare_features(df, IF_FEATURES)

        # ── Stage 1: Isolation Forest ──
        logger.info("Stage 1: Running IsolationForest...")
        self.iso_model = IsolationForest(
            contamination=0.08,
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        )
        iso_labels = self.iso_model.fit_predict(X_iso)
        # -1 = anomaly (fraud), 1 = normal
        df['pseudo_fraud'] = (iso_labels == -1).astype(int)

        # ── Stage 2: XGBoost ──
        logger.info("Stage 2: Training XGBoost...")
        X_xgb = self._prepare_features(df, XGB_FEATURES)
        y = df['pseudo_fraud']

        fraud_count_pseudo = int(y.sum())
        non_fraud_count = int((y == 0).sum())
        scale_pos = max(non_fraud_count / max(fraud_count_pseudo, 1), 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X_xgb, y, test_size=0.2, random_state=42, stratify=y
        )

        self.xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(X_train, y_train)

        # Predict probabilities
        fraud_proba = self.xgb_model.predict_proba(X_xgb)[:, 1]
        df['fraud_probability'] = fraud_proba
        df['predicted_fraud'] = (fraud_proba > 0.5).astype(int)

        # Evaluate on test set
        y_pred_test = self.xgb_model.predict(X_test)
        precision = float(precision_score(y_test, y_pred_test, zero_division=0))
        recall = float(recall_score(y_test, y_pred_test, zero_division=0))
        f1 = float(f1_score(y_test, y_pred_test, zero_division=0))

        # Feature importance
        importance = self.xgb_model.feature_importances_
        feat_names = X_xgb.columns.tolist()
        self.feature_importance = {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(feat_names, importance),
                key=lambda x: x[1], reverse=True
            )
        }

        # ── SHAP ──
        shap_summary = {}
        shap_reasons_map = {}
        try:
            import shap
            logger.info("Computing SHAP values...")

            # Only compute for top 1000 fraud rows to save memory
            fraud_mask = df['predicted_fraud'] == 1
            fraud_indices = df[fraud_mask].nlargest(
                min(1000, fraud_mask.sum()), 'fraud_probability'
            ).index

            X_shap = X_xgb.loc[fraud_indices]
            explainer = shap.TreeExplainer(self.xgb_model)
            shap_values = explainer.shap_values(X_shap)

            # Mean absolute SHAP per feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_summary = {
                name: round(float(val), 4)
                for name, val in zip(feat_names, mean_abs_shap)
            }

            # Per-row top 3 SHAP reasons
            for i, idx in enumerate(fraud_indices):
                row_shap = shap_values[i]
                top_indices = np.argsort(np.abs(row_shap))[-3:][::-1]
                reasons = []
                for ti in top_indices:
                    impact = "high" if abs(row_shap[ti]) > 0.5 else "medium" if abs(row_shap[ti]) > 0.2 else "low"
                    reasons.append({
                        "feature": feat_names[ti],
                        "value": round(float(X_shap.iloc[i, ti]), 2),
                        "impact": impact
                    })
                shap_reasons_map[idx] = reasons
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

        # ── Build fraud rows list ──
        fraud_df = df[df['predicted_fraud'] == 1].nlargest(
            min(500, df['predicted_fraud'].sum()), 'fraud_probability'
        ).copy()
        fraud_df['fraud_rank'] = range(1, len(fraud_df) + 1)

        fraud_rows = []
        for _, row in fraud_df.iterrows():
            reasons = shap_reasons_map.get(row.name, [])
            fraud_rows.append({
                "transaction_id": str(row.get('transaction_id', '')),
                "user_id": str(row.get('user_id', '')),
                "clean_amount": round(float(row.get('clean_amount', 0) or 0), 2),
                "clean_timestamp": str(row.get('clean_timestamp', '')),
                "user_city": str(row.get('user_city_canonical', '')),
                "merchant_category": str(row.get('clean_category', '')),
                "fraud_probability": round(float(row['fraud_probability']), 4),
                "fraud_rank": int(row['fraud_rank']),
                "shap_reasons": reasons,
                "device_id": str(row.get('device_id', '')),
                "hour_of_day": int(row.get('hour_of_day', 0)),
                "device_type": str(row.get('clean_device_type', '')),
            })

        total_fraud = int(df['predicted_fraud'].sum())
        total = len(df)

        result = {
            "fraud_count": total_fraud,
            "fraud_rate": round(total_fraud / max(total, 1) * 100, 2),
            "total_processed": total,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "fraud_rows": fraud_rows,
            "feature_importance": self.feature_importance,
            "shap_summary": shap_summary,
        }

        logger.info(f"Fraud detection complete. {total_fraud}/{total} flagged.")
        return result

    def predict_single(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud probability for a single transaction.

        Uses the model trained in the last detect() call. Applies cleaning
        and feature engineering to the single row using stored baselines.

        Args:
            row_dict: Dict with transaction field values.

        Returns:
            Dict with fraud_probability, is_fraud, confidence, reasons.
        """
        if self.xgb_model is None:
            return {
                "fraud_probability": 0.0,
                "is_fraud": False,
                "confidence": "Low",
                "reasons": ["No model trained yet. Upload a dataset first."]
            }

        try:
            from pipeline.cleaner import DataCleaner
            from pipeline.features import FeatureEngineer

            # Create single-row DataFrame
            single_df = pd.DataFrame([row_dict])

            # Clean
            cleaner = DataCleaner()
            single_df, _ = cleaner.clean(single_df)

            # Feature engineering with stored baselines
            fe = FeatureEngineer()
            if self.last_df_features is not None:
                # Merge user baselines from the last dataset
                fe.compute_user_baselines(self.last_df_features)
                single_df = single_df.merge(
                    fe.user_baselines, on='user_id', how='left'
                )

            single_df = fe.engineer_features(single_df)

            # Predict
            X = self._prepare_features(single_df, XGB_FEATURES)
            proba = float(self.xgb_model.predict_proba(X)[:, 1][0])
            is_fraud = proba > 0.5

            if proba > 0.8:
                confidence = "High"
            elif proba > 0.5:
                confidence = "Medium"
            else:
                confidence = "Low"

            # SHAP reasons
            reasons = []
            try:
                import shap
                explainer = shap.TreeExplainer(self.xgb_model)
                sv = explainer.shap_values(X)
                feat_names = X.columns.tolist()
                top_indices = np.argsort(np.abs(sv[0]))[-3:][::-1]
                for ti in top_indices:
                    reasons.append({
                        "feature": feat_names[ti],
                        "value": round(float(X.iloc[0, ti]), 2),
                        "impact": "high" if abs(sv[0][ti]) > 0.5 else "medium"
                    })
            except Exception:
                reasons = [{"feature": "model_score", "value": proba, "impact": "high"}]

            return {
                "fraud_probability": round(proba, 4),
                "is_fraud": is_fraud,
                "confidence": confidence,
                "reasons": reasons
            }
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            return {
                "fraud_probability": 0.0,
                "is_fraud": False,
                "confidence": "Low",
                "reasons": [f"Prediction error: {str(e)}"]
            }
