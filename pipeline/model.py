"""
ML Model Module for FraudGuard Pipeline.

Two-track parallel architecture:
  Track A (fast): IsolationForest → XGBoost → SHAP → dashboard (~15s)
  Track B (background): RF, LR, DT trained in parallel via threads → comparison table via polling
"""

import time
import logging
import threading
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score
)

logger = logging.getLogger(__name__)

# Features used by IsolationForest
IF_FEATURES = [
    'clean_amount', 'amount_zscore', 'hour_of_day', 'is_night',
    'is_zero_balance_success', 'is_micro_transaction',
    'is_new_device_prefix', 'is_cnp_device', 'is_location_mismatch',
    'is_international', 'user_txn_velocity_1hr', 'amount_vs_user_avg_ratio',
    'is_pending_status', 'is_duplicate_txn_id', 'amount_exceeds_balance',
]

# All engineered features used by supervised models
MODEL_FEATURES = [
    'clean_amount', 'amount_zscore', 'user_avg_amount', 'user_std_amount',
    'hour_of_day', 'day_of_week', 'is_night', 'is_weekend',
    'is_outlier_amount', 'is_micro_transaction', 'is_zero_amount',
    'is_zero_balance_success', 'amount_vs_user_avg_ratio',
    'is_location_mismatch', 'is_international',
    'is_new_device_prefix', 'is_cnp_device', 'device_is_new_for_user',
    'user_txn_velocity_1hr', 'user_txn_velocity_24hr',
    'is_failed_high_amount', 'device_multi_user',
    'is_pending_status', 'is_duplicate_txn_id', 'amount_exceeds_balance',
    'is_round_amount', 'is_atm_high_amount', 'is_failed_any',
    'risk_score_composite', 'amount_balance_ratio',
    'user_city_consistency', 'velocity_amount_product', 'night_high_spend',
]

WHY_USED = {
    "XGBoost": "Gradient boosting that corrects its own errors sequentially. Best at catching subtle multi-signal fraud patterns.",
    "Random Forest": "Ensemble of independent trees averaged together. Robust and resistant to overfitting.",
    "Logistic Regression": "Linear baseline model. Fast and interpretable but struggles with complex non-linear fraud patterns.",
    "Decision Tree": "Single tree with explicit if-then rules. Fully transparent but prone to overfitting on noisy data.",
}

# In-memory job store for background comparison results
job_store: Dict[str, Dict[str, Any]] = {}


def run_single_model(
    model_name: str,
    model_instance,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    results_dict: dict,
) -> None:
    """
    Train one model and store its metrics in results_dict.
    Thread-safe — each thread writes to a unique key.

    Args:
        model_name: Name of the model.
        model_instance: Sklearn-compatible estimator.
        X_train, X_test, y_train, y_test: Split data.
        results_dict: Shared dict to write results into.
    """
    try:
        start = time.time()
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)

        if hasattr(model_instance, 'predict_proba'):
            y_proba = model_instance.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            auc = 0.0

        results_dict[model_name] = {
            "model_name": model_name,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(auc, 4),
            "training_time_seconds": round(time.time() - start, 2),
            "is_best": False,
            "why_used": WHY_USED.get(model_name, ""),
            "status": "complete",
        }
        logger.info(f"  {model_name} done in {results_dict[model_name]['training_time_seconds']}s "
                     f"(F1={results_dict[model_name]['f1_score']})")
    except Exception as e:
        logger.warning(f"  {model_name} failed: {e}")
        results_dict[model_name] = {
            "model_name": model_name,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "f1_score": 0.0, "roc_auc": 0.0,
            "training_time_seconds": 0.0,
            "is_best": False,
            "why_used": WHY_USED.get(model_name, ""),
            "status": "failed",
            "error": str(e),
        }


class FraudDetector:
    """
    Two-track parallel fraud detector.

    Track A: IsolationForest → XGBoost → SHAP → dashboard (fast path).
    Track B: RF, LR, DT in background threads → comparison via polling.
    """

    def __init__(self):
        """Initialize FraudDetector."""
        self.best_model = None
        self.best_model_name: str = "XGBoost"
        self.iso_model = None
        self.feature_importance: Dict[str, float] = {}
        self.last_df_features: Optional[pd.DataFrame] = None
        self._X_train = None
        self._y_train = None

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
        for col in X.select_dtypes(include=['category']).columns:
            X[col] = X[col].cat.codes
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        for col in X.columns:
            if X[col].dtype == 'float16':
                X[col] = X[col].astype('float32')
        return X

    def detect(self, df: pd.DataFrame, job_id: str) -> Dict[str, Any]:
        """
        Run two-track fraud detection pipeline.

        Track A: XGBoost fast path — blocks, returns dashboard payload.
        Track B: Other 3 models — launched in background threads.

        Args:
            df: DataFrame with all engineered features.
            job_id: Unique job identifier for background comparison polling.

        Returns:
            Dict with fraud_count, fraud_rate, fraud_rows, feature_importance,
            shap_summary, model_comparison (XGBoost only initially), job_id.
        """
        self.last_df_features = df.copy()
        X_iso = self._prepare_features(df, IF_FEATURES)

        # ── Step 1: Isolation Forest — runs once, shared by both tracks ──
        logger.info("Step 1: Running IsolationForest for pseudo-labels...")
        self.iso_model = IsolationForest(
            contamination=0.107,
            random_state=42,
            n_estimators=100,
        )
        iso_labels = self.iso_model.fit_predict(X_iso)
        df['pseudo_fraud'] = (iso_labels == -1).astype(int)

        X_all = self._prepare_features(df, MODEL_FEATURES)
        y = df['pseudo_fraud']
        feat_names = X_all.columns.tolist()

        fraud_count_pseudo = int(y.sum())
        non_fraud_count = int((y == 0).sum())
        scale_pos = max(non_fraud_count / max(fraud_count_pseudo, 1), 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=0.2, random_state=42, stratify=y
        )
        self._X_train = X_train
        self._y_train = y_train

        X_train_np = X_train.values
        X_test_np = X_test.values
        y_train_np = y_train.values
        y_test_np = y_test.values

        # ══════════════════════════════════════════════════════════════
        # Track A: XGBoost FAST PATH — runs immediately, blocks
        # ══════════════════════════════════════════════════════════════
        logger.info("Track A: Training XGBoost (fast path)...")
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=1,
        )

        track_a_results = {}
        run_single_model("XGBoost", xgb_model, X_train_np, X_test_np,
                         y_train_np, y_test_np, track_a_results)

        self.best_model = xgb_model
        self.best_model_name = "XGBoost"

        # Predict on full dataset with XGBoost
        fraud_proba = xgb_model.predict_proba(X_all.values)[:, 1]
        df['fraud_probability'] = fraud_proba
        df['predicted_fraud'] = (fraud_proba > 0.45).astype(int)

        # Feature importance
        if hasattr(xgb_model, 'feature_importances_'):
            importance = xgb_model.feature_importances_
            self.feature_importance = {
                name: round(float(imp), 4)
                for name, imp in sorted(
                    zip(feat_names, importance),
                    key=lambda x: x[1], reverse=True
                )
            }



        # ── SHAP on XGBoost fraud rows only ──
        shap_summary = {}
        shap_reasons_map = {}
        try:
            import shap
            logger.info("Track A: Computing SHAP on XGBoost fraud rows...")

            fraud_mask = df['predicted_fraud'] == 1
            X_fraud = X_all[fraud_mask]

            if len(X_fraud) > 200:
                X_fraud = X_fraud.sample(200, random_state=42)

            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_fraud)

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_summary = {
                name: round(float(val), 4)
                for name, val in zip(feat_names, mean_abs_shap)
            }

            for i, idx in enumerate(X_fraud.index):
                row_shap = shap_values[i]
                top_indices = np.argsort(np.abs(row_shap))[-3:][::-1]
                reasons = []
                for ti in top_indices:
                    sv = float(row_shap[ti])
                    reasons.append({
                        "feature": feat_names[ti],
                        "direction": "increases fraud risk" if sv > 0 else "decreases fraud risk",
                        "impact_score": round(abs(sv), 4),
                        "value": round(float(X_fraud.iloc[i, ti]), 2),
                        "impact": "high" if abs(sv) > 0.5 else "medium" if abs(sv) > 0.2 else "low"
                    })
                shap_reasons_map[idx] = reasons
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

        # ── Build fraud rows list ──
        fraud_df = df[df['predicted_fraud'] == 1].nlargest(
            min(500, int(df['predicted_fraud'].sum())), 'fraud_probability'
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

        xgb_entry = track_a_results["XGBoost"]
        xgb_entry["is_best"] = True

        # ══════════════════════════════════════════════════════════════
        # Track B: Background comparison — launch in threads
        # ══════════════════════════════════════════════════════════════
        background_results = {
            "XGBoost": dict(xgb_entry)  # copy, already done
        }

        # Initialize job store entry
        job_store[job_id] = {
            "comparison": {
                "status": "processing",
                "models": [
                    dict(xgb_entry),
                    {"model_name": "Random Forest", "status": "processing", "why_used": WHY_USED["Random Forest"]},
                    {"model_name": "Logistic Regression", "status": "processing", "why_used": WHY_USED["Logistic Regression"]},
                    {"model_name": "Decision Tree", "status": "processing", "why_used": WHY_USED["Decision Tree"]},
                ],
                "best_model_name": "XGBoost",
                "best_model_f1": xgb_entry["f1_score"],
                "note": "XGBoost results used for dashboard. Other models training in background...",
            }
        }

        comparison_models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight='balanced',
                random_state=42, n_jobs=1,
            ),
            "Logistic Regression": LogisticRegression(
                class_weight='balanced', max_iter=1000, random_state=42,
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=5, class_weight='balanced', random_state=42,
            ),
        }

        def run_background_comparison():
            """Train remaining 3 models in parallel threads."""
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    futures = {
                        name: executor.submit(
                            run_single_model, name, model,
                            X_train_np, X_test_np, y_train_np, y_test_np,
                            background_results
                        )
                        for name, model in comparison_models.items()
                    }
                    for future in futures.values():
                        future.result()

                # Determine overall best across all 4
                completed = {
                    k: v for k, v in background_results.items()
                    if v.get("status") == "complete"
                }
                if completed:
                    best_name = max(completed, key=lambda k: completed[k]["f1_score"])
                    for name in completed:
                        completed[name]["is_best"] = (name == best_name)

                    job_store[job_id]["comparison"] = {
                        "status": "complete",
                        "models": list(background_results.values()),
                        "best_model_name": best_name,
                        "best_model_f1": completed[best_name]["f1_score"],
                        "note": (
                            f"XGBoost was used for dashboard predictions. "
                            f"Overall best model: {best_name} "
                            f"(F1={completed[best_name]['f1_score']})."
                        ),
                    }
                    logger.info(f"Track B: All comparisons done. Best overall: {best_name}")
                else:
                    job_store[job_id]["comparison"]["status"] = "complete"
                    logger.warning("Track B: No models completed successfully")
            except Exception as e:
                logger.error(f"Track B failed: {e}")
                job_store[job_id]["comparison"]["status"] = "complete"

        # Launch background thread — does NOT block the response
        bg_thread = threading.Thread(target=run_background_comparison, daemon=True)
        bg_thread.start()
        logger.info("Track B: Background comparison launched (3 models)")

        # ── Build dashboard result (Track A only) ──
        result = {
            "fraud_count": total_fraud,
            "fraud_rate": round(total_fraud / max(total, 1) * 100, 2),
            "total_processed": total,
            "precision": xgb_entry["precision"],
            "recall": xgb_entry["recall"],
            "f1_score": xgb_entry["f1_score"],
            "fraud_rows": fraud_rows,
            "feature_importance": self.feature_importance,
            "shap_summary": shap_summary,
            "model_comparison": [dict(xgb_entry)],
            "best_model_name": "XGBoost",
            "best_model_f1": xgb_entry["f1_score"],
            "tuned_model_metrics": None,
            "job_id": job_id,
        }

        logger.info(f"Track A complete. {total_fraud}/{total} flagged. Dashboard ready.")
        return result

    def predict_single(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud probability for a single transaction.
        Uses the XGBoost model from the last detect() call.

        Args:
            row_dict: Dict with transaction field values.

        Returns:
            Dict with fraud_probability, is_fraud, confidence,
            model_used, reasons, risk_level.
        """
        if self.best_model is None:
            return {
                "fraud_probability": 0.0,
                "is_fraud": False,
                "confidence": "Low",
                "model_used": "None",
                "reasons": [{"feature": "no_model", "direction": "N/A", "impact_score": 0.0}],
                "risk_level": "Low",
            }

        try:
            from pipeline.cleaner import DataCleaner
            from pipeline.features import FeatureEngineer

            single_df = pd.DataFrame([row_dict])
            cleaner = DataCleaner()
            single_df, _ = cleaner.clean(single_df)

            fe = FeatureEngineer()
            if self.last_df_features is not None:
                fe.compute_user_baselines(self.last_df_features)
                if fe.user_baselines is not None:
                    single_df = single_df.merge(
                        fe.user_baselines, on='user_id', how='left'
                    )

            single_df = fe.engineer_features(single_df)
            X = self._prepare_features(single_df, MODEL_FEATURES)
            proba = float(self.best_model.predict_proba(X)[:, 1][0])
            is_fraud = proba > 0.5

            if proba < 0.3:
                confidence, risk_level = "Low", "Low"
            elif proba < 0.5:
                confidence, risk_level = "Medium", "Medium"
            elif proba < 0.8:
                confidence, risk_level = "High", "High"
            else:
                confidence, risk_level = "High", "Critical"

            reasons = []
            try:
                import shap
                explainer = shap.TreeExplainer(self.best_model)
                sv = explainer.shap_values(X)
                feat_names = X.columns.tolist()
                top_indices = np.argsort(np.abs(sv[0]))[-3:][::-1]
                for ti in top_indices:
                    val = float(sv[0][ti])
                    reasons.append({
                        "feature": feat_names[ti],
                        "direction": "increases fraud risk" if val > 0 else "decreases fraud risk",
                        "impact_score": round(abs(val), 4),
                    })
            except Exception:
                reasons = [{"feature": "model_score", "direction": "increases fraud risk", "impact_score": round(proba, 4)}]

            return {
                "fraud_probability": round(proba, 4),
                "is_fraud": is_fraud,
                "confidence": confidence,
                "model_used": self.best_model_name,
                "reasons": reasons,
                "risk_level": risk_level,
            }
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            return {
                "fraud_probability": 0.0,
                "is_fraud": False,
                "confidence": "Low",
                "model_used": self.best_model_name or "None",
                "reasons": [{"feature": "error", "direction": str(e), "impact_score": 0.0}],
                "risk_level": "Low",
            }
