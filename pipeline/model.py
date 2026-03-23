"""
ML Model Module for FraudGuard Pipeline.

Multi-model fraud detection system:
  Stage 1: IsolationForest generates pseudo-labels (unsupervised)
  Stage 2: 4 models trained on pseudo-labels, auto-select best F1
  Stage 3: RandomizedSearchCV hyperparameter tuning on best model
  Stage 4: SHAP explainability on best model
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    # 5 combined features from improvement pass
    'risk_score_composite', 'amount_balance_ratio',
    'user_city_consistency', 'velocity_amount_product', 'night_high_spend',
]

WHY_USED = {
    "XGBoost": "Gradient boosting that corrects its own errors sequentially. Best at catching subtle multi-signal fraud patterns.",
    "Random Forest": "Ensemble of independent trees averaged together. Robust and resistant to overfitting.",
    "Logistic Regression": "Linear baseline model. Fast and interpretable but struggles with complex non-linear fraud patterns.",
    "Decision Tree": "Single tree with explicit if-then rules. Fully transparent but prone to overfitting on noisy data.",
}

PARAM_DISTRIBUTIONS = {
    "XGBoost": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.05, 0.1, 0.15, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
}


def _build_models(scale_pos_weight: float) -> Dict[str, Any]:
    """
    Build the 4 model instances with appropriate class weights.

    Args:
        scale_pos_weight: Ratio of non-fraud to fraud for class balancing.

    Returns:
        Dict mapping model name to sklearn estimator.
    """
    import xgboost as xgb

    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight='balanced',
            random_state=42,
        ),
    }


class FraudDetector:
    """
    Multi-model fraud detector.

    Stage 1: IsolationForest pseudo-labels.
    Stage 2: 4 models trained and compared on pseudo-labels.
    Stage 3: Best F1 model selected + optional hyperparameter tuning.
    Stage 4: SHAP explainability on best model.
    """

    def __init__(self):
        """Initialize FraudDetector with no trained models."""
        self.best_model = None
        self.best_model_name: str = ""
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
        # Convert category columns to numeric codes for sklearn
        for col in X.select_dtypes(include=['category']).columns:
            X[col] = X[col].cat.codes
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        # Ensure all float for model compatibility
        for col in X.columns:
            if X[col].dtype == 'float16':
                X[col] = X[col].astype('float32')
        return X

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the full multi-model fraud detection pipeline.

        Args:
            df: DataFrame with all engineered features (from FeatureEngineer).

        Returns:
            Dict with fraud_count, fraud_rate, total_processed,
            model_comparison list, best_model_name, best_model_f1,
            tuned_model_metrics, fraud_rows, feature_importance, shap_summary.
        """
        self.last_df_features = df.copy()
        X_iso = self._prepare_features(df, IF_FEATURES)

        # ── Stage 1: Isolation Forest ──
        logger.info("Stage 1: Running IsolationForest for pseudo-labels...")
        self.iso_model = IsolationForest(
            contamination=0.08,
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        )
        iso_labels = self.iso_model.fit_predict(X_iso)
        df['pseudo_fraud'] = (iso_labels == -1).astype(int)

        # ── Stage 2: Train & compare 4 models ──
        logger.info("Stage 2: Training and comparing 4 models...")
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

        models = _build_models(scale_pos)
        model_comparison = []
        trained_models = {}
        best_f1 = -1.0
        best_name = ""

        for name, model in models.items():
            try:
                logger.info(f"  Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred.astype(float)

                acc = float(accuracy_score(y_test, y_pred))
                prec = float(precision_score(y_test, y_pred, zero_division=0))
                rec = float(recall_score(y_test, y_pred, zero_division=0))
                f1 = float(f1_score(y_test, y_pred, zero_division=0))
                try:
                    auc = float(roc_auc_score(y_test, y_proba))
                except ValueError:
                    auc = 0.0

                trained_models[name] = model
                entry = {
                    "model_name": name,
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "roc_auc": round(auc, 4),
                    "is_best": False,
                    "why_used": WHY_USED.get(name, ""),
                }
                model_comparison.append(entry)

                if f1 > best_f1:
                    best_f1 = f1
                    best_name = name

            except Exception as e:
                logger.warning(f"  {name} training failed: {e}")
                model_comparison.append({
                    "model_name": name,
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                    "f1_score": 0.0, "roc_auc": 0.0,
                    "is_best": False,
                    "why_used": WHY_USED.get(name, ""),
                })

        # Mark best model
        for entry in model_comparison:
            if entry["model_name"] == best_name:
                entry["is_best"] = True

        self.best_model = trained_models.get(best_name)
        self.best_model_name = best_name
        logger.info(f"  Best model: {best_name} (F1={best_f1:.4f})")

        # ── Stage 3: Hyperparameter tuning on best model ──
        tuned_metrics = None
        f1_before = best_f1
        if best_name in PARAM_DISTRIBUTIONS and self.best_model is not None:
            try:
                logger.info(f"Stage 3: Tuning {best_name} with RandomizedSearchCV...")
                tuner = RandomizedSearchCV(
                    self.best_model,
                    PARAM_DISTRIBUTIONS[best_name],
                    n_iter=20,
                    cv=5,
                    scoring='f1',
                    random_state=42,
                    n_jobs=-1,
                )
                tuner.fit(X_train, y_train)
                self.best_model = tuner.best_estimator_
                trained_models[best_name] = self.best_model

                # Re-evaluate tuned model
                y_pred_tuned = self.best_model.predict(X_test)
                f1_after = float(f1_score(y_test, y_pred_tuned, zero_division=0))

                tuned_metrics = {
                    "f1_before_tuning": round(f1_before, 4),
                    "f1_after_tuning": round(f1_after, 4),
                    "improvement": round(f1_after - f1_before, 4),
                    "best_params": {k: (v if not isinstance(v, (np.integer, np.floating)) else
                                        int(v) if isinstance(v, np.integer) else float(v))
                                    for k, v in tuner.best_params_.items()},
                }

                # Update comparison entry
                for entry in model_comparison:
                    if entry["model_name"] == best_name:
                        entry["f1_score"] = round(f1_after, 4)
                        prec_t = float(precision_score(y_test, y_pred_tuned, zero_division=0))
                        rec_t = float(recall_score(y_test, y_pred_tuned, zero_division=0))
                        acc_t = float(accuracy_score(y_test, y_pred_tuned))
                        entry["accuracy"] = round(acc_t, 4)
                        entry["precision"] = round(prec_t, 4)
                        entry["recall"] = round(rec_t, 4)

                logger.info(f"  Tuning done. F1: {f1_before:.4f} → {f1_after:.4f}")
            except Exception as e:
                logger.warning(f"  Hyperparameter tuning failed: {e}")
        else:
            logger.info("Stage 3: Skipping tuning (not applicable for this model type)")

        # ── Predict on full dataset with best model ──
        if self.best_model is not None:
            fraud_proba = self.best_model.predict_proba(X_all)[:, 1]
        else:
            fraud_proba = np.zeros(len(df))
        df['fraud_probability'] = fraud_proba
        df['predicted_fraud'] = (fraud_proba > 0.5).astype(int)

        # Feature importance (for tree models)
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            self.feature_importance = {
                name: round(float(imp), 4)
                for name, imp in sorted(
                    zip(feat_names, importance),
                    key=lambda x: x[1], reverse=True
                )
            }
        elif hasattr(self.best_model, 'coef_'):
            coefs = np.abs(self.best_model.coef_[0])
            self.feature_importance = {
                name: round(float(imp), 4)
                for name, imp in sorted(
                    zip(feat_names, coefs),
                    key=lambda x: x[1], reverse=True
                )
            }
        else:
            self.feature_importance = {}

        # ── Stage 4: SHAP on best model ──
        shap_summary = {}
        shap_reasons_map = {}
        try:
            import shap
            logger.info("Stage 4: Computing SHAP values on best model...")

            fraud_mask = df['fraud_probability'] > 0.5
            X_fraud = X_all[fraud_mask]

            if len(X_fraud) > 2000:
                X_fraud = X_fraud.sample(2000, random_state=42)

            if best_name in ["XGBoost", "Random Forest", "Decision Tree"]:
                explainer = shap.TreeExplainer(self.best_model)
            else:
                explainer = shap.LinearExplainer(self.best_model, X_train)

            shap_values = explainer.shap_values(X_fraud)

            # Handle multi-output SHAP for RF/DT
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # class 1 = fraud

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_summary = {
                name: round(float(val), 4)
                for name, val in zip(feat_names, mean_abs_shap)
            }

            # Per-row top 3 SHAP reasons
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

        result = {
            "fraud_count": total_fraud,
            "fraud_rate": round(total_fraud / max(total, 1) * 100, 2),
            "total_processed": total,
            "precision": round(model_comparison[0]["precision"] if model_comparison else 0.0, 4),
            "recall": round(model_comparison[0]["recall"] if model_comparison else 0.0, 4),
            "f1_score": round(best_f1, 4),
            "fraud_rows": fraud_rows,
            "feature_importance": self.feature_importance,
            "shap_summary": shap_summary,
            "model_comparison": model_comparison,
            "best_model_name": best_name,
            "best_model_f1": round(best_f1, 4),
            "tuned_model_metrics": tuned_metrics,
        }

        # Use best model's metrics for top-level
        for entry in model_comparison:
            if entry["is_best"]:
                result["precision"] = entry["precision"]
                result["recall"] = entry["recall"]
                result["f1_score"] = entry["f1_score"]
                break

        logger.info(f"Fraud detection complete. {total_fraud}/{total} flagged. Best: {best_name}")
        return result

    def predict_single(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud probability for a single transaction.

        Uses the best model from the last detect() call.

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
                confidence = "Low"
            elif proba <= 0.7:
                confidence = "Medium"
            else:
                confidence = "High"

            if proba < 0.3:
                risk_level = "Low"
            elif proba < 0.5:
                risk_level = "Medium"
            elif proba < 0.8:
                risk_level = "High"
            else:
                risk_level = "Critical"

            # SHAP reasons
            reasons = []
            try:
                import shap
                feat_names = X.columns.tolist()
                if self.best_model_name in ["XGBoost", "Random Forest", "Decision Tree"]:
                    explainer = shap.TreeExplainer(self.best_model)
                else:
                    explainer = shap.LinearExplainer(self.best_model, self._X_train)
                sv = explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]
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
