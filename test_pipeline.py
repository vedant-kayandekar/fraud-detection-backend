"""Quick pipeline test."""
import pandas as pd
from pipeline.cleaner import DataCleaner
from pipeline.features import FeatureEngineer
from pipeline.model import FraudDetector

df = pd.read_csv('../../sample.csv')
print(f"Loaded {len(df)} rows")

c = DataCleaner()
df_clean, qr = c.clean(df)
print(f"Cleaned: {len(df_clean)} rows")

fe = FeatureEngineer()
df_feat = fe.engineer_features(df_clean)
print(f"Features: {len(df_feat)} rows, {len(df_feat.columns)} cols")

d = FraudDetector()
results = d.detect(df_feat)
print(f"Fraud: {results['fraud_count']}/{results['total_processed']}")
print(f"Best model: {results['best_model_name']} F1={results['best_model_f1']}")
print(f"Models compared: {len(results['model_comparison'])}")
for m in results['model_comparison']:
    print(f"  {m['model_name']}: F1={m['f1_score']} {'<< BEST' if m['is_best'] else ''}")
print("SUCCESS")
