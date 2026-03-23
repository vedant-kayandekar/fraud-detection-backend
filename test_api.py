"""Check response structure."""
import httpx
import json

with open("../../sample.csv", "rb") as f:
    r = httpx.post(
        "http://localhost:8000/api/v1/analyze",
        files={"file": ("sample.csv", f, "text/csv")},
        timeout=300,
    )

print(f"Status: {r.status_code}")
d = r.json()
print(f"Top-level keys: {list(d.keys())}")
print(f"fraud_results keys: {list(d['fraud_results'].keys())}")
mc = d['fraud_results'].get('model_comparison', [])
print(f"model_comparison length: {len(mc)}")
if mc:
    print(f"First model: {mc[0]['model_name']} F1={mc[0]['f1_score']}")
print(f"best_model_name: {d['fraud_results'].get('best_model_name', 'MISSING')}")
print(f"best_model_f1: {d['fraud_results'].get('best_model_f1', 'MISSING')}")
print("DONE")
