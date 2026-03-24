"""Test pipeline to verify fraud count hits target (154)."""
import httpx
import time

print("Target fraud count: 154")
print("=== Analyzing CSV ===")
start = time.time()
with open("../../sample.csv", "rb") as f:
    r = httpx.post(
        "http://localhost:8000/api/v1/analyze",
        files={"file": ("sample.csv", f, "text/csv")},
        timeout=300,
    )
elapsed = time.time() - start

print(f"Status: {r.status_code}")
if r.status_code == 200:
    d = r.json()
    fc = d['fraud_results']['fraud_count']
    total = d['fraud_results']['total_processed']
    print(f"Fraud count: {fc} out of {total} ({fc/total*100:.1f}%)")
    print(f"Difference from target: {fc - 154}")
    print(f"Track A time: {elapsed:.1f}s")
    
    # Let's peek at the top reasons
    reasons = {}
    for row in d['fraud_results']['fraud_rows']:
        for r in row['shap_reasons']:
            feat = r['feature']
            reasons[feat] = reasons.get(feat, 0) + 1
            
    print("\nTop 5 fraud indicators (SHAP):")
    for feat, count in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat}: {count} times")
else:
    print(r.text)
