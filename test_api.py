"""Test two-track parallel pipeline."""
import httpx
import time

# Track A: Fast path
print("=== Track A: Analyzing CSV ===")
start = time.time()
with open("../../sample.csv", "rb") as f:
    r = httpx.post(
        "http://localhost:8000/api/v1/analyze",
        files={"file": ("sample.csv", f, "text/csv")},
        timeout=300,
    )
elapsed_a = time.time() - start

print(f"Status: {r.status_code}")
d = r.json()
print(f"Track A time: {elapsed_a:.1f}s")
print(f"Fraud: {d['fraud_results']['fraud_count']}/{d['fraud_results']['total_processed']}")
print(f"Best model: {d['fraud_results']['best_model_name']}")
print(f"F1: {d['fraud_results']['best_model_f1']}")
print(f"Models in response: {len(d['fraud_results']['model_comparison'])}")

job_id = d.get('job_id')
print(f"Job ID: {job_id}")

# Track B: Poll comparison
if job_id:
    print(f"\n=== Track B: Polling comparison ===")
    for i in range(12):
        time.sleep(3)
        r2 = httpx.get(f"http://localhost:8000/api/v1/comparison/{job_id}", timeout=10)
        comp = r2.json()
        status = comp['status']
        done = [m['model_name'] for m in comp.get('models', []) if m.get('status') == 'complete']
        pending = [m['model_name'] for m in comp.get('models', []) if m.get('status') == 'processing']
        elapsed = time.time() - start
        print(f"  [{elapsed:.0f}s] status={status}, done={done}, pending={pending}")
        if status == 'complete':
            print(f"\n  Overall best: {comp['best_model_name']} (F1={comp['best_model_f1']})")
            for m in comp['models']:
                s = m.get('status', '?')
                f1 = m.get('f1_score', 0)
                t = m.get('training_time_seconds', 0)
                best = ' << BEST' if m.get('is_best') else ''
                print(f"    {m['model_name']}: status={s} F1={f1} time={t}s{best}")
            break
    print(f"\n  Total time: {time.time()-start:.1f}s")

print("\nDONE")
