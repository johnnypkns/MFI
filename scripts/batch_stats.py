"""Model accuracy (spatial-CV AUC) for every franchisable brand with >= 8 stores -> data/app/brand_model_stats.parquet"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd
from mfi.model.export import APP
from mfi.model.site_model import load_data, run
d = load_data()
reg = pd.read_parquet(APP / "brands.parquet")
todo = reg[reg.franchisable & (reg.n_stores >= 8)].sort_values("n_stores", ascending=False)
out, f = [], APP / "brand_model_stats.parquet"
for i, r in enumerate(todo.itertuples()):
    t = time.time()
    try:
        R = run(d, r.brand, 3)
        out.append({"brand": r.brand, "auc": R.metrics["auc_spatial_cv"], "reliability": R.metrics["reliability"],
                    "stores_modelled": R.metrics["stores"]})
    except Exception as e:
        out.append({"brand": r.brand, "auc": None, "reliability": "error", "stores_modelled": 0, "error": str(e)[:200]})
    if i % 10 == 0:
        pd.DataFrame(out).to_parquet(f, index=False)
    print(i, r.brand, out[-1]["auc"], f"{time.time()-t:.0f}s", flush=True)
pd.DataFrame(out).to_parquet(f, index=False)
print("done", len(out))
