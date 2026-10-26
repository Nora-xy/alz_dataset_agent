"""One-shot probe: how fast is RDD parse and what does it contain for key vars."""
import sys, time
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alz_agent.catalog.nacc_rdd_parser import parse_nacc_rdd

with open("config/paths.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

rdd_path = Path(cfg["nacc"]["rdd_pdf"])
print(f"RDD PDF: {rdd_path}  exists={rdd_path.exists()}")

t0 = time.time()
rdd = parse_nacc_rdd(rdd_path)
dt = time.time() - t0
print(f"RDD parsed: {len(rdd)} entries in {dt:.1f}s")

for k in ["NACCAGE", "NACCMMSE", "CVHATT", "NACCNMRI"]:
    v = rdd.get(k, {})
    concept = v.get("concept", None)
    desc = str(v.get("description", ""))[:80]
    coding = dict(list(v.get("value_coding", {}).items())[:4])
    print(f"  {k}: concept={concept!r}  desc={desc!r}  coding={coding}")
