"""Optional check: Spearman of the volume term V alone vs the 12 expert junctions, next to full PCIS."""
import pandas as pd
from scipy.stats import spearmanr

from parksight import config as C

j = pd.read_parquet(C.ROOT / "parksight/data/processed/junction_pcis.parquet")
gt = pd.DataFrame(C.PCIS_GROUND_TRUTH, columns=["code", "label", "tier", "expert_rank"])
gt["junction_name"] = [j.junction_name[j.junction_name.str.contains(c)].iloc[0] for c in gt.code]
m = gt.merge(j, on="junction_name")
out = pd.DataFrame([dict(score=s, n=len(m), spearman=spearmanr(-m.expert_rank, m[s])[0])
                    for s in ("V", "PCIS")])  # expert rank 1 = most congested, so negate
(C.ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
out.to_csv(C.ROOT / "results/tables/PCIS_volume_check.csv", index=False)
print(out)
