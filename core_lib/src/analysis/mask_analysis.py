from collections import defaultdict
import numpy as np
import pandas as pd

def collapse_mask(feature_names, mask_vals, agg="mean"):
    buckets = defaultdict(list)
    for n, v in zip(feature_names, mask_vals):
        key = "_".join(n.split("_")[1:]) if n.startswith("Atom") else n
        buckets[key].append(float(v))

    rows = []
    for k, vals in buckets.items():
        score = np.mean(vals) if agg == "mean" else np.max(vals)
        rows.append((k, score, len(vals)))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

def save_collapsed_csv(rows, path):
    df = pd.DataFrame(rows, columns=["Feature", "Mask_Value", "Count"])
    df.to_csv(path, index=False)
