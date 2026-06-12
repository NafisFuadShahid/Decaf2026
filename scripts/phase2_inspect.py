import pickle, json, numpy as np
from collections import defaultdict

with open("C:/DeCaf/fed_crc_results/volume_scores.pkl","rb") as f:
    vs = pickle.load(f)

with open("C:/DeCaf/fed_crc_results/step5_crc_results.json") as f:
    step5 = json.load(f)

sample_key = list(vs.keys())[0]
sample = vs[sample_key]
print(f"Subjects: {len(vs)}")
print(f"FNR curve length: {len(sample['fnr_curve'])}")
print(f"Sample keys: {list(sample.keys())}")
print(f"Sample site_id: {sample['site_id']}")
print(f"Phase 1 lambda_grid (len={len(step5['lambda_grid'])}): {step5['lambda_grid']}")
print(f"B3 lambda*: {step5['lambda_b3']}")
print(f"Ours_per_site lambdas: {step5['lambda_ours_per_site']}")

sites = defaultdict(int)
for v in vs.values():
    sites[v["site_id"]] += 1
print(f"Sites: {dict(sorted(sites.items()))}")
