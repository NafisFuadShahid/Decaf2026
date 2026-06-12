"""Try all known locations for the FeTS-2022 partition CSV."""
import urllib.request
import os, sys, io
from pathlib import Path

if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT = Path("C:/DeCaf/data")
OUT.mkdir(parents=True, exist_ok=True)

urls = [
    # FeTS-AI GitHub (various spellings)
    "https://raw.githubusercontent.com/FeTS-AI/Challenge/main/Task_1/partitioning_1.csv",
    "https://raw.githubusercontent.com/FeTS-AI/Challenge/master/Task_1/partitioning_1.csv",
    "https://raw.githubusercontent.com/FETS-AI/Challenge/main/Task_1/partitioning_1.csv",
    "https://raw.githubusercontent.com/FETS-AI/Challenge/main/Task_1/FeTS2022_partitioning_1.csv",
    "https://raw.githubusercontent.com/FETS-AI/Challenge/main/partitioning_1.csv",
    "https://raw.githubusercontent.com/cbica/CaPTk/master/src/applications/FeTS/data/partitioning_1.csv",
    # Direct GitHub API listing
]

found = False
for url in urls:
    try:
        save = str(OUT / "partitioning_1.csv")
        urllib.request.urlretrieve(url, save)
        with open(save) as f:
            content = f.read()
        if len(content) > 100 and "," in content:
            import pandas as pd
            df = pd.read_csv(save)
            if len(df) > 50:
                print(f"SUCCESS: {url}")
                print(f"  Rows: {len(df)}, Cols: {df.columns.tolist()}")
                print(df.head(3).to_string())
                found = True
                break
            else:
                print(f"  Too small ({len(df)} rows): {url}")
        else:
            print(f"  Not a CSV: {url}")
    except Exception as e:
        print(f"  FAIL {url}: {e}")

if not found:
    print("\nAll GitHub attempts failed.")
    print("FeTS partition CSV requires Synapse download access.")
    print("\nTo get real partitions, the user must:")
    print("  1. Go to synapse.org -> Profile -> Access Tokens")
    print("  2. Create a new PAT with 'Download' scope enabled")
    print("  3. Set SYNAPSE_AUTH_TOKEN to the new token")
    print("  4. OR accept the FeTS-2022 DUA at https://www.synapse.org/access")
