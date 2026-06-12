import synapseclient
import os

syn = synapseclient.Synapse()
syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"], silent=True)

# Check Data folder (syn29264504) — likely contains partition CSV
print("Contents of Data folder (syn29264504):")
try:
    children = list(syn.getChildren("syn29264504"))
    for c in children[:20]:
        print(f"  {c['id']}: {c['name']} [{c['type'].split('.')[-1]}]")
    if len(children) > 20:
        print(f"  ... and {len(children)-20} more")
except Exception as e:
    print(f"  Failed: {e}")

# Check Images folder (syn29302789)
print("\nContents of Images folder (syn29302789) — first 10:")
try:
    children_img = list(syn.getChildren("syn29302789"))
    for c in children_img[:10]:
        print(f"  {c['id']}: {c['name']} [{c['type'].split('.')[-1]}]")
    print(f"  Total: {len(children_img)} items")
except Exception as e:
    print(f"  Failed: {e}")

# Try getting the zip metadata directly
print("\nZip file metadata (syn29266807):")
try:
    ent = syn.get("syn29266807", downloadFile=False)
    print(f"  Name: {ent.name}")
    print(f"  Content type: {getattr(ent, 'contentType', 'unknown')}")
    print(f"  Size: {getattr(ent, 'contentSize', 'unknown')}")
    print(f"  MD5: {getattr(ent, 'md5', 'unknown')}")
except Exception as e:
    print(f"  Failed: {e}")

# Search Data folder for CSV files
print("\nLooking for CSVs in Data folder:")
try:
    children_data = list(syn.getChildren("syn29264504"))
    csv_files = [c for c in children_data if c["name"].endswith(".csv")]
    for c in csv_files:
        print(f"  FOUND CSV: {c['id']}: {c['name']}")
    if not csv_files:
        print("  No CSVs found — listing all files:")
        for c in children_data[:20]:
            print(f"    {c['id']}: {c['name']}")
except Exception as e:
    print(f"  Failed: {e}")
