import synapseclient
import os
import sys

syn = synapseclient.Synapse()
syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"], silent=True)
print("Synapse login OK")

# Probe candidate IDs
for syn_id in ["syn29266807", "syn28546456", "syn29264862", "syn51514105", "syn25829067"]:
    try:
        ent = syn.get(syn_id, downloadFile=False)
        name = getattr(ent, "name", "?")
        etype = type(ent).__name__
        size_bytes = getattr(ent, "contentSize", None)
        size_str = f"{size_bytes/1e9:.2f}GB" if size_bytes else "N/A (folder/project)"
        print(f"  {syn_id}: {name} [{etype}] {size_str}")
    except Exception as e:
        print(f"  {syn_id}: FAILED — {e}")

# Also try listing children of the most promising ones
print("\nChildren of syn28546456:")
try:
    children = list(syn.getChildren("syn28546456", includeTypes=["file", "folder"]))[:15]
    for c in children:
        csize = c.get("contentSize", "")
        csize_str = f" ({csize/1e9:.2f}GB)" if csize else ""
        print(f"  {c['id']}: {c['name']} [{c['type'].split('.')[-1]}]{csize_str}")
except Exception as e:
    print(f"  Failed: {e}")
