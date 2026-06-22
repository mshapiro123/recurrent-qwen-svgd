"""Copy this into one CPU Colab cell to recover backed-up direct-preservation results."""

import base64
import json
import time
import urllib.request

from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = "main"
CELL_PATH = "colab/RECOVER_DIRECT_PRESERVATION_RESULTS_CELL.py"

gh = userdata.get("GH_TOKEN") or userdata.get("GITHUB_TOKEN")
assert gh, "Missing GH_TOKEN in Colab secrets."

url = (
    f"https://api.github.com/repos/{REPO}/contents/{CELL_PATH}"
    f"?ref={REF}&cache_bust={int(time.time())}"
)
req = urllib.request.Request(
    url,
    headers={
        "Authorization": f"Bearer {gh}",
        "Accept": "application/vnd.github+json",
        "Cache-Control": "no-cache",
    },
)

payload = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
code = base64.b64decode(payload["content"]).decode("utf-8")

print("Fetched:", CELL_PATH)
print("sha:", payload.get("sha"))
assert "DIRECT PRESERVATION RESULT" in code
assert "stage5_direct_preservation_loop1_" in code

exec(compile(code, CELL_PATH, "exec"))
