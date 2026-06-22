"""Copy this into one Colab code cell to confirm loop1 direct preservation."""

import base64
import json
import time
import urllib.request

from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = "main"
CELL_PATH = "colab/STAGE5_DIRECT_PRESERVATION_CONFIRM_CELL.py"

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
assert "STAGE5_DIRECT_PRESERVATION_CONFIRM_CELL_VERSION" in code
assert "STAGE5_BENCHMARK_MAX_LOOPS" in code

exec(compile(code, CELL_PATH, "exec"))
