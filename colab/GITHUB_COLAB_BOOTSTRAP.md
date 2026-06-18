# GitHub-backed Colab bootstrap

Use this flow instead of uploading zip files. GitHub is the source of truth for
code, while Google Drive or Hugging Face Hub stores checkpoints and datasets.

## Colab secrets

Create these Colab secrets:

- `GH_TOKEN`: fine-grained GitHub token scoped only to this private repo.
- `HF_TOKEN`: Hugging Face token for model and dataset downloads.

## Private repo clone cell

```python
%cd /content

import os
from pathlib import Path
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
PROJECT = Path("/content/recurrent-qwen-svgd")

gh = userdata.get("GH_TOKEN")
assert gh, "Set GH_TOKEN in Colab Secrets."

repo_url = f"https://x-access-token:{gh}@github.com/{REPO}.git"

if PROJECT.exists():
    %cd /content/recurrent-qwen-svgd
    !git pull
else:
    !git clone {repo_url} {PROJECT}
    %cd /content/recurrent-qwen-svgd

!pip -q install -r requirements.txt
```

## Hugging Face auth cell

```python
import os
from google.colab import userdata
from huggingface_hub import HfApi, login

token = userdata.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
assert token, "Set HF_TOKEN in Colab Secrets."

os.environ["HF_TOKEN"] = token
os.environ["HUGGINGFACE_HUB_TOKEN"] = token
login(token=token, add_to_git_credential=False)
print("HF auth OK:", HfApi(token=token).whoami().get("name"))
```

## Update code during a session

```python
%cd /content/recurrent-qwen-svgd
!git pull
!pytest -q tests
```

Checkpoints stay outside git:

- local Colab runtime: `outputs/...`
- persisted artifacts: Google Drive or a private Hugging Face model repo
