"""Create and upload the three release repositories as private models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-root", type=Path, required=True)
    parser.add_argument("--namespace")
    args = parser.parse_args()
    api = HfApi()
    profile = api.whoami()
    namespace = args.namespace or str(profile["name"])
    report = {"namespace": namespace, "repos": {}}
    for folder in sorted(path for path in args.repos_root.iterdir() if path.is_dir()):
        repo_id = f"{namespace}/{folder.name}"
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder,
            commit_message="Prepare private Paper One model release",
        )
        info = api.model_info(repo_id)
        if info.private is not True:
            raise RuntimeError(f"Refusing release: {repo_id} is not private")
        report["repos"][folder.name] = {
            "repo_id": repo_id,
            "private": True,
            "commit": str(commit.oid),
            "url": str(commit.repo_url),
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

