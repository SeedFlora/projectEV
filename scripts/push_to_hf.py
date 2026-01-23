import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to model folder.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. username/model-name.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token or set HF_TOKEN.")
    parser.add_argument("--private", action="store_true", help="Create private repo.")

    args = parser.parse_args()

    if not args.token:
        raise ValueError("HF token missing. Set --token or HF_TOKEN env var.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    api = HfApi(token=args.token)
    create_repo(args.repo_id, private=args.private, exist_ok=True, token=args.token)

    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(model_dir),
        token=args.token,
        commit_message="Upload fine-tuned model",
    )

    print("Uploaded model to", args.repo_id)


if __name__ == "__main__":
    main()
