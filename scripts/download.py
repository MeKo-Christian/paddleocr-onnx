#!/usr/bin/env python3
"""
Download PP-OCRv5 models - individual or all at once.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Import configurations from the existing export script
from export import PPOCRV5_CONFIGS


def download_file(url: str, local_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        print(f"Downloading {local_path.name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"✓ Downloaded: {local_path}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def setup_paddleocr_repo(repo_path: Path) -> bool:
    """Setup PaddleOCR repository."""
    if repo_path.exists() and (repo_path / "tools" / "export_model.py").exists():
        print("✓ PaddleOCR repository already exists")
        return True

    print("Setting up PaddleOCR repository...")
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "https://github.com/PaddlePaddle/PaddleOCR.git", str(repo_path)]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("✓ PaddleOCR repository cloned")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone repository: {e}")
        return False


def download_pretrained_model(model_name: str, models_dir: Path) -> Optional[Path]:
    """Download pretrained model."""
    if model_name not in PPOCRV5_CONFIGS:
        print(f"Unknown model: {model_name}")
        return None

    config = PPOCRV5_CONFIGS[model_name]
    filename = Path(config["pretrained_url"]).name
    local_path = models_dir / "pretrained" / filename

    if local_path.exists():
        print(f"✓ Model already exists: {local_path}")
        return local_path

    success = download_file(config["pretrained_url"], local_path)
    return local_path if success else None


def download_single_model(model_name: str, base_dir: Path = Path("official_pipeline")) -> bool:
    """Download a single pretrained model."""
    if model_name not in PPOCRV5_CONFIGS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(PPOCRV5_CONFIGS.keys())}")
        return False

    print(f"Downloading {model_name}...")

    # Setup paths
    repo_path = base_dir / "PaddleOCR"
    models_dir = base_dir / "models"

    # Setup repository first
    if not setup_paddleocr_repo(repo_path):
        print("✗ Failed to setup PaddleOCR repository")
        return False

    # Download the specific model
    pretrained_path = download_pretrained_model(model_name, models_dir)
    if not pretrained_path:
        print(f"✗ Failed to download {model_name}")
        return False

    print(f"✓ {model_name} downloaded successfully")
    return True


def download_all_models(base_dir: Path = Path("official_pipeline")) -> bool:
    """Download all pretrained models."""
    print("Downloading all pretrained models...")

    repo_path = base_dir / "PaddleOCR"
    models_dir = base_dir / "models"

    # Setup repository first
    if not setup_paddleocr_repo(repo_path):
        return False

    success = True
    for model_name in PPOCRV5_CONFIGS.keys():
        print(f"\nDownloading {model_name}...")
        pretrained_path = download_pretrained_model(model_name, models_dir)
        if not pretrained_path:
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download PP-OCRv5 models"
    )

    parser.add_argument("model", nargs='?', choices=list(PPOCRV5_CONFIGS.keys()) + ['all'],
                        help="Model to download or 'all' for all models")
    parser.add_argument("--all", action="store_true",
                        help="Download all models")
    parser.add_argument("--base-dir", type=Path, default=Path("official_pipeline"),
                        help="Base working directory")

    args = parser.parse_args()

    if args.all or args.model == 'all':
        success = download_all_models(args.base_dir)
    elif args.model:
        success = download_single_model(args.model, args.base_dir)
    else:
        parser.print_help()
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()