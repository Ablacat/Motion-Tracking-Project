"""Copy cached RTMLib models or download pinned official checkpoints. Standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "detector.onnx": {
        "checkpoint": "yolox_m_8xb8-300e_humanart-c2c7a14a",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "sha256": "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23",
    },
    "pose.onnx": {
        "checkpoint": "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        "sha256": "26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf",
    },
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/rtmlib/hub/checkpoints")
    args = parser.parse_args()
    assets = ROOT / "app/src/main/assets/models"
    assets.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, model in MODELS.items():
        target = assets / name
        if target.is_file() and digest(target) == model["sha256"]:
            print(f"Verified {name}")
        else:
            cached = args.cache / (model["checkpoint"] + ".onnx")
            if cached.is_file() and digest(cached) == model["sha256"]:
                shutil.copyfile(cached, target)
                print(f"Copied {name} from RTMLib cache")
            else:
                with tempfile.TemporaryDirectory(dir=assets) as temporary:
                    archive = Path(temporary) / "checkpoint.zip"
                    print(f"Downloading {name} from OpenMMLab…", flush=True)
                    try:
                        urllib.request.urlretrieve(model["url"], archive)
                    except OSError:
                        mirror = model["url"].replace("https://download.openmmlab.com/mmpose/v1/projects/", "https://huggingface.co/Tau-J/RTMPose/resolve/main/")
                        urllib.request.urlretrieve(mirror, archive)
                    with zipfile.ZipFile(archive) as bundle:
                        candidates = [item for item in bundle.namelist() if item.endswith("end2end.onnx")]
                        if len(candidates) != 1:
                            raise RuntimeError(f"Unexpected checkpoint layout for {name}")
                        staged = Path(temporary) / name
                        # Stream one member instead of extracting untrusted ZIP paths.
                        with bundle.open(candidates[0]) as source, staged.open("wb") as output:
                            shutil.copyfileobj(source, output)
                    if digest(staged) != model["sha256"]:
                        raise RuntimeError(f"SHA-256 mismatch for {name}; refusing to package it")
                    shutil.copyfile(staged, target)
        manifest[name] = {"sha256": model["sha256"], "bytes": target.stat().st_size, "source": model["url"]}
    (assets / "models.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Models ready. Total model size:", sum(m["bytes"] for m in manifest.values()), "bytes")


if __name__ == "__main__":
    main()
