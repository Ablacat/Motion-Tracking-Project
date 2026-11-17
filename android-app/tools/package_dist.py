"""Stage the current APK and create a source ZIP without local settings or model binaries."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abi", choices=("arm64-v8a", "x86_64"), required=True)
    args = parser.parse_args()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    apk = ROOT / "app/build/outputs/apk/debug/app-debug.apk"
    with zipfile.ZipFile(apk) as bundle:
        names = bundle.namelist()
        assert "assets/models/detector.onnx" in names
        assert "assets/models/pose.onnx" in names
        native = [n for n in names if n.startswith("lib/") and n.endswith(".so")]
        assert native and all(n.split("/")[1] == args.abi for n in native), "Wrong APK architecture"
    name = "MotionTracker-S23-debug.apk" if args.abi == "arm64-v8a" else "MotionTracker-emulator-debug.apk"
    shutil.copy2(apk, dist / name)
    tests = ROOT / "app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
    if tests.is_file():
        shutil.copy2(tests, dist / "MotionTracker-device-tests.apk")
    for document in ("README.md", "VALIDATION.md"):
        shutil.copy2(ROOT / document, dist / document)
    excluded_dirs = {"build", ".gradle", ".idea", "dist", "__pycache__", ".git"}
    excluded_files = {"local.properties", "release-signing.properties"}
    with zipfile.ZipFile(dist / "MotionTracker-source.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for directory, subdirs, filenames in os.walk(ROOT):
            subdirs[:] = [d for d in subdirs if d not in excluded_dirs]
            for filename in filenames:
                path = Path(directory) / filename
                if filename in excluded_files or path.suffix in (".onnx", ".jks", ".keystore", ".pyc", ".log"):
                    continue
                bundle.write(path, Path("MotionTracker") / path.relative_to(ROOT))
    checksums = []
    for path in sorted(dist.iterdir()):
        if path.suffix in (".apk", ".zip"):
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            checksums.append(f"{sha}  {path.name}")
    (dist / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    print(f"Staged {name} and source ZIP in {dist}")


if __name__ == "__main__":
    main()
