"""Install pinned official Linux backup tools, verifying archives before use.

No shell, package manager, credentials, or changes to the bot entrypoint.
Data storage retains the small binaries across normal deployments.
"""
import argparse
import bz2
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import stat
import tempfile
import urllib.request
import zipfile

RELEASES = {
    "restic": ("0.19.1", {
        "amd64": "f415415624dcc452f2a02b8c33641791a8c6d6d3b65bbb3543fcf9a25151585c",
        "arm64": "a5f64aaab53d51e311fa3829124c5b703f2d14cf187d8640b6be3b2b49376465"}),
    "rclone": ("1.75.2", {
        "amd64": "349ac8fba6ff65d6247043f1750cdcb518ec5d500ef91463a10d37c0ccdf3702",
        "arm64": "7e1e8d69654941b7b7df84ee74c5f7fb09cce7d5496947fb7bf16b55ff427d10"}),
}


def install(directory):
    if platform.system() != "Linux":
        raise ValueError("Only Linux hosts are supported")
    arch = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine())
    if arch is None:
        raise ValueError("Unsupported host architecture")
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Tools directory must not be a link")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    for name, (version, hashes) in RELEASES.items():
        target, receipt = directory / name, directory / (name + ".json")
        expected = hashes[arch]
        if target.is_file() and not target.is_symlink() and receipt.is_file():
            previous = json.loads(receipt.read_text())
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if previous == {"version": version, "archive_sha256": expected, "binary_sha256": actual}:
                print(name + ": verified installation already present", flush=True)
                continue
        if name == "restic":
            filename = f"restic_{version}_linux_{arch}.bz2"
            url = f"https://github.com/restic/restic/releases/download/v{version}/{filename}"
        else:
            filename = f"rclone-v{version}-linux-{arch}.zip"
            url = f"https://github.com/rclone/rclone/releases/download/v{version}/{filename}"
        request = urllib.request.Request(url, headers={"User-Agent": "PsychologiaPRO-backup-setup/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(128 * 1024 ** 2 + 1)
        if len(raw) > 128 * 1024 ** 2 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Official archive checksum mismatch")
        if name == "restic":
            decoder = bz2.BZ2Decompressor()
            binary = decoder.decompress(raw, max_length=256 * 1024 ** 2 + 1)
            if not decoder.eof or len(binary) > 256 * 1024 ** 2:
                raise ValueError("Invalid compressed executable")
        else:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                info = archive.getinfo(f"rclone-v{version}-linux-{arch}/rclone")
                if info.file_size > 256 * 1024 ** 2 or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Invalid executable member")
                binary = archive.read(info)
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(binary)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o700)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        receipt.write_text(json.dumps({"version": version, "archive_sha256": expected,
                                      "binary_sha256": hashlib.sha256(binary).hexdigest()}) + "\n")
        receipt.chmod(0o600)
        print(name + ": official download and checksum verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path,
                        default=Path(os.environ.get("BOT_DATA_DIR", "/app/data")) / ".backup-tools")
    args = parser.parse_args()
    try:
        install(args.directory)
    except Exception as error:
        print("Backup tools setup failed:", type(error).__name__)
        raise SystemExit(1) from None
