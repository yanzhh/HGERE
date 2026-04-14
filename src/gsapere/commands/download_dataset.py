#!/usr/bin/env python3
"""Download datasets into the datasets/ directory.

Usage
-----
List available datasets::

    uv run download-dataset --list

Download all files for a dataset::

    uv run download-dataset gsap-ere

Download specific files only::

    uv run download-dataset gsap-ere --files train.jsonl dev.jsonl

Custom output directory::

    uv run download-dataset gsap-ere --outdir /tmp/gsap-ere

Adding a new dataset
--------------------
Register a new entry in DATASETS below.  Set ``files=None`` to mark a dataset
as not yet implemented — the CLI will print a clear error instead of silently
doing nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests
import requests.exceptions

from gsapere.data.vocabulary import convert_jsonl, load_gsap_ere_vocabulary

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass
class FileSpec:
    url: str
    md5: Optional[str] = None


@dataclass
class DatasetSpec:
    name: str
    description: str
    # Maps filename -> FileSpec.  None means "not yet implemented".
    files: Optional[dict[str, FileSpec]]
    # Human-readable pointer shown in error messages (alternative sources,
    # documentation URL, etc.).
    source_hint: str = ""
    # Called with the downloaded file path after each successful download.
    postprocess: Optional[Callable[[Path], None]] = None

    @property
    def implemented(self) -> bool:
        return self.files is not None

    def default_outdir(self) -> Path:
        return Path("datasets") / self.name


_GSAP_ERE_BASE = "https://berd-platform.de/records/c4c1d-s0587/files"
_SCIERC_BASE = "https://cloud.tsinghua.edu.cn/d/7dafc9a3d84d4151a755/files/?p="


def _gsap_ere_postprocess(path: Path) -> None:
    vocab = load_gsap_ere_vocabulary()
    n = convert_jsonl(path, path, vocab)
    print(f"  [vocab] replaced token IDs in {n} documents")


def _scinlp_postprocess(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines = []
    n = 0
    for line in lines:
        if not line.strip():
            continue
        doc = json.loads(line)
        if "doc_key" in doc:
            doc["doc_id"] = doc.pop("doc_key")
            n += 1
        out_lines.append(json.dumps(doc))
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"  [postprocess] renamed doc_key→doc_id in {n} documents")


DATASETS: dict[str, DatasetSpec] = {
    "gsap-ere": DatasetSpec(
        name="gsap-ere",
        description="GSAP-ERE — entity and relation extraction on scientific text",
        source_hint=(
            "Source: https://berd-platform.de/records/c4c1d-s0587\n"
            "DOI:    https://doi.org/10.60914/c4c1d-s0587\n"
            "See documentation/download-dataset.md for details."
        ),
        postprocess=_gsap_ere_postprocess,
        files={
            "dev.jsonl": FileSpec(
                url=f"{_GSAP_ERE_BASE}/dev.jsonl?download=1",
                md5="b3e379d168a21ca371cfaea80b1cbede",
            ),
            "test.jsonl": FileSpec(
                url=f"{_GSAP_ERE_BASE}/test.jsonl?download=1",
                md5="0462bc3719ec2ffbd2951fc4635a45a8",
            ),
            "train.jsonl": FileSpec(
                url=f"{_GSAP_ERE_BASE}/train.jsonl?download=1",
                md5="8466624a14973b2d88d658068c417db3",
            ),
        },
    ),
    "scier": DatasetSpec(
        name="scier",
        description="SciER — scientific entity and relation extraction dataset",
        source_hint=(
            "Source: https://github.com/edzq/SciER (PLM/ folder)\n"
            "See documentation/download-dataset.md for details."
        ),
        files={
            "dev.jsonl": FileSpec(
                url="https://raw.githubusercontent.com/edzq/SciER/main/SciER/PLM/dev.jsonl",
                md5="0858b20a98cdcb8844eac43229a50b93",
            ),
            "test.jsonl": FileSpec(
                url="https://raw.githubusercontent.com/edzq/SciER/main/SciER/PLM/test.jsonl",
                md5="678dc5623538e9c60e1fefefa6f4bd02",
            ),
            "test_ood.jsonl": FileSpec(
                url="https://raw.githubusercontent.com/edzq/SciER/main/SciER/PLM/test_ood.jsonl",
                md5="52d4415db4f02fe7aa21f2a3ae272d67",
            ),
            "train.jsonl": FileSpec(
                url="https://raw.githubusercontent.com/edzq/SciER/main/SciER/PLM/train.jsonl",
                md5="c27f6538a9b3bd117a5030ee2eb7f977",
            ),
        },
    ),
    "scierc": DatasetSpec(
        name="scierc",
        description="SciERC — scientific information extraction benchmark",
        source_hint=(
            "Primary source:  https://cloud.tsinghua.edu.cn/d/7dafc9a3d84d4151a755/\n"
            "Alternative:     https://drive.google.com/drive/folders/1_u6pIe7Dw3Lqy4mF2m1UFqmKmGeM40zS\n"
            "Referenced from: https://github.com/thunlp/PL-Marker\n"
            "See documentation/download-dataset.md for details."
        ),
        files={
            "dev.json": FileSpec(
                url=f"{_SCIERC_BASE}/dev.json&dl=1",
                md5="07993722c007cc000cc3f6d5327a10f9",
            ),
            "test.json": FileSpec(
                url=f"{_SCIERC_BASE}/test.json&dl=1",
                md5="2849b54dfa900d4f55c35dca791c4d61",
            ),
            "train.json": FileSpec(
                url=f"{_SCIERC_BASE}/train.json&dl=1",
                md5="2b61bfa738078d000dd79e0e154079d4",
            ),
        },
    ),
    "scinlp": DatasetSpec(
        name="scinlp",
        description="SciNLP — scientific NLP dataset",
        source_hint=(
            "Source: https://github.com/AKADDC/SciNLP (Dataset/ folder)\n"
            "See documentation/download-dataset.md for details."
        ),
        postprocess=_scinlp_postprocess,
        files={
            "dev.jsonl": FileSpec(
                url="https://raw.githubusercontent.com/AKADDC/SciNLP/main/Dataset/dev.json",
                md5="225912710ca250c7610ecb12b91a530e",
            ),
            "test.jsonl": FileSpec(
                url="https://raw.githubusercontent.com/AKADDC/SciNLP/main/Dataset/test.json",
                md5="141e61feca819c7353f91bc7f55ef029",
            ),
            "train.jsonl": FileSpec(
                url="https://raw.githubusercontent.com/AKADDC/SciNLP/main/Dataset/train.json",
                md5="e92aae561a376127ff3b938be84c65f9",
            ),
        },
    ),
}


# ---------------------------------------------------------------------------
# Core download helpers
# ---------------------------------------------------------------------------


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


class DownloadError(Exception):
    """Raised when a file download fails with a user-friendly message."""


def download_file(
    session: requests.Session,
    spec: FileSpec,
    filename: str,
    outdir: Path,
    source_hint: str = "",
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename

    try:
        with session.get(spec.url, stream=True, timeout=60) as r:
            if r.status_code == 404:
                raise DownloadError(
                    f"File not found (HTTP 404): {spec.url}\n"
                    "The source may have moved or been removed.\n"
                    + (f"{source_hint}" if source_hint else "")
                )
            if r.status_code == 403:
                raise DownloadError(
                    f"Access denied (HTTP 403): {spec.url}\n"
                    "The resource may require authentication or the share link may have expired.\n"
                    + (f"{source_hint}" if source_hint else "")
                )
            if r.status_code == 429:
                raise DownloadError(
                    f"Rate limited (HTTP 429): {spec.url}\n"
                    "Too many requests — wait a moment and try again."
                )
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                raise DownloadError(
                    f"HTTP error downloading {filename}: {e}\n"
                    + (f"{source_hint}" if source_hint else "")
                ) from e

            with outpath.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    except requests.exceptions.ConnectionError as e:
        raise DownloadError(
            f"Could not connect to {spec.url}\n"
            "Check your internet connection or whether the host is reachable.\n"
            + (f"{source_hint}" if source_hint else "")
        ) from e
    except requests.exceptions.Timeout:
        raise DownloadError(
            f"Request timed out while downloading {filename} from {spec.url}\n"
            "Try again later or download manually.\n"
            + (f"{source_hint}" if source_hint else "")
        )

    return outpath


def verify_file(path: Path, expected_md5: str) -> bool:
    actual = md5sum(path)
    ok = actual == expected_md5
    if ok:
        print(f"  [ok]   {path.name}  md5={actual}")
    else:
        print(
            f"  [warn] {path.name}  md5 mismatch — the data may have changed.\n"
            f"         expected: {expected_md5}\n"
            f"         actual:   {actual}\n"
            "         The file has been saved. Verify manually or re-download.",
            file=sys.stderr,
        )
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets into the datasets/ directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=sorted(DATASETS),
        metavar="DATASET",
        help="Dataset to download. Omit to use --list.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and exit.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all implemented datasets.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: datasets/<dataset>/ relative to cwd).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        metavar="FILE",
        default=None,
        help="Subset of files to download (default: all files for the dataset).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip MD5 verification after download.",
    )
    return parser.parse_args()


def cmd_list() -> None:
    print("Available datasets:\n")
    for name, spec in sorted(DATASETS.items()):
        status = "" if spec.implemented else "  [not yet implemented]"
        print(f"  {name:<12}  {spec.description}{status}")
    print()


def cmd_download(args: argparse.Namespace) -> int:
    spec = DATASETS[args.dataset]

    if not spec.implemented:
        print(
            f"Error: downloader for '{spec.name}' is not yet implemented.\n"
            "Add download URLs to DATASETS in src/hgere/commands/download_dataset.py.\n"
            + (f"{spec.source_hint}" if spec.source_hint else ""),
            file=sys.stderr,
        )
        return 1

    outdir = args.outdir if args.outdir is not None else spec.default_outdir()
    filenames = args.files if args.files is not None else list(spec.files.keys())

    unknown = set(filenames) - set(spec.files.keys())
    if unknown:
        print(
            f"Error: unknown file(s) for dataset '{spec.name}': {sorted(unknown)}\n"
            f"Available: {sorted(spec.files.keys())}",
            file=sys.stderr,
        )
        return 1

    session = requests.Session()
    session.headers["User-Agent"] = "hgere-dataset-downloader/1.0"

    all_ok = True

    for filename in filenames:
        file_spec = spec.files[filename]
        print(f"Downloading {filename} ...")
        try:
            path = download_file(session, file_spec, filename, outdir, spec.source_hint)
        except DownloadError as e:
            print(f"Error: {e}", file=sys.stderr)
            all_ok = False
            continue
        print(f"  Saved → {path}")

        if not args.skip_verify and file_spec.md5 is not None:
            if not verify_file(path, file_spec.md5):
                all_ok = False

        if spec.postprocess is not None:
            spec.postprocess(path)

    return 0 if all_ok else 1


def cli() -> None:
    args = parse_args()

    if args.list:
        cmd_list()
        return

    if args.all:
        if args.dataset is not None:
            print(
                "Error: --all and a dataset name are mutually exclusive.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        overall_ok = True
        for name in sorted(DATASETS):
            args.dataset = name
            print(f"\n=== {name} ===")
            if cmd_download(args) != 0:
                overall_ok = False
        raise SystemExit(0 if overall_ok else 1)

    if args.dataset is None:
        print(
            "Error: specify a dataset, use --all, or use --list to see available datasets.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    raise SystemExit(cmd_download(args))


if __name__ == "__main__":
    cli()
