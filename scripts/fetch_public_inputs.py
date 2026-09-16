#!/usr/bin/env python3
"""Stage public inputs used by the default-weight ML-PEG benchmarks."""

from __future__ import annotations

import argparse
import bz2
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import shutil
import time
import zipfile

import requests
from bs4 import BeautifulSoup


S3_ENDPOINT = "https://s3.echo.stfc.ac.uk/ml-peg-data"
S3_OBJECTS = {
    "inputs/bulk_crystal/elemental_tm_vacancies/elemental_tm_vacancies.zip": "elemental_tm_vacancies.zip",
    "inputs/defects/Defectstab/Defectstab.zip": "Defectstab.zip",
    "inputs/defects/Relastab/Relastab.zip": "Relastab.zip",
    "inputs/defects/split_vacancy/split_vacancy.zip": "split_vacancy.zip",
    "inputs/molecular_crystal/CPOSS209/CPOSS209.zip": "CPOSS209.zip",
    "inputs/nebs/O_diffusion_2D_TMDs/O_diffusion_2D_TMDs.zip": "O_diffusion_2D_TMDs.zip",
    "inputs/physicality/water_cl2_relaxation/water_cl2_relaxation.zip": "water_cl2_relaxation.zip",
    "inputs/surfaces/SBH17/SBH17.zip": "SBH17.zip",
    "inputs/surfaces/cleavage_energy/cleavage_energy.zip": "cleavage_energy.zip",
}

PRESSURE_BASE = "https://alexandria.icams.rub.de/data/pbe/benchmarks/pressure"
PRESSURE_FILES = [f"P{pressure:03d}.json.bz2" for pressure in range(0, 151, 25)]
LOW_DIMENSIONAL = {
    "https://alexandria.icams.rub.de/data/pbe_2d/alexandria_2d_001.json.bz2": "alexandria_2d_001.json.bz2",
    "https://alexandria.icams.rub.de/data/pbe_2d/alexandria_2d_000.json.bz2": "alexandria_2d_000.json.bz2",
    "https://alexandria.icams.rub.de/data/pbe_1d/alexandria_1d_000.json.bz2": "alexandria_1d_000.json.bz2",
}
PHONON_BASE = "https://alexandria.icams.rub.de/data/phonon_benchmark/pbe"


def download(url: str, destination: Path, attempts: int = 12) -> Path:
    """Download one file atomically with bounded exponential retries."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        print(f"[cached] {destination}", flush=True)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial.replace(destination)
            print(
                f"[downloaded] {destination} ({destination.stat().st_size} bytes)",
                flush=True,
            )
            return destination
        except Exception as error:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise RuntimeError(f"failed to download {url}") from error
            delay = min(60, 2 ** min(attempt, 6))
            print(f"[retry {attempt}/{attempts}] {url}: {error}", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def validate_zip(path: Path) -> None:
    """Reject truncated or non-ZIP responses before publishing the bundle."""
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise ValueError(f"corrupt member {bad_member!r} in {path}")


def stage_s3(cache_dir: Path, workers: int) -> None:
    """Download and validate all ML-PEG S3 archives."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download, f"{S3_ENDPOINT}/{key}", cache_dir / filename): filename
            for key, filename in S3_OBJECTS.items()
        }
        for future in as_completed(futures):
            path = future.result()
            validate_zip(path)


def stage_pressure(stage_root: Path, workers: int) -> None:
    """Download compressed Alexandria high-pressure inputs."""
    pressure_dir = (
        stage_root
        / "repo"
        / "ml_peg"
        / "calcs"
        / "bulk_crystal"
        / "high_pressure_relaxation"
        / "data"
    )
    items = [
        (f"{PRESSURE_BASE}/{filename}", pressure_dir / filename)
        for filename in PRESSURE_FILES
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download, url, path) for url, path in items]
        for future in as_completed(futures):
            path = future.result()
            with bz2.open(path, "rb") as handle:
                handle.read(1)


def stage_low_dimensional(stage_root: Path, workers: int) -> None:
    """Download compressed Alexandria low-dimensional inputs."""
    cache_dir = stage_root / "home-cache" / "ml_peg"
    items = list(
        (url, cache_dir / "low_dimensional_relaxation" / filename)
        for url, filename in LOW_DIMENSIONAL.items()
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download, url, path) for url, path in items]
        for future in as_completed(futures):
            path = future.result()
            with bz2.open(path, "rb") as handle:
                handle.read(1)


def stage_phonons(stage_root: Path, workers: int) -> None:
    """Download the Alexandria phonon index and compressed YAML inputs."""
    cache_dir = stage_root / "home-cache" / "ml_peg"
    index = requests.get(f"{PHONON_BASE}/", timeout=(30, 300))
    index.raise_for_status()
    soup = BeautifulSoup(index.text, "html.parser")
    # Match the benchmark loader's server-order selection exactly.  Avoid a
    # set/sort here: ``calc_phonons`` consumes the first ``N_PHONONS`` IDs.
    mp_ids = list(
        dict.fromkeys(
            match.group(1)
            for link in soup.find_all("a", href=True)
            if (match := re.fullmatch(r"(mp-\d+)\.yaml\.bz2", link["href"]))
        )
    )
    if not mp_ids:
        raise RuntimeError("Alexandria phonon index contained no MP identifiers")

    phonon_root = cache_dir / "alex_phonons"
    phonon_data = phonon_root / "alex_phonon_data"
    phonon_data.mkdir(parents=True, exist_ok=True)
    (phonon_root / "mp_ids.txt").write_text("\n".join(mp_ids) + "\n")
    (phonon_root / "mp_ids_subsampled.txt").write_text("\n".join(mp_ids) + "\n")

    def fetch_phonon(mp_id: str) -> None:
        yaml_path = phonon_data / f"{mp_id}.yaml"
        if yaml_path.is_file() and yaml_path.stat().st_size:
            print(f"[cached] {yaml_path}", flush=True)
            return

        compressed = phonon_data / f"{mp_id}.yaml.bz2"
        download(f"{PHONON_BASE}/{mp_id}.yaml.bz2", compressed)
        partial = yaml_path.with_suffix(yaml_path.suffix + ".part")
        try:
            with bz2.open(compressed, "rb") as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
            if not partial.stat().st_size:
                raise ValueError(f"decompressed phonon input is empty: {compressed}")
            partial.replace(yaml_path)
        finally:
            partial.unlink(missing_ok=True)
        # The benchmark consumes the YAML directly. Avoid publishing both the
        # compressed source and its decompressed copy in the relay artifact.
        compressed.unlink()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_phonon, mp_id) for mp_id in mp_ids]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--group",
        choices=("all", "s3", "pressure", "low-dimensional", "phonons"),
        default="all",
    )
    args = parser.parse_args()

    stage_root = args.root.resolve()
    cache_dir = stage_root / "home-cache" / "ml_peg"
    if args.group in {"all", "s3"}:
        stage_s3(cache_dir, args.workers)
    if args.group in {"all", "pressure"}:
        stage_pressure(stage_root, args.workers)
    if args.group in {"all", "low-dimensional"}:
        stage_low_dimensional(stage_root, args.workers)
    if args.group in {"all", "phonons"}:
        stage_phonons(stage_root, args.workers)
    print(f"Public input group {args.group!r} staged under {stage_root}", flush=True)


if __name__ == "__main__":
    main()
