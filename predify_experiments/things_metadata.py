from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import pickle
import random
import re
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WNID_RE = re.compile(r"n\d{8}", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _find_column(
    fieldnames: Iterable[str], aliases: set[str], *, required: bool = True
) -> str | None:
    by_key = {_column_key(name): name for name in fieldnames}
    for alias in aliases:
        if alias in by_key:
            return by_key[alias]
    if required:
        raise ValueError(
            f"Required column {sorted(aliases)} not found; available columns: "
            f"{list(fieldnames)}"
        )
    return None


def load_imagenet_wnids(path: Path) -> tuple[str, ...]:
    wnids = tuple(
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(wnids) != 1000:
        raise ValueError(f"Expected 1000 ImageNet wnids in {path}, found {len(wnids)}")
    if len(set(wnids)) != len(wnids) or any(WNID_RE.fullmatch(x) is None for x in wnids):
        raise ValueError(f"Invalid or duplicate ImageNet wnid in {path}")
    return wnids


@dataclass(frozen=True)
class ConceptMapping:
    word: str
    unique_id: str
    wnid: str | None
    imagenet_index: int | None


def load_exact_concept_mapping(
    concept_metadata: Path, imagenet_wnids: tuple[str, ...]
) -> list[ConceptMapping]:
    """Map THINGS concepts to ImageNet-1k by exact WordNet ID equality only."""

    wnid_to_index = {wnid: index for index, wnid in enumerate(imagenet_wnids)}
    with concept_metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"THINGS concept metadata has no header: {concept_metadata}")
        unique_column = _find_column(reader.fieldnames, {"uniqueid"})
        wnid_column = _find_column(reader.fieldnames, {"wordnetid"})
        word_column = _find_column(
            reader.fieldnames, {"word", "concept", "name"}, required=False
        )
        mappings: list[ConceptMapping] = []
        seen_unique: set[str] = set()
        mapped_wnids: set[str] = set()
        for row in reader:
            unique_id = str(row.get(unique_column, "")).strip()
            if not unique_id:
                continue
            if unique_id in seen_unique:
                raise ValueError(f"Duplicate THINGS uniqueID: {unique_id}")
            seen_unique.add(unique_id)
            raw_wnid = str(row.get(wnid_column, ""))
            matches = [value.lower() for value in WNID_RE.findall(raw_wnid)]
            exact = next((value for value in matches if value in wnid_to_index), None)
            if exact is not None:
                if exact in mapped_wnids:
                    raise ValueError(f"Multiple THINGS concepts map to ImageNet wnid {exact}")
                mapped_wnids.add(exact)
            mappings.append(
                ConceptMapping(
                    word=str(row.get(word_column, unique_id)).strip()
                    if word_column
                    else unique_id,
                    unique_id=unique_id,
                    wnid=exact,
                    imagenet_index=None if exact is None else wnid_to_index[exact],
                )
            )
    return mappings


def imagefolder_index(images_dir: Path) -> tuple[list[Path], list[str]]:
    """Return ImageFolder-compatible paths and class names in stable order."""

    paths: list[Path] = []
    classes: list[str] = []
    for class_dir in sorted((p for p in images_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        classes.append(class_dir.name)
        for path in sorted(
            (
                p
                for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=lambda p: p.as_posix(),
        ):
            paths.append(path)
    return paths, classes


def load_image_sources(image_metadata_dir: Path | None) -> dict[str, str]:
    """Read any usable image/source table from the official metadata tree.

    Keys include the lower-cased filename and any supplied relative path.  An
    explicit ImageNet marker wins over conflicting or more generic sources.
    Missing metadata is intentionally represented later as ``source_unknown``.
    """

    if image_metadata_dir is None or not image_metadata_dir.is_dir():
        return {}
    result: dict[str, str] = {}
    filename_aliases = {
        "image",
        "imageid",
        "imagename",
        "imagefilename",
        "filename",
        "filepath",
        "path",
    }
    source_aliases = {"source", "imagesource", "originsource", "origin", "dataset"}
    for table in sorted(image_metadata_dir.rglob("*")):
        if not table.is_file() or table.suffix.casefold() not in {".tsv", ".csv"}:
            continue
        delimiter = "\t" if table.suffix.casefold() == ".tsv" else ","
        try:
            with table.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                if not reader.fieldnames:
                    continue
                filename_column = _find_column(
                    reader.fieldnames, filename_aliases, required=False
                )
                source_column = _find_column(reader.fieldnames, source_aliases, required=False)
                if filename_column is None or source_column is None:
                    continue
                for row in reader:
                    raw_name = str(row.get(filename_column, "")).strip().replace("\\", "/")
                    raw_source = str(row.get(source_column, "")).strip()
                    if not raw_name or not raw_source:
                        continue
                    source = "imagenet" if "imagenet" in raw_source.casefold() else raw_source
                    for key in {raw_name.casefold(), Path(raw_name).name.casefold()}:
                        if result.get(key) != "imagenet" or source == "imagenet":
                            result[key] = source
        except (OSError, UnicodeError, csv.Error):
            continue
    return result


def _source_for(path: Path, images_dir: Path, sources: dict[str, str]) -> str:
    relative = path.relative_to(images_dir).as_posix().casefold()
    explicit = sources.get(relative, sources.get(path.name.casefold()))
    if explicit is not None:
        return explicit
    # THINGS encodes provenance in the final filename: n=ImageNet,
    # b=reference image, s=all other selected images (Hebart et al., 2019).
    suffix = path.stem[-1:].casefold()
    return {
        "n": "imagenet",
        "b": "reference",
        "s": "selected_other",
    }.get(suffix, "source_unknown")


def _stable_concept_seed(seed: int, unique_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{unique_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_things_cohort(
    *,
    images_dir: Path,
    concept_metadata: Path,
    image_metadata_dir: Path | None,
    imagenet_wnids_file: Path,
    output_dir: Path,
    seed: int = 114514,
    concept_count: int = 200,
    images_per_concept: int = 5,
    exclude_imagenet_source: bool = True,
    expected_exact_overlaps: int | None = None,
) -> dict[str, Any]:
    """Create the deterministic paired THINGS trial manifest and audit tables."""

    wnids = load_imagenet_wnids(imagenet_wnids_file)
    mappings = load_exact_concept_mapping(concept_metadata, wnids)
    exact = [item for item in mappings if item.imagenet_index is not None]
    if expected_exact_overlaps is not None and len(exact) != expected_exact_overlaps:
        raise ValueError(
            f"Expected {expected_exact_overlaps} exact THINGS/ImageNet overlaps, "
            f"found {len(exact)}"
        )

    all_paths, _classes = imagefolder_index(images_dir)
    dataset_index = {path.resolve(): index for index, path in enumerate(all_paths)}
    by_concept: dict[str, list[Path]] = {}
    for path in all_paths:
        unique_id = path.relative_to(images_dir).parts[0]
        by_concept.setdefault(unique_id, []).append(path)
    sources = load_image_sources(image_metadata_dir)

    candidates: list[tuple[ConceptMapping, list[tuple[Path, str]]]] = []
    audit_rows: list[dict[str, Any]] = []
    for mapping in sorted(exact, key=lambda item: item.unique_id):
        eligible: list[tuple[Path, str]] = []
        known_imagenet = 0
        for path in by_concept.get(mapping.unique_id, []):
            source = _source_for(path, images_dir, sources)
            if source == "imagenet":
                known_imagenet += 1
                if exclude_imagenet_source:
                    continue
            eligible.append((path, source))
        if len(eligible) >= images_per_concept:
            candidates.append((mapping, eligible))
        audit_rows.append(
            {
                "word": mapping.word,
                "unique_id": mapping.unique_id,
                "wnid": mapping.wnid,
                "imagenet_index": mapping.imagenet_index,
                "image_count": len(by_concept.get(mapping.unique_id, [])),
                "known_imagenet_count": known_imagenet,
                "eligible_count": len(eligible),
                "selected": False,
            }
        )
    if len(candidates) < concept_count:
        raise ValueError(
            f"Only {len(candidates)} exact-overlap THINGS concepts have at least "
            f"{images_per_concept} eligible images; need {concept_count}"
        )

    selected = candidates[:concept_count]
    selected_ids = {mapping.unique_id for mapping, _ in selected}
    for row in audit_rows:
        row["selected"] = row["unique_id"] in selected_ids

    selected_images: dict[str, list[tuple[Path, str]]] = {}
    for mapping, eligible in selected:
        choices = list(sorted(eligible, key=lambda item: item[0].as_posix()))
        random.Random(_stable_concept_seed(seed, mapping.unique_id)).shuffle(choices)
        selected_images[mapping.unique_id] = choices[:images_per_concept]

    rng = random.Random(seed)
    trials: list[dict[str, Any]] = []
    compact_label = {
        mapping.unique_id: index for index, (mapping, _eligible) in enumerate(selected)
    }
    mapping_by_id = {mapping.unique_id: mapping for mapping, _eligible in selected}
    for block in range(images_per_concept):
        order = [mapping.unique_id for mapping, _eligible in selected]
        rng.shuffle(order)
        for unique_id in order:
            mapping = mapping_by_id[unique_id]
            path, source = selected_images[unique_id][block]
            base = {
                "dataset_index": dataset_index[path.resolve()],
                "native_label": compact_label[unique_id],
                "classifier_target": int(mapping.imagenet_index),
                "classifier_target_valid": True,
                "unique_id": unique_id,
                "word": mapping.word,
                "wnid": mapping.wnid,
                "image_relpath": path.relative_to(images_dir).as_posix(),
                "image_source": source,
                "block": block,
            }
            trials.extend(({**base, "pair_role": "prime"}, {**base, "pair_role": "target"}))

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = output_dir / "exact_imagenet1k_mapping.tsv"
    with mapping_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(audit_rows)
    selection_file = output_dir / "selected_200_concepts.tsv"
    selection_rows = [row for row in audit_rows if row["selected"]]
    with selection_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(selection_rows)

    metadata = {
        "dataset": "THINGS",
        "schema_version": 1,
        "seed": seed,
        "concept_count": concept_count,
        "images_per_concept": images_per_concept,
        "trial_count": len(trials),
        "exact_overlap_count": len(exact),
        "eligible_concept_count": len(candidates),
        "exclude_imagenet_source": exclude_imagenet_source,
        "unknown_source_image_count": sum(
            row["image_source"] == "source_unknown" for row in trials[1::2]
        ),
        "concept_metadata_sha256": sha256_file(concept_metadata),
        "imagenet_wnids_sha256": sha256_file(imagenet_wnids_file),
        "mapping_audit_sha256": sha256_file(mapping_file),
        "selection_audit_sha256": sha256_file(selection_file),
    }
    sample = {
        "schema_version": 1,
        "out_list": [row["dataset_index"] for row in trials],
        # Compatibility key: historically this was the classifier label.
        "full_clss_list": [row["classifier_target"] for row in trials],
        "native_labels": [row["native_label"] for row in trials],
        "classifier_targets": [row["classifier_target"] for row in trials],
        "classifier_target_valid_mask": [row["classifier_target_valid"] for row in trials],
        "unique_ids": [row["unique_id"] for row in trials],
        "words": [row["word"] for row in trials],
        "wnids": [row["wnid"] for row in trials],
        "image_relpaths": [row["image_relpath"] for row in trials],
        "image_sources": [row["image_source"] for row in trials],
        "blocks": [row["block"] for row in trials],
        "pair_roles": [row["pair_role"] for row in trials],
        "sample_metadata": metadata,
        "selected_concepts": selection_rows,
    }
    sample_file = output_dir / f"THINGS_N_Back-1_seed{seed}.pckl"
    with sample_file.open("wb") as handle:
        pickle.dump(sample, handle)
    manifest_file = output_dir / "THINGS_N_Back-1_manifest.json"
    manifest_file.write_text(
        json.dumps({**metadata, "sample_file": sample_file.name}, indent=2),
        encoding="utf-8",
    )
    return sample
