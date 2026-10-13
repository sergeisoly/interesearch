"""Separation of the published model inputs (the window directories under data/): published dates, source rows and
token rows, checked from their metadata and array bytes.

Checked: each directory against its SHA256SUMS. The salted bucket of every Europarl date against its role's split.
The ordered slices (target fine-tune text = rows of the donor readout windows, quickstart = rows of the fit windows).
Disjoint dates between the independent Europarl date sources (the repair roles, selective train/val/test). Equal
dates where directories are parallel translations of one source split. Disjoint source rows and token rows between
the English pools and guards, and disjoint token rows between the independent Europarl sources.
Not checked: that the target model never saw these windows (the earlier date scan is cited by the protocols, not
repeated), near-duplicate text across sets (no normalized-text deduplication), and the unpublished reserve and
confirm windows.
"""

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from interesearch.checksums import verify_checksums

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "data" / "repair_transfer" / "windows"
SELECTIVE = ROOT / "data" / "selective_editing" / "windows"
QUICKSTART = ROOT / "data" / "quickstart"
DIRS = sorted({p.parent for d in (REPAIR, SELECTIVE, QUICKSTART) for p in d.rglob("windows.json")})

SALT = "vpd-diff-v1"
BUCKETS = {"train": range(0, 70), "val": range(70, 85), "test": range(85, 100)}
SPLIT = {  # directory -> split of the Europarl date buckets its dates come from
    **{REPAIR / r: "train" for r in ("fit", "selection", "donor_readout", "donor_fit", "finetune_val",
                                     "finetune_train/primary_target", "finetune_train/replication_target",
                                     "finetune_train/donor")},
    REPAIR / "readout": "val",
    QUICKSTART: "train",
    SELECTIVE / "train": "train",
    SELECTIVE / "val": "val",
    SELECTIVE / "protected_europarl_val": "val",
    SELECTIVE / "test": "test",
    SELECTIVE / "guard_europarl_en": "test",
    SELECTIVE / "protected_europarl_test": "test",
}
INDEPENDENT = [REPAIR / r for r in ("fit", "selection", "readout", "donor_readout", "donor_fit", "finetune_val",
                                    "finetune_train/donor")] + [SELECTIVE / r for r in ("train", "val", "test")]
ENGLISH = [SELECTIVE / r for r in ("retain_en", "qcheck_en", "val_en", "guard_europarl_en", "guard_wikitext",
                                   "guard_pile")]


def meta(d: Path) -> dict:
    return json.loads((d / "windows.json").read_text())


def tokens(d: Path, stem: str) -> np.ndarray:
    return np.load(d / f"{stem}.npy")


def source_rows(d: Path) -> set[str]:
    """Canonical source rows: Europarl session date, WikiText split + article_index, Pile shard + row or document."""
    m = meta(d)
    if "dates" in m:
        return {f"europarl-v7:date={x}" for x in m["dates"]}
    return {r.split(",")[0] for r in m["rows"]}


def row_hashes(d: Path, stem: str) -> set[str]:
    return {hashlib.sha256(r.tobytes()).hexdigest() for r in tokens(d, stem)}


def test_every_directory_matches_its_sha256sums():
    assert len(DIRS) == 22  # 9 repair transfer, 12 selective editing, quickstart
    for d in DIRS:
        m = meta(d)
        verify_checksums(d, ("windows.json", *(f"{s}.npy" for s in m["languages"])))
        for s in m["languages"]:
            a = tokens(d, s)
            assert a.dtype == np.int32 and a.shape == (len(m.get("dates") or m["rows"]), 257), (d, s)


def test_every_europarl_date_is_in_its_split_bucket():
    assert set(SPLIT) == {d for d in DIRS if "dates" in meta(d)}
    for d, split in SPLIT.items():
        for x in meta(d)["dates"]:
            assert int(hashlib.sha256(f"{SALT}|{x}".encode()).hexdigest(), 16) % 100 in BUCKETS[split], (d, x)


def test_target_finetune_text_and_quickstart_are_ordered_slices():
    for d, src, (a, b) in ((REPAIR / "finetune_train/primary_target", REPAIR / "donor_readout", (0, 64)),
                           (REPAIR / "finetune_train/replication_target", REPAIR / "donor_readout", (64, 128)),
                           (QUICKSTART, REPAIR / "fit", (0, 4))):
        m = meta(d)
        assert m["source"] == {"role": meta(src)["role"], "rows": [a, b]}
        assert m["dates"] == meta(src)["dates"][a:b]
        for s in m["languages"]:
            assert np.array_equal(tokens(d, s), tokens(src, s)[a:b]), (d, s)


def test_independent_date_sources_are_disjoint():
    for a, b in combinations(INDEPENDENT, 2):
        assert not set(meta(a)["dates"]) & set(meta(b)["dates"]), (a, b)


def test_parallel_directories_share_the_dates_of_their_split():
    def dates(r):
        return meta(SELECTIVE / r)["dates"]

    assert dates("guard_europarl_en") == dates("test") == dates("protected_europarl_test")
    assert dates("protected_europarl_val") == dates("val")
    assert meta(SELECTIVE / "retain_en")["rows"][:128] == [f"europarl-v7:date={x}" for x in dates("train")]
    assert meta(SELECTIVE / "val_en")["rows"][:32] == [f"europarl-v7:date={x}" for x in dates("val")]


def test_english_pools_and_guards_have_disjoint_source_rows():
    for a, b in combinations(ENGLISH, 2):
        assert not source_rows(a) & source_rows(b), (a, b)


def test_independent_directories_have_disjoint_token_rows():
    for a, b in [*combinations(ENGLISH, 2), *combinations(INDEPENDENT, 2)]:
        for s in set(meta(a)["languages"]) & set(meta(b)["languages"]):
            assert not row_hashes(a, s) & row_hashes(b, s), (a, b, s)
