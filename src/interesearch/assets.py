"""Fetch, verify and load model assets.

Asset paths, relative to the directory passed to each loader:

    target/model_step_99999.pt            the target model, W&B run goodfire/spd/t-9d2b8f02
    target/model_config.yaml, tokenizer.json, final_config.yaml
    decomposition/model_400000.pth        the published decomposition, W&B run goodfire/spd/s-55ea3f9b
    decomposition/final_config.yaml
    labels/labels.json                    subcomponent labels from static.goodfire.ai

Sizes and sha256 are those of the files the original results were computed with (fetched 2026-10-09). They are
checked by ``verify`` on request, not at every load. ``fetch`` asks W&B's anonymous GraphQL API for a signed URL,
the size and the MD5 of each run file, downloads it with curl, and accepts it only if it matches W&B's MD5 and the
sha256 here. Files already present are hashed before they are accepted.

    uv run python -m interesearch.assets fetch assets
    uv run python -m interesearch.assets verify assets
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .checksums import sha256_file
from .model import LlamaSimpleMLP, LlamaSimpleMLPConfig

# The 24 decomposed nn.Linear modules, layer-major. Every per-module list and record follows this order.
MODULES: list[str] = [f"h.{layer}.{s}" for layer in range(4) for s in (
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]


@dataclass(frozen=True)
class Asset:
    path: str  # relative to the assets directory
    size: int
    sha256: str
    run: str = ""  # W&B run in goodfire/spd that holds the file under its base name
    url: str = ""  # direct URL of a file outside W&B


ASSETS = (
    Asset("target/model_step_99999.pt", 267_736_421,
          "650f2eb603fe9fe649671c0c89b86746655bc49c96aa1ad00fbb4c1bf3c6c3b6", run="t-9d2b8f02"),
    Asset("target/model_config.yaml", 323,
          "9664c12d3492ee58520f89703e67ea2790ea13de1f88bf8e3c4594943e0cc59d", run="t-9d2b8f02"),
    Asset("target/tokenizer.json", 3_564_303,
          "d4c0d99d84af59e9126913fafe5210822963e9a3065ee43e6833b358b0c2f825", run="t-9d2b8f02"),
    Asset("target/final_config.yaml", 1_318,
          "565f88480dacaa6131309a9f6abe88e8fd7ee3a8bb6bb493bcc0745eb00584d7", run="t-9d2b8f02"),
    Asset("decomposition/model_400000.pth", 2_908_595_703,
          "fdd794abda8ca79e98f4644590c0c6e696261d9b84b34c8e83f94a2cc3607a93", run="s-55ea3f9b"),
    Asset("decomposition/final_config.yaml", 3_087,
          "75af17e6813ad0e939c18221fa188a757911e292141323cd375b8aa5d06a4ffc", run="s-55ea3f9b"),
    Asset("labels/labels.json", 660_125,
          "9ec9c834392b3835dced0725f4d683d03248c4669d60b8aaf49ef9004c7f933c",
          url="https://static.goodfire.ai/vpd-blog-post/data/model-overview/labels.json"),
)

GRAPHQL = "https://api.wandb.ai/graphql"
ENTITY, PROJECT = "goodfire", "spd"


# ----------------------------------------------------------------------------- loaders
def load_target(assets_dir: str | Path, device: str | torch.device = "cpu",
                dtype: torch.dtype = torch.float32) -> LlamaSimpleMLP:
    """The target model in eval mode with requires_grad False on every parameter."""
    target = Path(assets_dir) / "target"
    model = LlamaSimpleMLP(LlamaSimpleMLPConfig.from_yaml(target / "model_config.yaml"))
    model.load_state_dict(torch.load(target / "model_step_99999.pt", map_location="cpu", weights_only=True), strict=True)
    assert model.wte.weight.data_ptr() == model.lm_head.weight.data_ptr(), "wte/lm_head must stay tied"
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def load_tokenizer(assets_dir: str | Path) -> Tokenizer:
    """The run's own tokenizer.json (EleutherAI/gpt-neox-20b, no BOS added)."""
    return Tokenizer.from_file(str(Path(assets_dir) / "target" / "tokenizer.json"))


# ----------------------------------------------------------------------------- verify and fetch
def check(assets_dir: str | Path, asset: Asset) -> str | None:
    """What is wrong with one file on disk, or None when its size and sha256 match."""
    path = Path(assets_dir) / asset.path
    if not path.is_file():
        return f"{asset.path}: missing"
    if path.stat().st_size != asset.size:
        return f"{asset.path}: {path.stat().st_size} bytes, expected {asset.size}"
    if sha256_file(path) != asset.sha256:
        return f"{asset.path}: sha256 differs from the expected {asset.sha256}"
    return None


def verify(assets_dir: str | Path) -> list[str]:
    """Problems with the asset files, empty when every file matches its size and sha256."""
    return [p for a in ASSETS if (p := check(assets_dir, a)) is not None]


def _wandb_file(run: str, name: str) -> dict:
    """Size, base64 MD5 and a fresh signed download URL of one run file."""
    q = ('query { project(name:"%s", entityName:"%s") { run(name:"%s") { '
         "files(names:[%s]) { edges { node { name sizeBytes md5 directUrl } } } } } }"
         % (PROJECT, ENTITY, run, json.dumps(name)))
    req = urllib.request.Request(GRAPHQL, data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    if "errors" in d:
        raise RuntimeError(d["errors"])
    (edge,) = d["data"]["project"]["run"]["files"]["edges"]
    return edge["node"]


def _curl(url: str, part: Path) -> bool:
    """Resumable download into ``part``. Aborts a stalled transfer (<20 kB/s for 2 min)."""
    part.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-fL", "-sS", "--retry", "5", "--retry-delay", "5",
           "--speed-limit", "20000", "--speed-time", "120", "-C", "-", "-o", str(part), url]
    return subprocess.run(cmd).returncode == 0


def _hashes(path: Path) -> tuple[str, str]:
    """Base64 MD5 (W&B's form) and hex sha256 in one pass."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            md5.update(chunk)
            sha.update(chunk)
    return base64.b64encode(md5.digest()).decode(), sha.hexdigest()


def fetch(assets_dir: str | Path, attempts: int = 6) -> None:
    """Download every absent asset. A file already present is accepted only if its size and sha256 match.

    A download goes to ``<file>.part`` and is renamed only after it matches W&B's MD5 and the sha256 here. A
    mismatching download is deleted. Signed W&B URLs expire, so every attempt asks for a fresh one and resumes.
    """
    for a in ASSETS:
        out = Path(assets_dir) / a.path
        if out.exists():
            problem = check(assets_dir, a)
            if problem is not None:
                raise ValueError(f"{problem}. Remove the file to download it again")
            print(f"[ok  ] {a.path} present")
            continue
        print(f"[get ] {a.path} ({a.size:,} B)", flush=True)
        t0 = time.time()
        part = out.with_suffix(out.suffix + ".part")
        if a.run:
            for i in range(attempts):
                node = _wandb_file(a.run, Path(a.path).name)
                if node["sizeBytes"] != a.size:
                    raise ValueError(f"{a.path}: W&B lists {node['sizeBytes']} bytes, expected {a.size}")
                if _curl(node["directUrl"], part):
                    break
                print(f"[retry] {a.path}: attempt {i + 1} failed, asking for a new URL", flush=True)
                time.sleep(10)
            else:
                raise RuntimeError(f"could not download {a.run}/{Path(a.path).name}")
        elif not _curl(a.url, part):
            raise RuntimeError(f"could not download {a.url}")
        md5, sha = _hashes(part)
        if (a.run and md5 != node["md5"]) or sha != a.sha256:
            part.unlink()
            raise ValueError(f"{a.path}: download has MD5 {md5} (W&B {node['md5'] if a.run else '-'}) and sha256 "
                             f"{sha} (expected {a.sha256}). Deleted it")
        part.rename(out)
        print(f"[ok  ] {a.path} in {time.time() - t0:.0f} s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch or verify the model assets.")
    ap.add_argument("command", choices=("fetch", "verify"))
    ap.add_argument("assets_dir", type=Path, help="assets directory, e.g. assets")
    a = ap.parse_args(argv)
    if a.command == "fetch":
        fetch(a.assets_dir)
        return 0
    bad = 0
    for asset in ASSETS:
        problem = check(a.assets_dir, asset)
        print(problem or f"ok  {asset.path}", flush=True)
        bad += problem is not None
    print(f"{len(ASSETS) - bad}/{len(ASSETS)} files match")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
