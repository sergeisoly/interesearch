"""Published tables and explainer values are generated from the saved analyses."""

import json
from pathlib import Path

import pytest

from interesearch.repair_transfer import figures as repair
from interesearch.selective_editing import figures as selective
from interesearch.recipes import load_recipe

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.mark.parametrize("study, figures", [("selective_editing", selective), ("repair_transfer", repair)])
def test_tables_and_explainer_match_saved_analyses(study, figures):
    directory = RESULTS / study
    if study == "selective_editing":
        results = {name: json.loads((directory / f"{name}.json").read_text()) for name in (
            "confirm_result", "confirm_analysis", "dev_analysis", "posthoc", "extended_comparisons_analysis")}
        explainer = figures.explainer_data(results)
        args = (results,)
    else:
        source = (directory / "analysis.json").read_bytes()
        results = json.loads(source)
        recipe = results.get("resolved_recipe") or load_recipe("interesearch.repair_transfer")
        explainer = figures.explainer_data(results, source, recipe)
        args = (results, recipe)
    assert sorted(figures.TABLES) == sorted(p.name for p in (directory / "tables").glob("*.md"))
    for name, make in figures.TABLES.items():
        assert make(*args) == (directory / "tables" / name).read_text(), name
    assert json.dumps(explainer, indent=2) + "\n" == (directory / "explainer_data.json").read_text()
