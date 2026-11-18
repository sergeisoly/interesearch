"""Published tables and explainer values are generated from the saved analyses."""

import json
from pathlib import Path

import pytest

from interesearch.repair_transfer import figures as repair
from interesearch.selective_editing import figures as selective
from interesearch.selective_editing.analysis.explainer_calibration import export as calibration_points
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


def test_summary_calibration_matches_recorded_searches():
    data = RESULTS.parent / "data" / "selective_editing"
    results = RESULTS / "selective_editing"
    exported = calibration_points(data, results / "posthoc.json")
    assert json.dumps(exported, indent=2) + "\n" == (results / "explainer_calibration.json").read_text()


@pytest.mark.parametrize(("effects", "scale"), [
    ([(0.5, 0.001), (1.0, 0.1)], "log"),
    ([(-0.2, -0.005), (1.4, 0.0)], "linear"),
    ([], "linear"),
])
def test_selective_effects_axes_include_observed_values(monkeypatch, tmp_path, effects, scale):
    results = {
        "confirm_result": {"reading_table": [{"contrast": "de->nl|tau=0.5|vpd_atom-lora_rank1"}]},
        "posthoc": {"edits": [
            {"split": "confirm", "method": "vpd_atom", "target": "de", "target_rise": x, "protected_rise": y}
            for x, y in effects
        ]},
    }
    save = selective._save

    def check_axes(fig, path):
        ax = fig.axes[0]
        assert ax.get_yscale() == scale
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        assert all(xlo < x < xhi and ylo < y < yhi for x, y in effects)
        save(fig, path)

    monkeypatch.setattr(selective, "_save", check_axes)
    output = tmp_path / "effects.svg"
    selective.effects_figure(results, output)
    assert output.is_file()
