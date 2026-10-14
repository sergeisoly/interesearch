import pytest
import yaml

from interesearch import recipes


@pytest.fixture
def published(tmp_path, monkeypatch):
    profile = {"training": {"steps": 200, "lr": 0.0001, "alpha": None}, "targets": ["de", "fr"]}
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(profile))
    monkeypatch.setattr(recipes, "files", lambda package: tmp_path)
    return profile


def test_recipe_replacement_does_not_change_published_profile(published, tmp_path, monkeypatch):
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump({**published, "training": {"steps": 3, "lr": 1, "alpha": 2}}))
    monkeypatch.chdir(tmp_path.parent)
    loaded = recipes.load_recipe("study", custom)
    assert loaded["training"] == {"steps": 3, "lr": 1.0, "alpha": 2}
    assert type(loaded["training"]["lr"]) is float
    assert recipes.load_recipe("study") == published


@pytest.mark.parametrize("replacement, message", [
    ("training: {steps: 2, steps: 3, lr: 0.1, alpha: null}\ntargets: [de]", "duplicate"),
    ("training: {steps: 3, lr: 1e-4, alpha: null}\ntargets: [de]", "expected float"),
    ("training: {steps: 3, lr: .nan, alpha: null}\ntargets: [de]", "finite"),
    ("training: {steps: true, lr: 0.1, alpha: null}\ntargets: [de]", "expected int"),
    ("training: {steps: 3, lr: 0.1, alpha: null, typo: 2}\ntargets: [de]", "unknown keys"),
    ("training: {steps: 3, lr: 0.1}\ntargets: [de]", "missing keys"),
])
def test_recipe_rejects_invalid_input(published, tmp_path, replacement, message):
    custom = tmp_path / "invalid.yaml"
    custom.write_text(replacement)
    with pytest.raises(ValueError, match=message):
        recipes.load_recipe("study", custom)
