import math


def assert_close(new, old, path="$"):
    """Equal structure, strings and integers. Floats within 1e-12 relative (other CPUs may round reductions
    differently), NaN only where NaN."""
    if isinstance(old, dict):
        assert new.keys() == old.keys(), path
        for k in old:
            assert_close(new[k], old[k], f"{path}.{k}")
    elif isinstance(old, list):
        assert len(new) == len(old), path
        for i, (a, b) in enumerate(zip(new, old)):
            assert_close(a, b, f"{path}[{i}]")
    elif isinstance(old, float):
        assert (math.isnan(new) and math.isnan(old)) or math.isclose(new, old, rel_tol=1e-12, abs_tol=1e-15), path
    else:
        assert new == old, path
