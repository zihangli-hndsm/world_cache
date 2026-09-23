import pandas as pd
import pytest

from scripts.summarize_framelevel_layers import validate_compatible


def test_framelevel_selection_rejects_mismatched_feature_grids():
    source = pd.DataFrame({"feature": ["layers_2_20_40"], "method": ["mutual"]})
    target = pd.DataFrame({"feature": ["layers_2_12_24"], "method": ["mutual"]})

    with pytest.raises(ValueError, match="source/target frame-level grids differ"):
        validate_compatible(source, target)


def test_framelevel_selection_accepts_matching_feature_grids():
    source = pd.DataFrame({
        "feature": ["layers_2_20_40", "layers_2_20_40"],
        "method": ["mutual", "one_way"],
    })
    target = pd.DataFrame({
        "feature": ["layers_2_20_40", "layers_2_20_40"],
        "method": ["one_way", "mutual"],
    })

    validate_compatible(source, target)
