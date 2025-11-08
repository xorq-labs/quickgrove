import pandas as pd
import numpy as np
import pyarrow as pa
import quickgrove

from quickgrove import Feature
from pathlib import Path


TEST_DIR = Path(__file__).parent.parent.parent # ../../


def test_predict():
    df = pd.read_csv(
        TEST_DIR
        / "tests/data/reg_squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    )
    model = quickgrove.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
    )
    actual_preds = df["prediction"].copy().to_list()
    df = df.drop(["target", "prediction"], axis=1)
    batch = pa.RecordBatch.from_pandas(df)
    predictions = model.predict_batches([batch])
    assert len(predictions) == len(df)
    np.testing.assert_array_almost_equal(
        np.array(predictions), np.array(actual_preds), decimal=3
    )


def test_pruning():
    df = pd.read_csv(
        TEST_DIR
        / "tests/data/reg_squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    ).query("carat <0.2")
    model = quickgrove.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
    )
    batch = pa.RecordBatch.from_pandas(df)
    predicates = [Feature("carat") < 0.2]
    actual_preds = df["prediction"].copy().to_list()
    df = df.drop(["target", "prediction"], axis=1)
    pruned_model = model.prune(predicates)
    predictions = pruned_model.predict_batches([batch])
    np.testing.assert_array_almost_equal(
        np.array(predictions), np.array(actual_preds), decimal=3
    )

def test_tree_info():
    model = quickgrove.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
    )
    tree = model.tree_info(0)
    assert isinstance(str(tree), str)
    assert "VecTree:" in str(tree)
    assert "Leaf (weight:" in str(tree)
    
    try:
        model.tree_info(999)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        model.tree_info(None)
        assert False, "Should have raised ValueError" 
    except ValueError:
        pass

def test_prediction_chunking():
    df = pd.read_csv(
        TEST_DIR / "tests/data/reg_squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    )
    model = quickgrove.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
    )
    actual_preds = df["prediction"].copy().to_list()
    df = df.drop(["target", "prediction"], axis=1)
    batch = pa.RecordBatch.from_pandas(df)

    chunk_configs = [
        (32, 4),
        (64, 8),  # default
        (128, 16),
        (256, 32)
    ]

    for row_chunk, tree_chunk in chunk_configs:
        predictions = model.predict_batches([batch], row_chunk_size=row_chunk, tree_chunk_size=tree_chunk)
        np.testing.assert_array_almost_equal(
            np.array(predictions),
            np.array(actual_preds),
            decimal=3,
            err_msg=f"Failed with row_chunk={row_chunk}, tree_chunk={tree_chunk}"
        )

def test_to_json():
    """Test that to_json() returns a valid JSON string representation"""
    import json

    model = quickgrove.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
    )

    # Test that to_json() returns a string
    json_str = model.to_json()
    assert isinstance(json_str, str)

    # Test that the JSON is valid and can be parsed
    json_obj = json.loads(json_str)

    # Check basic structure
    assert "learner" in json_obj
    assert "learner_model_param" in json_obj["learner"]
    assert "gradient_booster" in json_obj["learner"]
    assert "feature_names" in json_obj["learner"]
    assert "feature_types" in json_obj["learner"]
    assert "objective" in json_obj["learner"]

    # Check that key fields are present
    assert "base_score" in json_obj["learner"]["learner_model_param"]
    assert "num_feature" in json_obj["learner"]["learner_model_param"]
    assert "trees" in json_obj["learner"]["gradient_booster"]["model"]

    # Check that we have the correct number of trees
    trees = json_obj["learner"]["gradient_booster"]["model"]["trees"]
    assert len(trees) == 100

    # Check that feature names are preserved
    feature_names = json_obj["learner"]["feature_names"]
    assert len(feature_names) > 0
    assert "carat" in feature_names

    print(f"JSON representation test passed. Model has {len(trees)} trees.")


def test_to_json_roundtrip():
    """Test that a model can be serialized and deserialized with to_json()"""
    import json

    # Load original model
    original_model = quickgrove.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
    )

    # Get test data
    df = pd.read_csv(
        TEST_DIR / "tests/data/reg_squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    )
    actual_preds = df["prediction"].copy().to_list()
    df = df.drop(["target", "prediction"], axis=1)
    batch = pa.RecordBatch.from_pandas(df)

    # Serialize to JSON and reload
    json_str = original_model.to_json()
    reloaded_model = quickgrove.PyGradientBoostedDecisionTrees(json_str)

    # Test that predictions match
    original_predictions = original_model.predict_batches([batch])
    reloaded_predictions = reloaded_model.predict_batches([batch])

    np.testing.assert_array_almost_equal(
        np.array(original_predictions),
        np.array(reloaded_predictions),
        decimal=5,
        err_msg="Predictions differ after JSON roundtrip"
    )

    # Also verify against expected predictions
    np.testing.assert_array_almost_equal(
        np.array(reloaded_predictions),
        np.array(actual_preds),
        decimal=3
    )

    print("JSON roundtrip test passed.")


def test_to_json_pruned_model():
    """Test that to_json() works with pruned models"""
    import json

    model = quickgrove.json_load(
        TEST_DIR / "tests/models/reg_squarederror/diamonds_model_trees_100_mixed.json"
    )

    # Prune the model
    predicates = [Feature("carat") < 0.5]
    pruned_model = model.prune(predicates)

    # Test that we can serialize the pruned model
    json_str = pruned_model.to_json()
    assert isinstance(json_str, str)

    # Parse and verify structure
    json_obj = json.loads(json_str)
    assert "learner" in json_obj

    # Reload the pruned model and test predictions
    df = pd.read_csv(
        TEST_DIR / "tests/data/reg_squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    ).query("carat < 0.5")

    df_test = df.drop(["target", "prediction"], axis=1)
    batch = pa.RecordBatch.from_pandas(df_test)

    # Get predictions from original pruned model
    pruned_predictions = pruned_model.predict_batches([batch])

    # Reload from JSON and predict
    reloaded_pruned = quickgrove.PyGradientBoostedDecisionTrees(json_str)
    reloaded_predictions = reloaded_pruned.predict_batches([batch])

    # Verify predictions match
    np.testing.assert_array_almost_equal(
        np.array(pruned_predictions),
        np.array(reloaded_predictions),
        decimal=5,
        err_msg="Pruned model predictions differ after JSON roundtrip"
    )

    print("Pruned model JSON test passed.")
