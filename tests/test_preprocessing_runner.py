import numpy as np
import pandas as pd

from wq_framework.preprocessing import get_outlier_detector, run_outlier_detection


def test_columns_are_processed_independently():
    # An outlier in TDS should not affect the pH column at all — this is
    # the key behavior confirmed with the user (unlike the original
    # sequential/row-dropping code).
    df = pd.DataFrame(
        {
            "TDS": [100, 101, 99, 100, 500],  # 500 is an outlier
            "pH": [7.0, 7.1, 6.9, 7.0, 7.2],  # no outliers
        }
    )
    detector = get_outlier_detector("iqr", k=3.0)
    cleaned, reports = run_outlier_detection(df, ["TDS", "pH"], detector)

    assert np.isnan(cleaned.loc[4, "TDS"])
    # the row is NOT dropped — pH value for that same row is untouched
    assert cleaned.loc[4, "pH"] == 7.2
    assert len(cleaned) == len(df)  # no rows removed at all


def test_bounds_computed_on_original_data_not_shrinking_dataset():
    # If column A's outlier removal shrank the dataset before column B's
    # quantiles were computed (the OLD behavior), B's bounds would differ
    # depending on column order. Confirm order-independence here.
    df = pd.DataFrame(
        {
            "A": [10, 11, 9, 10, 100],  # outlier at row 4
            "B": [20, 21, 19, 20, 21],  # no outliers
        }
    )
    detector = get_outlier_detector("iqr", k=3.0)

    cleaned_ab, _ = run_outlier_detection(df, ["A", "B"], detector)
    cleaned_ba, _ = run_outlier_detection(df, ["B", "A"], detector)

    pd.testing.assert_frame_equal(cleaned_ab, cleaned_ba)


def test_missing_optional_column_is_skipped_silently():
    df = pd.DataFrame({"TDS": [100, 101, 99]})
    detector = get_outlier_detector("iqr", k=3.0)
    cleaned, reports = run_outlier_detection(df, ["TDS", "pH", "Cl"], detector)

    assert list(cleaned.columns) == ["TDS"]
    assert len(reports) == 1
    assert reports[0].parameter == "TDS"


def test_report_returned_per_processed_column():
    df = pd.DataFrame({"TDS": [100, 200], "pH": [7.0, 7.1]})
    detector = get_outlier_detector("iqr", k=3.0)
    _, reports = run_outlier_detection(df, ["TDS", "pH"], detector)
    assert {r.parameter for r in reports} == {"TDS", "pH"}
