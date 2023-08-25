import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from lowess_grouped.lowess_grouped import lowess_grouped

DATA_DIR = Path(__file__).parent / "data"


class TestLowessGrouped(unittest.TestCase):
    def setUp(self):
        # load the data once for all tests
        self.temp_region = pd.read_csv(DATA_DIR / "temperature-by-region.csv")

    def test_lowess_has_no_side_effects(self):
        input_data_copy1 = self.temp_region.copy()
        input_data_copy2 = self.temp_region.copy()

        lowess_grouped(input_data_copy1, "year", "temperature_anomaly", "region_name", frac=0.05)

        self.assertTrue(
            input_data_copy1.equals(input_data_copy2),
            "lowess_grouped seems to change the input dataframe"
        )

    def test_lowess_for_multiple_groups(self):
        # copy data so that we won't accidentally change the data of other tests
        temp_region = self.temp_region.copy()

        # smooth data with lowess-grouped
        lowess_grouped_output = lowess_grouped(
            temp_region,
            "year",
            "temperature_anomaly",
            "region_name",
            frac=0.05
        )

        # foreach region (aka group), check if lowess-grouped produces the same output as statmodels lowess()
        groups = temp_region["region_name"].unique().tolist()
        for group in groups:
            temp_region_subset = temp_region[temp_region["region_name"] == group]

            # get smoothed values from statsmodels lowess, for this region:
            smooth_values_statsmodels: np.ndarray = lowess(temp_region_subset["temperature_anomaly"],
                                                           temp_region_subset["year"], frac=0.05)[:, 1]

            # get smoothed values from lowess-grouped, for this region:
            smooth_values_lowess_grouped = lowess_grouped_output[lowess_grouped_output["region_name"] == group][
                "temperature_anomaly_smooth"].to_numpy()

            self.assertTrue(
                np.array_equal(smooth_values_statsmodels, smooth_values_lowess_grouped),
                f"lowess-grouped values are different from statmodels lowess, for region {group}"
            )

    def test_lowess_for_single_groups(self):
        # copy data so that we won't accidentally change the data of other tests
        temp_region = self.temp_region.copy()

        # foreach region (aka group), check if lowess-grouped produces the same output as statmodels lowess()
        groups = temp_region["region_name"].unique().tolist()
        for group in groups:
            temp_region_subset = temp_region[temp_region["region_name"] == group]

            # get smoothed values from statsmodels lowess, for this region:
            smooth_values_statsmodels: np.ndarray = lowess(temp_region_subset["temperature_anomaly"],
                                                           temp_region_subset["year"], frac=0.05)[:, 1]

            # get smoothed values from lowess-grouped, for this region:
            smooth_values_lowess_grouped = \
                lowess_grouped(temp_region_subset, "year", "temperature_anomaly", None, frac=0.05)[
                    "temperature_anomaly_smooth"].to_numpy()

            self.assertTrue(
                np.array_equal(smooth_values_statsmodels, smooth_values_lowess_grouped),
                f"lowess-grouped values are different from statmodels lowess, for region {group}"
            )


if __name__ == '__main__':
    unittest.main()
