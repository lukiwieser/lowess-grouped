# Lowess Grouped

**Apply groupwise lowess smoothing to a dataframe**

Smooth data for each category using the lowess (aka loess) algorithm.
You can use this code for all forms of data that should be smoothed independently by group.

![lowess-grouped-example](https://raw.githubusercontent.com/lukiwieser/lowess-grouped/main/docs/lowess-grouped-example.png)

*Figure 1: Smoothed temperature data for each region*


## Usage

Install the package (Python 3.6 or higher):

```console
pip install lowess-grouped
```

Import the package and call the function `lowess_grouped` with your dataframe `df`. Use the parameter `frac` to control the strength of the smoothing:

```python
from lowess_grouped.lowess_grouped import lowess_grouped

df_smoothed = lowess_grouped(df, 
                             x_name="year", 
                             y_name="temperature_anomaly",
                             group_name="region_name", 
                             frac=0.05)
```

For a detailed example, refer to the notebook [temperature-example.ipynb](https://github.com/lukiwieser/lowess-grouped/blob/main/example/temperature-example.ipynb).


## Motivation

Smoothing data can make plots more readable, and one commonly used method is lowess/loess.

Statsmodels lowess only smooths the entire dataframe, leading to undesirable results when you need independent smoothing for multiple groups (e.g., temperature data by regions).

This package was developed to address this limitation.
It internally uses statsmodels, that's why some parameters have the same names.
Feel free to use to code as inspiration.


## Attribution

This project builds upon the lowess function from [statsmodels](https://www.statsmodels.org).
