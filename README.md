# Lowess Grouped

**Apply groupwise lowess smoothing to a dataframe.**

Smooth data for each category using the [lowess](https://en.wikipedia.org/wiki/Local_regression) (aka loess) algorithm.
You can use this code for all forms of data that should be smoothed independently by group:

![lowess-grouped-example](https://raw.githubusercontent.com/lukiwieser/lowess-grouped/main/docs/lowess-grouped-example.svg)
*Figure 1: Smoothed temperature data for each region*


## Usage

Install the package (Python 3.8 or higher):

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


## Testcases

Tests are defined in the folder `tests`. To run them manually, follow these steps: 

1) Download the [source code](https://github.com/lukiwieser/lowess-grouped) from GitHub.

2) Install package locally by executing the following command in the project folder:
    ```console
    pip install -e .
    ```

    You might need to upgrade your version of pip for this to work:
    ```console
    pip install --upgrade pip
    ```

3) Run the tests:
    ```console
    python ./tests/test_lowess_grouped.py -v
    ```


## Motivation

Smoothing data can greatly improve the interpretability of visualizations.
One commonly used method is lowess, also knows as loess, sometimes also referred as *Savitzky–Golay filter*.

However, the built-in lowess function in Statsmodels (a popular statistics package) applies smoothing to the entire dataframe.
This can lead to undesirable results when you need independent smoothing for multiple groups (e.g., temperature data by regions).

This package was developed to address this limitation and provide some convenience, like getting a dataframe with column names back, instead of unnamed numpy arrays.
Internally it still uses Statsmodels.


## Attribution

This project builds upon the lowess function from [statsmodels](https://www.statsmodels.org).
The temperature data used in the example notebook and testcases is from [Berkley Earth](https://berkeleyearth.org/data/), and licensed under [Creative Commons BY-NC 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).
