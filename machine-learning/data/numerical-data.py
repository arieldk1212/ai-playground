"""
Integers or floatin-point values that behave like numbers.
Temperature, Weight, The number of deer wintering in a nature preserve
BUT number like postal code belongs to categorial data, no relation when you multiply for example 2002 postal code to the 4004 postal code.

Feature Vector -> how we pass it into the model, two main techniques for feature engineering:
  1. Normalization: Converting numerical values into a standard range.
  2. Binning/Bucketing: Converting numerical values into buckets of ranges.

But, Before we start creating feature vectors we need to study the numerical data:
  1. Visualize the data in plots or graphs (Recommended to work with pandas).
  2. Get statistics about the data (mean, median, standard deviation, the values at the quartile divisions).
  3. Find outliers (Values that are distict from others) -> They cause problems, But, they can be born maybe by a mistake that we need
      to delete, or its a legitimate data point, not a mistake, if we decide to keep it, it could help us and improve the model,
      but we should be careful, extreme values can hurt the model, if we decide to delete it, its fine, its called clipping.
"""

import pandas as pd

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_df = pd.read_csv(
    filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
)

# Catch outliers
print(training_df.describe())

