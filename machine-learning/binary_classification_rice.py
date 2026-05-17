import keras
import numpy as np
import pandas as pd
import plotly.express as px

import ml_edu.results
import ml_edu.experiment

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

rice_dataset_raw: pd.DataFrame = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"
)
rice_dataset: pd.DataFrame = rice_dataset_raw[
    [
        "Area",
        "Perimeter",
        "Major_Axis_Length",
        "Minor_Axis_Length",
        "Eccentricity",
        "Convex_Area",
        "Extent",
        "Class",
    ]
]

rice_dataset.describe()

# print(rice_dataset.Major_Axis_Length.max())
# print(rice_dataset.Major_Axis_Length.min())
# print(rice_dataset.Area.max())
# print(rice_dataset.Area.min())
# print(
#     (rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())
#     / rice_dataset.Perimeter.std()
# )

# for x_axis_data, y_axis_data in [
#     ("Area", "Eccentricity"),
#     ("Convex_Area", "Perimeter"),
#     ("Major_Axis_Length", "Minor_Axis_Length"),
#     ("Perimeter", "Extent"),
#     ("Eccentricity", "Major_Axis_Length"),
# ]:
#     px.scatter_3d(rice_dataset, x=x_axis_data, y=y_axis_data, color="Class").show()

# for x_axis_data, y_axis_data, z_axis_data in [
#     ("Area", "Major_Axis_Length", "Minor_Axis_Length"),
#     ("Area", "Major_Axis_Length", "Eccentricity"),
# ]:
#     px.scatter_3d(
#         rice_dataset, x=x_axis_data, y=y_axis_data, z=z_axis_data, color="Class"
#     ).show()


# Converting to Z-Scores for normalization

# Calculate the Z-scores of each numerical column in the raw data and write
# them into a new DataFrame named df_norm.
feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes("number").columns
normalized_dataset = (rice_dataset[numerical_features] - feature_mean) / feature_std

# Copy the class to the new dataframe
normalized_dataset["Class"] = rice_dataset["Class"]

# Examine
print(rice_dataset.head())
print(normalized_dataset.head())

# Set same seed to keep number generation consistent.
keras.utils.set_random_seed(42)

normalized_dataset["Class"] = (normalized_dataset["Class"] == "Cammeo").astype(int)
normalized_dataset.sample(10)
