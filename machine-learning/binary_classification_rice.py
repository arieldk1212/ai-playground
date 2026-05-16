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
