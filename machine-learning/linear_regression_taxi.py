import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

chicago_taxi_dataset = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"
)

training_df = chicago_taxi_dataset.loc[
    :, ("TRIP_MILES", "TRIP_SECONDS", "FARE", "COMPANY", "PAYMENT_TYPE", "TIP_RATE")
]

print(training_df.head(200))

print("Total number of rows: {0}\n\n".format(len(training_df.index)))
training_df.describe(include="all")
