import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ml_edu.results
import ml_edu.experiment

chicago_taxi_dataset = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"
)

training_df = chicago_taxi_dataset.loc[
    :, ("TRIP_MILES", "TRIP_SECONDS", "FARE", "COMPANY", "PAYMENT_TYPE", "TIP_RATE")
]

print(training_df.head(200))

print("Total number of rows: {0}\n\n".format(len(training_df.index)))
training_df.describe(include="all")


max_fare = training_df["FARE"].max()
mean_distance = training_df["TRIP_MILES"].mean()
num_unique_companies = training_df["COMPANY"].unique()
most_freq_payment_type = training_df["PAYMENT_TYPE"].value_counts().idxmax()
missing_values = training_df.isnull().sum().sum()

print("Max Fare: {0}\n".format(max_fare))
print("Mean Distance: {fare:.4f}\n".format(fare=mean_distance))
print("Unique Companies: {0}\n".format(len(num_unique_companies)))
print("Most Frequent Payment Type: {0}\n".format(most_freq_payment_type))
print("Missing Values: {0}\n".format("No" if missing_values == 0 else "Yes"))

print("\n")

# Runs correlation with other fields
print(training_df.corr(numeric_only=True))

training_df[["FARE", "TRIP_MILES", "TRIP_SECONDS"]].plot()
# plt.show()


def create_model(
    settings: ml_edu.experiment.ExperimentSettings, metrics: list[keras.metrics.Metric]
) -> keras.Model:
    inputs = {
        name: keras.Input(shape=(1,), name=name) for name in settings.input_features
    }
    concat_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concat_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
        loss="mean_squared_error",
        metrics=metrics,
    )
    return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values
    history = model.fit(
        x=features,
        y=label,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
    )
    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epochs,
        metrics_history=pd.DataFrame(history.history),
    )
