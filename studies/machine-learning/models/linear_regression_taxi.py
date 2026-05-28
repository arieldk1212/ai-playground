from typing import List

import keras
import numpy as np
import pandas as pd

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
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )


settings_1 = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=20,
    batch_size=50,
    input_features=["TRIP_MILES"],
)
metrics = [keras.metrics.RootMeanSquaredError(name="rmse")]
model_1 = create_model(settings_1, metrics)
# experiment_1 = train_model("One", model_1, training_df, "FARE", settings_1)
# ml_edu.results.plot_experiment_metrics(experiment_1, ["rmse"])
# ml_edu.results.plot_model_predictions(experiment_1, training_df, "FARE")


settings_2 = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=20,
    batch_size=50,
    input_features=["TRIP_MILES", "TRIP_MINUTES"],
)
training_df["TRIP_MINUTES"] = training_df["TRIP_SECONDS"] / 60
metrics = [keras.metrics.RootMeanSquaredError(name="rmse")]
model_2 = create_model(settings_2, metrics)
experiment_2 = train_model("One", model_2, training_df, "FARE", settings_2)
# ml_edu.results.plot_experiment_metrics(experiment_2, ["rmse"])
# ml_edu.results.plot_model_predictions(experiment_2, training_df, "FARE")

# comparison = ml_edu.results.compare_experiment(
#     [experiment_1, experiment_2], ["rmse"], training_df, training_df["FARE"].values
# )


class Predictions:
    @staticmethod
    def _format_currency(x):
        return "${:.2f}".format(x)

    @staticmethod
    def build_batch(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        batch = df.sample(n=batch_size).copy()
        batch.set_index(np.arange(batch_size), inplace=True)
        return batch

    @staticmethod
    def predict_fare(
        model: keras.Model,
        df: pd.DataFrame,
        features: List[str],
        label: str,
        batch_size=50,
    ) -> pd.DataFrame:
        batch = Predictions.build_batch(df, batch_size)
        predicted_values = model.predict_on_batch(
            x={name: batch[name].values for name in features}
        )

        data = {
            "PREDICTED_FARE": [],
            "OBSERVED_FARE": [],
            "L1_LOSS": [],
            features[0]: [],
            features[1]: [],
        }
        for i in range(batch_size):
            predicted = predicted_values[i][0]
            observed = batch.at[i, label]
            data["PREDICTED_FARE"].append(Predictions._format_currency(predicted))
            data["OBSERVED_FARE"].append(Predictions._format_currency(observed))
            data["L1_LOSS"].append(
                Predictions._format_currency(abs(observed - predicted))
            )
            data[features[0]].append(batch.at[i, features[0]])
            data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

        output_df = pd.DataFrame(data)
        return output_df

    @staticmethod
    def show_predictions(output: pd.DataFrame) -> None:
        header = "-" * 80
        banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
        print(banner)
        print(output)
        return


output = Predictions.predict_fare(
    experiment_2.model, training_df, experiment_2.settings.input_features, "FARE"
)
Predictions.show_predictions(output)
