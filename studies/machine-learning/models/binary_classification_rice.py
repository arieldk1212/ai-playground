import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# print(rice_dataset.head())
# print(normalized_dataset.head())

# Set same seed to keep number generation consistent.
keras.utils.set_random_seed(42)

normalized_dataset["Class_Bool"] = (normalized_dataset["Class"] == "Cammeo").astype(int)
# normalized_dataset.sample(10)


# Train, Validate, Test
number_samples = len(normalized_dataset)
index_80 = round(number_samples * 0.8)
index_90 = index_80 + round(number_samples * 0.1)

shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80]
validation_data = shuffled_dataset.iloc[index_80:index_90]
test_data = shuffled_dataset.iloc[index_90:]

print(test_data.head())

label_cloumns = ["Class", "Class_Bool"]

train_features = train_data.drop(columns=label_cloumns)
train_labels = train_data["Class_Bool"].to_numpy()
validation_features = validation_data.drop(columns=label_cloumns)
validation_labels = validation_data["Class_Bool"].to_numpy()
test_features = test_data.drop(columns=label_cloumns)
test_labels = test_data["Class_Bool"].to_numpy()

# Train
input_features = [
    "Eccentricity",
    "Major_Axis_Length",
    "Area",
]


def create_model(
    settings: ml_edu.experiment.ExperimentSettings, metrics: list[keras.metrics.Metric]
) -> keras.Model:
    model_inputs = [keras.Input(name=feature, shape=(1,)) for feature in input_features]

    # Next we concatenate layer to assemble the different inputs into
    # a single tensor which will be given as input to the Dense layer.
    # For example: [input_1[0][0], input_2[0][0]]
    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    model_output = keras.layers.Dense(
        units=1, name="dense_layer", activation=keras.activations.sigmoid
    )(concatenated_inputs)
    model = keras.Model(inputs=model_inputs, outputs=model_output)

    # Call the compile method to transform  the layers into a model that keras can execute.
    model.compile(
        optimizer=keras.optimizers.RMSprop(settings.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
):
    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    # The x parameter of keras.Model.fit can be a list of arrays, where
    # each array contains the data for one feature.
    history = model.fit(
        x=features,
        y=labels,
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


# Experiment 1
settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.01,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

metrics: list = [
    keras.metrics.BinaryAccuracy(
        name="accuracy",
        threshold=settings.classification_threshold,  # type: ignore
    ),
    keras.metrics.Precision(
        name="precision", thresholds=settings.classification_threshold
    ),
    keras.metrics.Recall(name="recall", thresholds=settings.classification_threshold),
    keras.metrics.AUC(num_thresholds=100, name="auc"),
]

plt.rcdefaults()
keras.backend.clear_session()

model = create_model(settings, metrics)

experiment_1 = train_model("baseline", model, train_features, train_labels, settings)

# Shown seperatly because the AUC threshold is 100, and the rest only for the specific threshold.
ml_edu.results.plot_experiment_metrics(
    experiment_1, ["accuracy", "precision", "recall"]
)
ml_edu.results.plot_experiment_metrics(experiment_1, ["auc"])


# Validation
def compare_train_validation(
    experiment: ml_edu.experiment.Experiment, validation_metrics: dict[str, float]
) -> None:
    print("\nComparing metrics between train and validation:")
    for metric, validation_value in validation_metrics.items():
        print("------")
        print(f"Train {metric}: {experiment.get_final_metric_value(metric):.4f}")
        print(f"Validation {metric}:  {validation_value:.4f}")


validation_metrics: dict[str, float] = experiment_1.evaluate(
    validation_features, validation_labels
)
compare_train_validation(experiment=experiment_1, validation_metrics=validation_metrics)

# To Improve
all_input_features = [
    "Eccentricity",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Area",
    "Perimeter",
    "Convex_Area",
    "Extent",
]

# Experiment 2 - All features
settings_all = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.01,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=all_input_features,
)

all_model = create_model(settings_all, metrics)

experiment_2_all = train_model(
    "baseline", all_model, train_features, train_labels, settings_all
)

ml_edu.results.plot_experiment_metrics(
    experiment_2_all, ["accuracy", "precision", "recall"]
)
ml_edu.results.plot_experiment_metrics(experiment_2_all, ["auc"])

validation_metrics_all: dict[str, float] = experiment_2_all.evaluate(
    validation_features, validation_labels
)
compare_train_validation(
    experiment=experiment_2_all, validation_metrics=validation_metrics_all
)

# Now Compare
ml_edu.results.compare_experiment(
    [experiment_1, experiment_2_all],
    ["accuracy", "auc"],
    validation_features,
    validation_labels,
)

# Test
test_metrics_all_features = experiment_2_all.evaluate(
    test_features,
    test_labels,
)
for metric, test_value in test_metrics_all_features.items():
    print(f"Test {metric}:  {test_value:.4f}")
