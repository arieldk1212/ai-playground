import torch
import numpy as np
import torch.nn as nn

# Introduction
"""
Model - mathematical relationship derived from data that an 
  ML system uses to make predictions
Supervised, Unsupervised, Reinforcement, GenAI

Supervised Learning - Models can make predictions after seeing lots
  of data with correct answers and then discovering the connections
  between the elements in the data that produce the correct answers.
  We basically tell it the correct answer, the data contains it.
  1. Regression Model - Predicts a numaric value (Weather model that
      predicts the amount of rain.)
  2. Classification Model - The likelihood that something belongs to 
      to a category.
      The output value can be or binary or multiclass.

Unsupervised Learning - Models aim to identify meaningful patterns
  within a dataset, they do that by relaying on a technique called 
  "Clustering" - Organize to groups.
  Differs from Classification because the clusters aren't defined by us.

Reinforcement Learning - Models make predictions by getting rewards or
  penalties based on their actions performed within an environment.
  The system creates a policy that defines the best strategy for
  getting the most rewards.

Generative AI - Class of models that creates content from user input.
  For example: can create unique images, music, jokes..
  Takes a variety of inputs and creates a variety of outputs.
  Text-to-text, Text-to-image, -video, -code, -speech, Image-to-text..
  How? they learn patterns in data with the goal to produce new
  but similar data.
"""

"""
Supervised Learning in detail:
  Core Concepts: Data, Model, Training, Evaluating, Inference.
Data -> Could be made of numbers, pixels, images, prices, weather information..
  Collection of features that try and predict the value of the label.
  There could also be collection of features with no label.
  Datasets need to be large and highly diverse for them to be concidered good!
Model -> The complex collection of numbers that define the mathematical 
  relationship from specific input feature patterns to specific output
  label values. The model discovers these patterns throught training.
Training -> Before we make predictions, it must be trained!
  To train, we give dataset with labeled examples.
  The model finds the best solution by comparing its predicted
  value to the label's actual value.
  Based on the Loss from the comparisons, the model gradually updates
  its solution, it modifies it!
  The user can choose which feature to ignore and run tests.
Evaluating -> We evaluate to determine how well it learned.
  We can achieve that by giving the model only the features without
  the labels, and then check and compare the generated labels from the 
  given dataset to the original label values.

A model needs to be trained to learn the mathematical relationship
between the features and the label in a dataset.
"""

# Linear Regression
"""
 bias = an intercept or offset from an origin, for example: the y-axis
   value.
 y = mx + y
 in ml: y' = b + w1x1 -> could also be more depending on the number
   of features. w1x1 + w2x2 + w3x3...
   engine displacement, acceleration, number of cylinders, horsepower.
   all of these are features, that can predict gas mileage.
   in the equation their position is the x, beacuse its the only
   value that is not calculated during training.
 where:
   y' = the  predicted label = output
   b = the bias - calculated during training
   w1 = the weight of the feature - parameter calculated during training
   x1 = the feature input
"""

"""
Loss
  Describes how wrong are the model's predictions, Measures the distance
  between the model's predictions and the actual labels.
  We only care about the distance, not if the arrow is negative/positive.
  Methods to remove the sign: abs between the value and the prediction,
    Square the difference between the actual value and the prediction.
Types of Loss:
1. L1 Loss -> The sum of the absolute values of the difference
  between the predicted values and the actual value (abs(actual-predicted)).
2. Mean absolute error -> The average of L1 losses across a set of N examples. (average)
3. L2 Loss -> The same as L1 but squared. (pow)
4. Mean squared error -> The same as MAE but squared. (average, pow)
5. Root mean squared error -> The same as MSE but square root. (average, pow, sqrt)

The higher the loss the larger squaring makes it, so is the opposite.
MAE & RMSE are more suitable for human-interpretable.
Example:
  Model:   y' = 34 +(-0.46) * x1;
    Weight = -4.6
    Bias = 34
  Now lets say we expect to get 24, but we got 23.1 after formatting with the label.
  Therefore: (actual - predicted)^2 = (24 - 23.1)^2 = 0.81 -> L2 Loss
Usually given a dataset we have a "normal" value of where it will be located, but
  there can always be unpredicted values, we call them "outliers",
  It can also refer to how far off a model's prediction are from the real values.
"""

"""
Gradian Descent
  Mathematical technique for training a modle that finds the weight and bias that produces
  the lowest loss, done by using calculus.
  It knows by how much to adjust the weight and bias from the loss value.
  It basically repeats this pattern in order to find to most optimal values for the model.
  Multiplies the weight and bias by some constant and runs it again.
To study more we can also take a look at the loss graph and see if the model has converged - 
  meaning that the loss has gotten to a point where it almost doesn't change (flat line).
  What we can take from it is that once the model gets converged, the value at which it gets
  to that state is how many iterations the model needed to be do in order to get the lowest value
  of loss.
  They produce a convex surface (U-Shape).
"""

"""
Hyperparameters
  Variables that control different aspects of training: Learning Rate, Batch Size, Epochs.
  Values that you can control.

Learning Rate -> a float that influences how quickly the model converges, if too low: the model
  can take along time to converge, if too high: it may never converges, but instead bounces
  around with the values of weight and bias that minimize the loss.
  It determines the magnitude of the changes to make for the weight and bias during
  each step of the gradiant descent process.
"""
