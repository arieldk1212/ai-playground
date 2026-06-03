"""
PRETRAINING

Preprocessing -> Getting data across the internet, FineWeb.

Tokenization -> Transforming text into bytes, or something else.
  * See tiktokenizer for example.
  * We combine symbols into 1, to decrease the length.
  * We optimize this by going thru the dataset, finding symbols that are close to each other
    and then we symbol them as 1 token (combine the 2 bytes).
  * For GPT 4 for example there are 100,277 possible tokens.
  * Each token is basically 1 byte, just a unique ID.

  * Downloading and Tokenization are part of the preprocessing step, it happens only once.

* Context -> Sequence of tokens == Context Window -> Limit of how much tokens.
* Batch Size -> The length of the sequence of tokens.
* For nn we pass a Context as input, and as output the NN "guesses" the next token,
  based on linear algebra, 100,277 probabilities for the next token.
  The longer we train the better its adjusting the probabilities.

NN:
  * At the train step we get inputs, and we train with random weights, therefor the output
    is also random, only after the first initial iteration step the nn starts to adjust
    the probabilities of the inputs to that it wont be random.
    "Training a NN is Discovering a setting of parameters that seems to be consistant
      with the statistics of the training set".
    * Input -> Weights -> Pred -> Loss -> Adjust -> Again.
    * Production-grade example of NN: bbycroft.net/llm
    * n_params -> the number of different responses the model can generate for the prompt.
    * Embedding -> Each token has a vector inside the neural network.
    * Inference -> To generate data, just predict one token at a time, and then continue
      to feed back tokens to get the next one.
      Basically when we open chatgpt and talk to it, its inference, it has nothing to do
      with training, all the weights are fixed.
    * Base Model -> Internet <--> text token simulator, to interact with them we can go to
      hyperbolic website.
      if the base model isn't instruct model, its basically just a very expensive auto
      complete model, we can't ask it questions and expect answers.

POST-TRAINING
"""
