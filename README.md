# Deep-Learning-Finance

Neural Network fundamentals built from scratch using numpy and comparing it with results from PyTorch.
LSTM tested on Financial Time Series Data.
Transformer Architecture applied on Financial Data

## Notebooks

### nn_fundamentals.ipynb
### lstm_financial_timeseries.ipynb
### attention_transformer.ipynb
### NLP_Finance.ipynb (in progress)


## Features

# Neural Networks
- It covers single neuron mechanism
- Sigmoid activation function
- forward + backward propagation
- Binary Cross Entropy Loss

# LSTM
- It covers single LSTM layer (hidden_size = 64) , linear output layer, sigmoid activation funciton with Adam Optimizer
- Data used is Nifty50 and S&P500 from 2010 to 2024
- Number of epochs used were 100 with a batch size of 32

# Transformer

- Architecture: input projection (5→32), positional encoding (sine/cosine), 2-layer TransformerEncoder (4 heads), linear output, sigmoid
- self-attention implemented from scratch in numpy, verified (softmax rows sum to 1)
- Positional encoding before/after comparison: 47.1% → 50.6% test accuracy, loss curve consistently lower with positional encoding
- Confusion matrix finding: model biased toward predicting "Up" (59.7% of predictions), only 52.5% accurate on those calls — suggests the model learned the      overall market trend rather than genuine day-specific signal
- Three visualizations included: loss comparison, price/prediction overlay, confusion matrix
  

## Key Findings

# Neural Networks
- The numpy training loop reduced loss from ~2.35 to ~0.019 over 300 iterations
- PyTorch rebuild converged to a closely matching loss (~0.019), confirming the manual implementation was correct.

# LSTM
- The loss decreased from 0.6898 to 0.1865
- Severe overfitting observed in training data (accuracy - 92.39%)
- Test accuracy came out to be (49.33%)

# Key Lesson
- Complex model does not guarantee better performance on financial time series
- Focus on feature engineering and simpler models can outperform deep learning when data is limited

## Author
Snehil Kejriwal — Economics + B.Tech, BITS Pilani Hyderabad
github.com/snehilkejriwal-exp19
