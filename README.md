# Deep-Learning-Finance

Neural Network fundamentals built from scratch using numpy and comparing it with results from PyTorch.
LSTM tested on Financial Time Series Data.
Transformer Architecture applied on Financial Data

## Notebooks

### nn_fundamentals.ipynb
### lstm_financial_timeseries.ipynb
### attention_transformer.ipynb
### NLP_Finance.ipynb 


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


# NLP
- Data pipeline: Alpha Vantage NEWS_SENTIMENT API, 5465 JPM-related articles (2018-2024), relevance-weighted sentiment scores, daily aggregation via groupby-mean, missing days filled with 0
- Architecture: same Transformer, input_proj expanded to accept 6 features instead of 5
- Methodology upgrade: 5 independent training runs to account for weight-initialization randomness, reporting mean ± std instead of single-run numbers
- Result: mean test accuracy 48.94% ± 1.68% (range 46.3%-51.4%)
- Three visualizations: model comparison bar chart with error bars, price/prediction overlay, confusion matrix

# RAG_Gen AI
- Data source: SEC EDGAR API (no key required, 10 req/sec), Goldman Sachs (GS) 10-K filing, CIK 886982
- Pipeline: HTML fetch → BeautifulSoup cleaning → XBRL noise removal → chunking (~500 words, 338 chunks) → embedding (all-MiniLM-L6-v2, sentence-transformers) → cosine similarity retrieval → grounded generation (Gemini 3.5 Flash)
- Explicit grounding instruction used to prevent hallucination — model told to answer only from retrieved context, explicitly say when the answer isn't available
- Four test cases demonstrating a range of RAG behaviors: full grounded answer (FICC revenue), correct refusal on weak retrieval (business segments), correct refusal despite reasonable similarity (total net revenue — adjacent-but-wrong figures retrieved), partial grounded answer with explicit gap-flagging (MD&A net revenues)
- Key finding: specific/factual questions retrieve more reliably than abstract/conceptual ones; high similarity score indicates topical relevance, not guaranteed answer availability — the LLM's explicit grounding check is what ultimately prevents overconfident wrong answers, not retrieval alone
- Known limitation: XBRL-stripping relies on finding a known header phrase, works for this filing but isn't robust to arbitrary filing formats without adjustment
  

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
