# GARCH-LSTM-forecasting

This project is about curating a machine learning model that combines econometrics and ML, it can forecast stocks and capture both the short-term and long-term events. Short-term events, such as panic-selling, or long-term events, such as monetary policies. 

## Motivation
From my previous project at [Machine-Learning-Data-Analysis](https://github.com/dancheung555/Machine-Learning-Data-Analysis), I explored CNN and LSTM models, which motivated me to improve, excel, and succeed further. Before starting this project, I was exploring how I could model the difficult parts of time series, such as what seemed to be outliers or noise, but was affected by real-world events. This also ties into what ways I can predict the biggest buys or sells in the stock market, and how to implement a better model to predict these kinds of events. After reading some articles and finding how some of these models work, I will start with the GARCH model combined with the LSTM model.

## Models at play
### GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model
This model is from the category of autoregressive conditional heteroskedasticity (ARCH) models. ARCH models are a family of volatility prediction models. These models focus on how the variance of the current error term depends on the previous time period's error term. This model works well when the error variance follows an autoregressive (AR) model. If the error variance follows an autoregressive moving average (ARMA) model, then the model is a GARCH model.

- **AR** (Autoregressive) **model**: a type of model where the current time period's value is a function of the previous time period's value. In other words, the function's current value depends on its previous period's value. It predicts based on a weighted sum of past values.
- **Heteroskedasticity**: the variance of error terms is not uniform
- **ARCH** (AutoRegressive Conditional Heteroskedasticity) **model**: the variance of error terms is not uniform; instead, it depends on past values of the variance of error terms. In other words, past volatility affects current volatility
- **MA** (Moving Average) **Model**: a type of model where the current time period's value is a function of the previous period's value. It predicts based on a weighted sum of past errors.
- **ARMA** (AutoRegressive Moving Average) **Model**: It combines an AR model with an MA (Moving Average) model. Both AR and MA models predict based on previous time period's values, the AR model is based on the weighted sum of past **values**, while the MA model is based on the weighted sum of past **errors**. Also, do not get ARMA confused with ARIMA (AutoRegressive Integrated Moving Average) models.
  - ARMA models are a simple yet foundational model is learning stochastic processes and time series analysis. This is the crux of time series analysis: past affects current, current predicts future.
  - **ARIMA** (AutoRegressive Integrated Moving Average) **models**: This model uses both past values and past forecast errors. The Integrated part refers to **differencing** in the time series data, which helps handle non-stationary data (non-constant mean and variance). 
- **GARCH** (Generalized Autoregressive Conditional Heteroskedasticity) models: If the error term's variance follows an ARMA model, then it is a GARCH model. It is called generalized because it focuses on both past values and errors, it incorporates the two together.
  - This model is especially well-suited for predicting volatility and returns because it balances risks and returns.

GARCH can help predict the volatility that LSTM models lack, and it also helps by not overshooting the predicted risk since it balances risk and return, making this integral in working with LSTM models.

### LSTM (Long Short Term Memory)
This model is a type of **recurrent neural network** (RNN), which is suitable for holding memory in its cell states. There are 3 layers to this model: input layer, LSTM-cell layer, output layer. The input layer is just the input based on how large each time period is, and the output layer is how many predictions the model will make. I will focus on the LSTM-cell units more in the following description.

The LSTM cell has 3 parts: input, output, and hidden memory. This hidden memory is what allows the LSTM to have long term memory.
- **LSTM cell components**:
  - **Forget gate**: how much of the current cell should forget the current data.
  - **Input gate**: how much of the current cell should remember current data.
  - **Candidate memory gate**: how much of the current memory should be remembered
  - **Output gate**: the combination of the new hidden memory data and the new input data
^This is _just a rough idea_ of an LSTM cell, it is slightly more complicated than this, I suggest searching up the mathematics and workings behind the LSTM cell to fully understand. But to just get a rough idea, an LSTM can hold both short and long-term data by incorporating a memory unit in the cell so it can spit out the short-term data quickly while still retaining the long-term information. At the same time, both of these data are processed to determine how much of the long-term and short-term data should be kept/forgotten through weight and are corrected after each training/testing epoch through back-propagation.

## Data used:
I will download the CBOE Volatility Index (VIX) Historical data and the S&P 500 Data from online

**CBOE Volatility Index (VIX)**: This index determines the market volatility on the S&P 500 index. In other words, it determines how much the S&P 500 will fluctuate in the next 30 days.

- VIX < 20: low volatility, good stability
- 20 < VIX < 30: moderate volatility, normal market environment
- 30 < VIX: High volatility, turbulence, higher chance of price swings

**S&P 500 Index**: this index tracks the stock performance of the 500 leading companies listed on stock exchanges in the United States.

I will download the data from the yfinance package.


