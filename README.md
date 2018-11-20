# Neural Networks for Time Series Forecasting

In this project a multilayer perceptron is built using scikit-learn's MLPRegressor, to forecast future natural gas prices. 
The aim is to forecast the gas price for two weeks into the future based on the prices of the past 8 weeks.

## Dataset

The original dataset is available via https://www.eia.gov/dnav/ng/hist/rngwhhdW.htm, the dataset shows the weekly gas prices between 29/9/2017 and 1/10/1997. 

Please note that the original dataset is arranged in reverse chronological order.

The original file has been processed to create a time series file “time_series_8_2.csv”. In the time series file, the prices have been re- formatted so that each line includes the prices of 10 consecutive weeks.

The first 8 prices will be used as input and the 9th and 10th prices as targets.

## Model Configuration

### Attempting different numbers of hidden layers and number of nodes
* #### First, experimenting with the number of nodes within a hidden layer showing the score with each number of nodes:

    300:
    0.794623716254

    600:
    0.800270526383

    900:
    0.832937038159

    so, 900 was choosen as the number of nodes, as it resulted in the highest score.

* #### Then, expiremnting with Number of Layers with 900 as number of nodes (obtained from the above expiremnt) with their corresponding scores:

    15:
    0.806816500115

    20:
    0.821570534339

    25:
    0.852278055001

    Thus, 25 layers were choosen, as it resulted in the highest score. 

Generally, we don't need all these layers, but since we have a slightly high dimensional input (9), it is possible. As adding layers solves higher dimensions.

Thus, our optimal score so far is 0.852278055001.

### Attempting 2 different learning rates

The initial learning rate of 0.001 was used, and it provided the best results, when it was increased or decreased, the score decreased.
When it was increased to 0.001, it resulted in a score of 0.828010829428.

#### Reasoning:
Increasing the learning rate helps the model learns faster but with the risk of the model skipping the global minima or some better local minima, which we assume was the case.
When it was decreased to 0.0001, it resulted in a score of 0.813443956902.

Generally decreasing the learning rate should help the model reach global minimum, but in this case it became too small that it got stuck in some local minima.

### Extra Tuning
Stochastic training was used, but when expirementing with mini-batch showed an increase in the score, with batch size=13, and score became 0.8860549529.

## Observations on the progress of both training and validation errors 
During training, it was observed that the error decreased each iteration, but sometimes it increased slightly and then decreased again, and this is expected during training, because the model is still learning and updating weights according to the instances it sees.

The validation score generally increased through out training, sometimes decreeing slightly, eventually the validation score does not change and this results in stopping the training as the model starts to over fit.

This is achieved by using early stopping and the validation fraction parameter, and setting it to 0.2, which takes 20 percent of the training set, to check if the model is generalizing, once the validation set score does not change, it stops the training, because that indicates that the model started memorizing the data.
After, applying overfitting measures.

The final optimal score is 0.891005553354.

## timeseries_4_1 versus timeseries_8_2
comparing the performance score of a timerseries that predicts into 1 week in the future, based on the prices of the past 4 weeks, with the timeseries that predicts 2 weeks into the future based on the prices of the past 8 weeks.
 
Because we are predicting 1 week in the future, instead of 2, also because the input to the model became 4 instead of 8, which reduces the dimensionality of the input and thus requires less hidden layers.

By reducing layers from 25 to 7 in time_seris_4_1, the score became 0.931027060505, when it was 0.891005553354 in time_series_8_2. Thus, the performance increased with less number of hidden layers incase of time_seris_4_1.

## Libraries
sklearn

numpy

panda
