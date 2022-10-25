#Predicting the future price of the market index S&P 500 using Linear Regression and K Nearest Neighbour

#Introduction:
#Computer science and machine learning has been a big part of the stock market trading for a few decades now and a great variety of mathematical indicators
#for stock trading has been developed over these past years. These indicators are very often based on the historical prices of the stocks or indexes which 
#are being traded and the methods are used by many professionals. It is often said that the best results in stock trading comes when multiple different indicators 
#are indicating the same thing, which is why machine learning and deep learning methods can be very useful to predict the stock market. More detailed information 
#about indicators and theory of trading can be found for example in Investopedia (Folger 2022). This machine learning project is predicting the close price for the 
#market index S&P 500. This could be useful information, when deciding whether to sell the stock or buy it on that given day.
#Outline of this report consist of Introduction, Problem Solving, Methods, Results and Conclusions. Section “Problem Solving” defines what exactly this project is
#trying to do as well as it explains the origin of the dataset. In section “Methods” is explained how the problem is approached using the data set. In sections 
#“Results” and “Conclusions” the results are presented and discussed.

#Problem Formulation:
#Even though the best possible result would be achieved by combining multiple different indicators for predicting the future close price S&P500 index, 
#because of tight schedule of this course, this application will be using a very simple strategy. In addition to historical price data, also two extra 
#features will be calculated and used in the prediction process: 1) repetitive peak points of the index (where the price movement has stopped and turned around,
#either up or down) and 2) price variability during each day (datapoint), which is simply a difference between highest and lowest price of the day. These values 
#can be useful because, when price movement stops at a certain level multiple times it becomes so called resistance level and it is likely that movement will stop
#there in future as well. Usually near these resistance levels the variability in prices decreases.
#Prediction is done by using supervised linear regression and supervised K nearest neighbour (kNN) regression methods. The features are the historical prices of 
#S&P 500 index which are (continuous numerical), calculated peak points (binary) and variability of prices during each day (continuous numerical). The label is 
#future close price for the index which is continuous numerical.

#Methods:
#The daily stock market data was downloaded from website WSJ Market (2022) as csv file. The data includes 1258 data points and each data point represent one day 
#during 3rd of October 2017 – 3rd of October 2022. There are opening, close, highest, and lowest prices listed for each day. Opening and closing prices means the 
#price in US dollars where the trading day started and ended, and the highest and lowest prices are the highest and lowest prices of the day.
#Pre-processing of the data included formatting data type for dates, adding a column which has the difference between "High" and "Low", and finding the turning points, where the price has either bottomed or peaked and save this as binary data column (1 = yes, 0=no).
#Linear regression will be used in this study because stock price data is continuous and numerical but also because over a long-term period the increasing price 
#movement has been nearly linear. K nearest neighbour is another regression (also used as categorical) method which will be used as a comparison to linear regression.
#Mean squared error is commonly used with linear regression, so it will be adapted in this project as well. Also, r2 score will be calculated to estimate the model. 
#Both methods measure how far observed values differ from the average of predicted values.
#The data is split roughly into 60% training, 20% validation and 20% test data randomly without shuffle. This is because the data set is not too big and as much data 
#points as possible should be used for the validation and test sets. Ideally data would be split by day, using the oldest data for training and the latest data for 
#validation and testing but because it is not possible to validate kNN method with this kind of datasets. The method needs neighbouring data around to make 
#predictions and therefore the split needs to be random.

#Results:
#The loss function errors mean squared error and R2 are presented in the tables 1 and 2 below for both methods. R2 scores are very good for both methods, nearly 
#perfect. This can be seen also in Figure 1, where the predicted close prices are plotted with the actual prices. Both graphs look similar.
#However, there is some variance in mean squared errors. The validation data error relatively to training data is much larger for the kNN method than it is for the 
#linear regression. This is a sign of overfitting. Even though the training data error is bigger for linear regression, the validation data error is much smaller 
#than for kNN method. Therefore, linear regression is the final chosen method.

#Table 1. Errors for linear regression

#          Mean squared error    R2 Score 
#---------- -------------------- ---------- 
#Training    217.422               0.999541 
#Validation  198.977               0.999574 
#Test         181.682              0.999604

#Table 2. Errors for k Nearest Neighbour 
#            Mean squared error   R2 Score
#---------- -------------------- ---------- 
#Training     206.869              0.999563 
#Validation   463.925              0.999006

#The test error of the final chosen method is 181.6 USD which is quite big, if used in trading. In percentage it means 3.5-5% of the value of the whole index. 
#Normally the index moves only 1-2% in a day, so the prediction is not very useful in real life. There are no signs of overfitting, since the validation error 
#is little bit smaller than the training error.

#Conclusion:
#This method was found to be useless for real life trading and multiple problems were found. As mentioned in the previous chapter, the error margin is too big, 
#and that is why it doesn’t make sense to use the model in real trading. Another problem is that the model uses data such as highest and lowest price which can 
#change during the day. Using historical data, which won’t change, makes the model much better than it is. This is because the highest, lowest, and open prices 
#are highly correlative with the close price, and if these values change dramatically during the trading day, also the close price changes dramatically and the 
#error becomes bigger. Finally, third problem found was the historical data is always overrepresented and the most recent data is underrepresented. This is a 
#problem because the interest is in the most recent data. This could be solved by balancing the data, but there’s risks in it too because it is a difficult to 
#say, what data should be highlighted and where to draw the line of “old” and “new” data.
#Therefore, some other features should be used either in addition or on its own. These features need to be something that are available and not changing in time. These features could be for example fundamental factors which can be measured, such as number of news related to the index.

#References:
Boxer S. 2019. S&P 500 Stock Price Prediction Using Machine Learning and Deep Learning. Medium. 
#Viewed 04/10/2022: https://medium.com/shiyan-boxer/s-p-500-stock-price-prediction-using-machine-learning-and-deep-learning-328b1839d1b6

#Folger J. 2022. Using Trading Indicators Effectively. Investopedia. 
#Viewed 03/10/2022: https://www.investopedia.com/articles/trading/12/using-trading-indicators-effectively.asp

#WSJ Markets, S&P 500 Index. 2022. Viewed in 03/10/2022:
#https://www.wsj.com/market-data/quotes/index/SPX/historical-prices
