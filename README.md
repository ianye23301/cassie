Explaining the strategy

I have two main folders - ML way and tree way
The ML way is the neural network - it’s a predictive tool that does a pretty good job at predicting price based on the six factors
Because numerous factors which are a mix of categorical and numerical, using a regression model seemed ok. Also the engineSize had a non-linear effect on the price, so I deemed it justified
The problem with this way is that it’s kind of a black box, you don’t really see trends between things and that might not be sufficient
R squared was about 0.90
The “tree” way solves that problem by using a decision tree, which is a little complicated to explain with regards to the math (i don’t really know it that well either), but it works based on MSE (mean squared error), which is good for our case since we’re doing regression
The tree will identify important features (model, year, etc.) and rank them on their importance, while also serving as a model to predict. These features are shown in Appendix H. Note that model_Focus being important means that model itself is important, since it’s binary. Not all features will be on there since the decision tree may have not deemed them significant. 
We have some knowledge on the ‘trends’, which we get by plotting a bunch of different relationships (Appendix A through F), which validate the important features given to us by the decision tree. 
Some important features end up being redundant, which I explain in Appendix H
The tree actually outperformed the neural network, with an R-squared of 0.94. Impressive, right?

Some technical explanations
For both methods we split the data into a training set and a testing set (80:20). We build the model on the training data and test it on the test set. We make sure to shuffle the data because the data is sorted in some way, and we don’t want that affecting our training.
Upon splitting there’s a few things we need to do. First we normalize the scale of our numerical values so that numbers of larger scale don’t have more weight. For example engine sizes are in the 1-3 range, while mileage is in the thousands range. 
We also need to encode our categorical variables (like model, fuelType) into numbers. We use one-hot-encoding for this. Basically this means creating a column for each option in variable (so for model we create model_focus and model_C class) and then make it a binary (1 for true, 0 for false)
This is basically all process.py does - it defines a class that processes the data for you. It assumes a specific structure of the data that the code will specify. 
Not much explaining to do in the neural network
Training loss = difference between predicted value and real value in training set
Validation loss = while training we see how the model currently fares with test data
These should be similar to prevent overfitting, which isn’t a problem for us.
Validation and test loss graph shown in Appendix I
An epoch is a pass through on the entire dataset. 500 is an arbitrary number but too much more can lead to overfitting. 
R score: 0.9140
For the tree, we save the ‘rules’ that the tree generated into a text file, though it’s not that important and very long. All you need to know is that it shows you which features are most important so it’s more interpretable than a neural network. 
R score: 0.94
