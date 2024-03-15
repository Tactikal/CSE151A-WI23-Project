# CSE151A-WI23-Project
(test modification, will delete afterward)
Jupiter Notebook: https://drive.google.com/file/d/1RqJJukdvk2iLKsriBnT8vFXMK4Qp7YE7/view?usp=sharing

## Dataset
https://archive.ics.uci.edu/dataset/2/adult

## Abstract
This dataset contains information about demographic and job-related features of individuals, such as age, work class, education, and occupation. This data can then be used to predict whether a person makes over 50k a year, based solely on these features. Additionally, it can also be used to predict if one or multiple features affects another. We will be doing some form of regression in order to predict whether these features affect one another or overall income.

## Background
The data was extracted by Barry Becker from the 1994 Census database, and the goal is to determine whether an individual makes over $50,000 a year based on demographic and job-related information.

## Introduction
We were interested in this dataset because it was very robust, and had potential for a variety of different tasks to be completed using it. In addition to the prediction task, we considered exploring whether different variables influence each other (e.g. income vs. marital status, correlation between age/education/occupation); for these tasks, we would perform regression, classification, and/or clustering. Our main objective is to determine how individual features affect each other as well as to get an idea of how our own futures may look. As college students graduating within the next couple years, we are curious to see what types of jobs that we potentially end up in, as well as how our backgrounds and education can predict our future careers. We can also get a better idea of how and where underprivileged demographics fall behind, and consider what can be done in order to uplift those groups. By the completion of this project, we hope to gain a better understanding of how people's backgrounds and education play a role in their future.

## Preprocessing
To preprocess our data, we begun with imputation to remove meaningless or null entries, and also renamed some labels to make it easier to unify our data. Additionally, we made some slight transforms to the data to further eliminate unnecessary data. We then normalized the data and did a logistic regression using only the numerical data. Further standardization of the normalized numerical data was performed, and another logistic regression was completed to check our results. Finally, we performed both label and one-hot encoding on the categorical variables in our data, and determined that label encoding is a more appropriate approach due to the large number of ordinal categorical features. 

## Data Visualization 
We did basic data visualization in order to gain a broad understanding of our data, and see how categorical features are distributed to note any sample size bias that may occur.

## First Model: Logistic Regression
Our model is consistent throughout the testing and training predictions, and the log loss values are very close. We have high precision and recall for Class 0 (<=50K) but lower precision and significantly lower recall for Class 1 (>50K), likely due to the class imbalance (6000 samples for Class 0 but 2000 for Class 1).
The model is well fit with good generalization, but definitely could benefit from reducing the amount of samples for class 0. We think that it might fit within the ideal range for model complexity, as errors on both test data and train data are low.

For our next 2 models, we are thinking of using SVM classification to classify who gets what income. We plan on using SVM classification specifically because it is very useful in binary classification and, more importantly, there are methods for us to able to penalize misclassification of the minority class. Additionally, for our second model, we can use a Keras ANN since those are often used to perform prediction / classification tasks with high effectiveness.

We can conclude that our first model has a high accuracy rate for predicting if someone makes less than or equal to 50k, but is much weaker for predicting if they make more than 50k. This was because there was less data for those making over 50k, which led to worse predictions. For future models, we may try making adjustments to the data to reduce the impact of the unbalanced data, or using models that perform relatively well regardless of the number of input entries. We could also perform multiple training runs exclusively on the entries that have income over 50k, to make sure that the model gets to work with it more.

## Second Model: Logistic Regression
We saw a slight increase in accuracy from our first model, with greatly increased precision and recall for Class 1. Similarly to the first model, it would fit within the ideal range for model complexity due to the fact that there is a small difference between train and test errors. However, it may lean more towards overfitting since the test error appears to be a bit higher than that of the train error.

We performed cross validation and hyperparameter tuning. Grid search seems to have created a stable model that has similar levels of test and train error regardless of number of features. However, the training and testing error are both at least above 0.4. More information needs to be uncovered before we can come to a conclusion on what is good. We will do K-fold cross-validation next.

To figure out what features might be most important, we could use decision trees. Overfitting seems like an issue, however, so careful planning would need to be done. Naive Bayes is not likely to be a good idea because most of the features are likely not independent. If we use it to figure out which features correlate with the target, we would have to carefully choose only the most independent features. We could still try it, but we expect bad results. We are also interested in K-Nearest Neighbors classification because of the unsupervised nature of the approach. It seems useful for exploration because it makes less assumptions about the data. We might be able to use it to lead up to other forms of classification, or as an additional verification measure for decision tree learning. Finally, we are interested in SVM classification for the simple nature of the model. We expect to achieve relatively quick results using it with some polynomial transformations. In the end, we may use all of these methods (minus Naive Bayes) and take an average, using majority vote to generalize the model. Some models might be better for some features, and we could leverage it for much benefit.

In conclusion, we did not find significant improvements by using a neural network. We found good accuracy, but significant overfitting. We believe neither model is truly sufficient, so we would like to try several others.
