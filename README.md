# CSE151A-WI23-Project

Jupiter Notebook: https://drive.google.com/file/d/1RqJJukdvk2iLKsriBnT8vFXMK4Qp7YE7/view?usp=sharing

# Abstract
This dataset contains information about demographic and job-related features of individuals, such as age, work class, education, and occupation. This data can then be used to predict whether a person makes over 50k a year, based solely on these features. Additionally, it can also be used to predict if one or multiple features affects another. We will be doing some form of regression in order to predict whether these features affect one another or overall income.

# Preprocessing
To preprocess our data, we begun with imputation to remove meaningless or null entries, and also renamed some labels to make it easier to unify our data. We also made some slight transforms to the data to further eliminate unnecessary data. We then normalized the data and did a logistic regression using only the numerical data. Further standardization of the normalized numerical data was performed, and another logistic regression was completed to check our results. Finally, we performed both label and one-hot encoding on the categorical variables in our data, and determined that label encoding is a more appropriate approach due to the large number of ordinal categorical features. 
