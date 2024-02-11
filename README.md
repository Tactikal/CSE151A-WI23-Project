# CSE151A-WI23-Project

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
