<div class="content">

## Machine Learning: Log

Day-by-day notes as I study machine learning.


### 84. 2018-12-04


- [] [CS231n: Module 2: Convolutional Neural Networks](http://cs231n.github.io/)
- [] [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [] [Data Loading and Processing Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [] [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)



### 83. 2018-12-03

Deep Learning:

- [Human Protein Atlas Image Classification Kaggle](https://www.kaggle.com/c/human-protein-atlas-image-classification): Classifying subcellular protein patterns in human cells: 
  - documented work so far for collaborator
  - discussed possible next steps and ways forward

### 82. 2018-11-29

Deep Learning:
- Fast.ai Lesson 6 – RNNs: Interpreting embeddings; RNNs from scratch 
- Fast.ai Lesson 7 – CNN Architecture: Resnets from scratch 


### 80. 2018-11-28

Deep Learning:
- Fast.ai Lesson 5 – NLP: Collaborative filtering; Inside the training loop
- [Human Protein Atlas Image Classification Kaggle](https://www.kaggle.com/c/human-protein-atlas-image-classification): Classifying subcellular protein patterns in human cells: 
  - **Leaderboard Score: 0.474 (Top 10% of 1200)**
  - developed training loop to get a good baseline
  - implemented data processing scripts

### 79. 2018-11-27

- [Human Protein Atlas Image Classification Kaggle](https://www.kaggle.com/c/human-protein-atlas-image-classification): Classifying subcellular protein patterns in human cells: 
  - preprocessing
  - setup pipeline
  - initial model



### 78. 2018-11-26

Deep Learning:
- Fast.ai Lesson 4 – Embeddings: Structured, time series, & language models
- Fast.ai Lesson 3 – Overfitting: Improving your image classifier

Maths Practice:
- Linear Algebra [Lemma](https://www.lem.ma/) 

### 77. 2018-11-23

Deep Learning:
- [Plant Seedlings Kaggle](https://www.kaggle.com/c/plant-seedlings-classification): Determining the species of a seedling from an image: 
  - **Leaderboard Score: 0.97858 (Top 18% of 836)**
  - Feature engineering
  - Data augmentation from scratch

- Fast.ai Lesson 2 – CNNs: Convolutional neural networks applied

Maths Practice:

- Linear Algebra [Lemma](https://www.lem.ma/) 

### 76. 2018-11-22

Deep Learning:

- [Plant Seedlings Kaggle](https://www.kaggle.com/c/plant-seedlings-classification): Determining the species of a seedling from an image: 
  - Baseline implementation achieves Leaderboard Score: 0.96473 (96.4% accuracy)
  - Hyperparemeter tuning
  - training pipeline optimisation

Maths Practice:

- Linear Algebra [Lemma](https://www.lem.ma/) 

### 75. 2018-11-21

Deep Learning:

- Fast.ai Lesson 1 – CNNs: Recognizing cats and dogs
- [Plant Seedlings Kaggle](https://www.kaggle.com/c/plant-seedlings-classification): 
  - Setup a pipeline for classifying 14 species of plants from images of seedlings
  - Data exploration



Maths Practice:

- Linear Algebra [Lemma](https://www.lem.ma/) 

### 74. 2018-11-20

Deep learning:

- Setup a remote GPU server optimised for deep learning

Maths practice:

- Linear algebra revision: [Lemma](https://www.lem.ma/) 1.1 Introduction
- Probability: I watched first few lectures from Mathmonk, but didn't find it very engaging so don't think I'l pursue that course


### 73. 2018-11-19

Met with study partner and planned curriculumn for studying deep learning.


Set goal:
- To implement in code 4-5 research papers that describe important concepts in deep-learning.

Decided to:
- Complete part 1 of [fast.ai mooc](https://course.fast.ai/) as an introduction to deep learning
- Learn PyTorch
- Start by implementing papers that describe concepts like Reccurrent Neural Netowrks (RNNs), and LSTMs



### 72. 2018-11-09

Introduction to Statistical Learning: ch9: **Support Vector Machines** Conceptual:

- numpy implementation of maximal margin classifier
- linear and non-linear decisin boundaries
- [Notebook: Support Vector Machines – Conceptual](https://github.com/a-martyn/ISL-python/blob/master/Notebooks/ch9_support_vector_machines_conceptual.ipynb)

### 71. 2018-11-08

Introduction to Statistical Learning: ch9: **Support Vector Machines** Applied:

- SVM in context of minimal linear seperation
- Application of SVM to real datasets
- reducing training time, preprocessing for SVM
- [Notebook: Support Vector Machines – Applied](https://github.com/a-martyn/ISL-python/blob/master/Notebooks/ch9_support_vector_machines_applied.ipynb)

### 70. 2018-11-07

Introduction to Statistical Learning: ch9: **Support Vector Machines** Applied:

- dataset simulation
- SVM kernel selection 
- [Notebook: Support Vector Machines – Applied](https://github.com/a-martyn/ISL-python/blob/master/Notebooks/ch9_support_vector_machines_applied.ipynb)

### 69. 2018-11-06

Introduction to Statistical Learning: ch9: **Support Vector Machines** Labs:

- Tuning support vector machines
- ROC curves
- Multi-class SVM
- [Notebook: Support Vector Machines – Labs](https://github.com/a-martyn/ISL-python/blob/master/Notebooks/ch9_support_vector_machines_labs.ipynb)


### 68. 2018-10-30

##### Day 2: Predicting and preventing bursts - Data Hackathon (South East Water)

- Debugged our preprocessing pipeline
- Fit a model and achieved some pretty exciting results
- Presented our results and earned a special mention
- Check our [slides here](https://docs.google.com/presentation/d/1kaTD7jr0DhJR7IDXVY0lVQ7-jD_r_qhhgqzK9_5nKRc/edit?usp=sharing)


### 67. 2018-10-29

##### Day 1: Predicting and preventing bursts - Data Hackathon (South East Water)

I entered this [data science competition](https://www.eventbrite.co.uk/e/predicting-and-preventing-bursts-data-hackathon-south-east-water-tickets-51138148579#) along with my study partner. 

- Started with some initital data exploration. Easily the largest dataset I've come across yet with 3 billion observations.
- Identified a **Goal**: To help the South East Water team understand:
  - Will there be a burst tomorrow?
  - If so, where?
- Designed an initial feature set with 10-day lag variables for flow by geographic area
- Set about pre-processing
- Hit some scale issues to work around
- Lef pre-processing running over night 




### 66. 2018-10-23

Introduction to Statistical Learning: ch8: **Tree-Based Methods** Applied:

- Comparison of tree based methods with KNN, regression and the lasso.
- [Notebook: ch8_tree_based_methods_applied.ipynbb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch8_tree_based_methods_applied.ipynb)

### 65. 2018-10-22

Introduction to Statistical Learning: ch8: **Tree-Based Methods** Applied:

- Application of tree based methods – including Tree Pruning, Bagging, Random Forest, Boosting – to various datasets
- [Notebook: ch8_tree_based_methods_applied.ipynbb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch8_tree_based_methods_applied.ipynb)

### 64. 2018-10-19

Introduction to Statistical Learning: ch8: **Tree-Based Methods** Labs:

- Review of tree based methods including Tree Pruning, Bagging, Random Forest, Boosting
- [Notebook: ch8_tree_based_methods_labs.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch8_tree_based_methods_labs.ipynb)

### 63. 2018-10-18

Introduction to Statistical Learning: ch7: **Moving Beyond Linearity** Applied:

- [Notebook: ch7_moving_beyond_linearity_applied.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch7_moving_beyond_linearity_applied.ipynb)


### 62. 2018-10-17

Introduction to Statistical Learning: ch7: **Moving Beyond Linearity** Applied:

- GAMs
- implemented backfitting algorithm from scratch
- comparison of backfitting and multivariate OLS in simulated setting
- [Notebook: ch7_moving_beyond_linearity_applied.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch7_moving_beyond_linearity_applied.ipynb)

### 61. 2018-10-16

Introduction to Statistical Learning: ch7: **Moving Beyond Linearity** Applied:

- Analysis of Variance (ANOVA)
- Comparison of Anova with Cross-validation for model selection
- polynomial regression
- step function regression
- cubic spline regression
- natural spline regression
- [Notebook: ch7_moving_beyond_linearity_applied.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch7_moving_beyond_linearity_applied.ipynb)

### 60. 2018-10-15

Introduction to Statistical Learning: ch7: **Moving Beyond Linearity** Lab:

- Ch7 Lab
- univariate polynomial regression
- analytic confidence intervals
- bootstrapped confidence intervals
- comparison of splines in univariate setting
- comparison of GAM configurations with ANOVA
- [Notebook: ch7_moving_beyond_linearity_labs.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch7_moving_beyond_linearity_labs.ipynb)


### 59. 2018-10-12

Introduction to Statistical Learning: ch6: **Linear Model Selection and Regularization** Applied:

- Analysis to compare linear models
- Ch7 Reading
- [Notebook: ch6_linear_model_selection_and_regularisation_applied.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_applied.ipynb)

### 58. 2018-10-11

Introduction to Statistical Learning: ch6: **Linear Model Selection and Regularization** Applied:

- detailed comparison of model selection techniques in context **real dataset** to predict the number of applications received by U.S. colleges. Techniques considered:
  - lasso
  - ridge regression 
  - backward stepwise
  - PCR
  - PLS
- [Notebook: ch6_linear_model_selection_and_regularisation_applied.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_applied.ipynb)

### 57. 2018-10-11

Introduction to Statistical Learning: ch6: **Linear Model Selection and Regularization** Applied:

- Detailed comparison of model selection techniques in a **simulated** setting.  Techniques considered:
    - lasso
    - ridge regression 
    - backward stepwise
    - PCR
    - PLS
- [Notebook: ch6_linear_model_selection_and_regularisation_applied.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_applied.ipynb)


### 56. 2018-10-10

Completed Introduction to Statistical Learning: ch6: **Linear Model Selection and Regularization** Labs:

- implemented principle components analysis
- implemented principle components regression
- implemented partial least squares
- comparison of principle components regression partial least squares in predicting salaries of baseball players
- [Notebook: ch6_linear_model_selection_and_regularisation_labs.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_labs.ipynb)


### 55. 2018-10-09

Working through Introduction to Statistical Learning: ch6: **Linear Model Selection and Regularization** Labs:

- implemented backward stepwise selection from scratch
- compared results from best subset, backward stepwise and forward stepwise selection using cross validation
- implemented k-fold cross validation from scratch
- completed Lab 1
- Lab 2: comparison of ridge regression and the lasso with k-fold cross validation
- [Notebook: ch6_linear_model_selection_and_regularisation_labs.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_labs.ipynb)

### 54. 2018-10-08

Read an Introduction to Statistical Learning: ch7: **Moving Beyond Linearity**. 

Working through chapter 6 Lab 1: Subset Selection Methods:

- implemented best subset selection from scratch
- implemented forward stepwise selection from scratch
- [Notebook: ch6_linear_model_selection_and_regularisation_labs.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_labs.ipynb)

### 53. 2018-09-28

ISL considers ridge regression and the Lasso from both the frequentist and the bayesian perspectives. Although ISL has introduced bayesian statistics and some probability theory, there is clearly much more to this subject.

Researched: what are the best resources for learning probability theory in more depth? 

- Probability Theory: The Logic of Science
    - Lectures: https://www.youtube.com/user/elfpower/videos
    - Book: https://books.google.co.uk/books/about/Probability_Theory.html?id=tTN4HuUNXjgC&source=kp_book_description&redir_esc=y

- Statistical Rethinking
    - Lectures: https://www.youtube.com/playlist?list=PLDcUM9US4XdM9_N6XUUFrhghGJ4K25bFc
    - Book: https://www.amazon.com/Statistical-Rethinking-Bayesian-Examples-Chapman/dp/1482253445
    - Recommendation: https://www.reddit.com/r/statistics/comments/85hxgt/bayesian_statistics_courserecommendation/


### 52. 2018-09-27

An Introduction to Statistical Learning: ch6: **Linear Model Selection and Regularization**. 

Completed conceptual exercises:

- best subset, forward stepwise, backward stepwise selection
- the lasso and ridge regression
- coefficient penalties
- [Notebook: ch6_linear_model_selection_and_regularisation_conceptual.ipynb](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_conceptual.ipynb)

### 51. 2018-09-26

Revision day. Met with study partner and reviewed our solutions to all exercises from chapters 3-5 of An Introduction to Statistical Learning. Discussed and compared intuitions.


### 50. 2018-09-25

Read ISL ch5: Linear Model Selection and Regularization. 


### 49. 2018-09-20

An Introduction to Statistical Learning: ch5: **Resampling Methods**. 

Working through applied exercises:[My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch5_resampling_methods_applied.ipynb)

- implemented LOOCV from scratch
- dataset simulation
- compared bias variance trade-offs of validation approaches
- bootstrap advanced applications

### 48. 2018-09-19

An Introduction to Statistical Learning: ch5: **Resampling Methods**. 

Working through applied exercises: [My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch5_resampling_methods_applied.ipynb)

- estimating test error
- validation sets
- implemented bootstrap from scratch
- bootstrap (to estimate standard errors for logistic regression)


### 47. 2018-09-18

An Introduction to Statistical Learning: ch5: **Resampling Methods**. 

Completed conceptual exercises: [My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch5_resampling_methods_conceptual.ipynb)

- probability theory review
- probability theory and the bootstrap sampling technique
- k-fold cross validation
- comparison of validation approaches

### 46. 2018-09-17

Studied Introduction to Statistical Learning: ch5: **Resampling Methods**.



### 45. 2018-09-16

An Introduction to Statistical Learning: ch4: **Classification**. 

Completed applied exercises: [My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch4_classification_applied.ipynb)

- Logistic regression to classify automobiles by fuel efficiency
- Linear discriminant analysis (LDA)
- Quadratic discriminant analysis (QDA)
- K-nearest neighbour
- Comparison of methods
- log transforms


### 44. 2018-09-15

An Introduction to Statistical Learning: ch4: **Classification**. 

Working through applied exercises: [My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch4_classification_applied.ipynb)

- Logistic regression to predict stock returns
- Confusion matrix
- Logistic regression statistics, false positive, true negatives etc.


### 43. 2018-09-14

An Introduction to Statistical Learning: ch4: **Classification**. 

Completed conceptual exercises: [My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch4_classification_conceptual.ipynb)

- simplification of cost function for logistic regression
- bayes classifier
- proof that babes classifier is not linear for Quadratic Discriminant Analysis
- demonstration of curse of dimensionality for KNN
- Comparison of LDA and QDA
- Bayes theorem 
- odds


### 42. 2018-09-13

An Introduction to Statistical Learning: ch3: **Linear Regression**. 

Completed applied exercises:

[My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch3_linear_regression_practical.ipynb)

### 41. 2018-09-12

An Introduction to Statistical Learning: ch3: **Linear Regression**. 

Working through applied exercises:

[My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch3_linear_regression_practical.ipynb)

### 40. 2018-09-10

An Introduction to Statistical Learning: ch3: **Linear Regression**. 

Working through applied exercises:

[My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch3_linear_regression_practical.ipynb)

### 39. 2018-09-8

An Introduction to Statistical Learning: ch3: **Linear Regression**. 

Working through applied exercises:

[My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch3_linear_regression_practical.ipynb)


### 38. 2018-09-07

Working on practical exercises from chapter 3 of An Introduction to Statistical Learning. [My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch3_linear_regression_practical.ipynb).

- implemented several key statistics from scratch to bolster my understanding
- emulated R's powerful lm().plot() functionality in python, it wasn't available

Thoughts: 

I'm suprised how much work I had to do to emulate R's lm().plot() functionality. The resultant plots seem really useful, particularly as they provide insight for multivariate models, and anomaly detection in that context. Perhaps worth a blog post to see if useful to others?


### 37. 2018-09-06

Completed exercises on conceptual topics from chapter 3 of An Introduction to Statistical Learning. [My notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch3_linear_regression_conceptual.ipynb).

Thoughts:

I've noticed that there are different mathematical approaches to linear regression. Hastie and Tibshirani's notation includes a lot of summations, perhaps to avoid linear algebra being pre-requisite, but I find the vectorised forms more intuitive.

### 36. 2018-09-05

Study Linear Regression, chapter 3 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/). 

- Multiple Linear Regression
- Qualitative predictors
- Extensions of the linear model
- Potential problems
- Non-parametric regression: KNN regression


Thoughts:

Picked up some ideas on how to improve my House Prices kaggle submission, including:

- Interaction variables p89
- Outlier detection by residual plots p97

Had an idea for project using cluster analysis / unsupervised methods for classification of head impact during sport. I've ordered a 9-DOF sensor to hook up to my arduino and gather some data.


### 35. 2018-09-03

Study Linear Regression, chapter 3 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/). 

- Simple Linear Regression
- Multiple Linear Regression

Cross referenced with [Stanford CS229 lecture 2](https://see.stanford.edu/Course/CS229/54) and Murphy's *Machine Learning: A Probabilistic Perspective*.



### 34. 2018-09-02

Today I did the applied exercises from chapter 2 of ISL. Here's [my notebook](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch2_statistical_learning.ipynb). Good exercise in preprocessing and exploration of datasets, sharpening up in Pandas and Matplotlib / Seaborn.


### 33. 2018-09-01

- Read chapters 1 & 2 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/) (ISL)
- Completed [conceptual exercises](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch2_statistical_learning.ipynb) from book


### 33. 2018-08-31

Next up a tour of supervised learning techniques. I've decided to take the Stanford course on [Statistical Learning](https://lagunita.stanford.edu/courses/HumanitiesSciences/StatLearning/Winter2016/about) by Trevor Hastie and Robert Tibshirani. My worfklow for each chapter will be:

1. Watch video lectures and complete online quizzes
2. Review chapter in the accompanying textbook [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), to get more detail
3. Complete exercises from book

Today I completed the video lectures for chapters 1 sand 2.




### 32. 2018-08-28

Kaggle Competition – [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)


- implemented Lasso regularisation model to gain 400 places in leaderboard
- met with study buddy to compare notes and revise
- revised regularisation and discussed learnings from Kaggle progress so far

| **Leaderboard Results** |  Description | CV  | Public Score | Position | 
| --- | --- | --- | --- | --- |
| v1 | Basic LinReg & feature engineering | 0.14698 | 0.13500 | 1825/4394 |
| v2 | Feature normalisation | 0.13360 | 0.13170 | ▲ 1610/4394 |
| v3 | Outlier anomaly detection | 0.11436 | 0.13227 | ▼ |
| v6 | Lasso | 0.12059 | 0.12361 | ▲ 1182/4394  |


Thoughts:

This Kaggle competition has been excellent opportunity to learn about the more practical aspects of machine learning beyond algorithms. Feature engineering and Cross-validation have stuck out as important areas that are not well covered by theoretical study.

I feel like I've taken my Kaggle score as far as I can using a simple linear model and feature engineering. Next I'm going to learn about more advanced models and techniques. 


See my [notebook here](https://www.kaggle.com/alanmartyn/linear-regression)

### 31. 2018-08-27

Kaggle Competition – [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Today I looked at improving my results by removing anomalies from my training set. My hypothesis was that removing anomalies would allow my model to better fit the majority of data, at the cost of a worst fit for anomalies. My assumption here was that the error in fitting anomalies would be outweighed by better fit for majority of datapoints.

- reviewed fundamental statistical measures such as Kurtosis
- learned about Dbscan and Isolation Forest techniques for anomaly detection in high-dimensional settings, a useful reference [here](https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561).
- reviewed Sklearn documentation and user guides on Isolation Forest
- Implemented Isolation Forest
- Observed over-fitting improvement but no measurable improvement on test data – highlighting the importance of a good CV :(

| **Leaderboard Results** |  Description | CV  | Public Score | Position | 
| --- | --- | --- | --- | --- |
| v1 | Basic LinReg & feature engineering | 0.14698 | 0.13500 | 1825/4394 |
| v2 | Feature normalisation | 0.13360 | 0.13170 | ▲ 1610/4394 |
| v3 | Outlier anomaly detection | 0.11436 | 0.13227 | ▼ |


See my [notebook here](https://www.kaggle.com/alanmartyn/linear-regression)

### 30. 2018-08-26

Kaggle Competition – [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Today I focused on minimising exponential characteristics in to make relationships in the dataset more linear. This allows my simple linear model to fit with reduced error. This jumped me 200 places in the leaderboard.


| **Leaderboard Results** |  Description | CV  | Public Score | Position | 
| --- | --- | --- | --- | --- |
| v1 | Basic LinReg & feature engineering | 0.14698 | 0.13500 | 1825/4394 |
| v2 | Feature normalisation | 0.13360 | 0.13170 | ▲ 1610/4394 |

See my [notebook here](https://www.kaggle.com/alanmartyn/linear-regression)

### 29. 2018-08-25

Kaggle Competition – [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

| **Leaderboard Results** |  Description | CV  | Public Score | Position | 
| --- | --- | --- | --- | --- |
| v1 | Basic LinReg & feature engineering | 0.14698 | 0.13500 | 1825/4394 |

- My first submission to leaderboard using the most basic Multivariate Linear Regression model. I'm quite pleased with the result given that this is using a basic model. My efforts in feature engineering seem to have paid off.
- Read [In Depth: Linear Regression](https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html) from VanderPlas' book

See my [notebook here](https://www.kaggle.com/alanmartyn/linear-regression)

### 28. 2018-08-24

Kaggle Competition – [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

- Reviewed Andrew NG's advice on model validation
- Read [Hyperparameters and Model Validation](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html) chapter from Jake VanderPlas
- Decided that a solid cross-validation (CV) is crucial
- bolstered understanding by implementing my own hold-out and cross-validation functions
- reviewed cross validation features in SkLearn
- tested above CV's and chose method 


Thoughts:

Cross-validation like feature engineering is another topic that seems critical to practical application of ML, yet is not widely covered in the literature I've seen so far. The current 1 Kaggler, Shubin Dai (bestfitting), puts a lot of weight on designing good CVs as part of his approach described in this [interview](http://blog.kaggle.com/2018/05/07/profiling-top-kagglers-bestfitting-currently-1-in-the-world/)

During my own experimentation I observed surprisingly high variation between different approaches. 

See my [notebook here](https://www.kaggle.com/alanmartyn/linear-regression)

### 27. 2018-08-23

Today I focused on visualisation and feature engineering of data for the Kaggle competition – [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

- indexed data by statistical data types using Pandas
- applied numeric mappings for ordinal data
- OneHot treatment of categorical data
- thought carefully about how to handle missing data
- exploratory analysis with Matplotlib and Seaborn

Thoughts:

Spending time on visualisation helped me to think clearly about the data. The ability to knock out a quick visualisation to check my intuition seems invaluable. 

Feature engineering can be quite laborious, but seems like one of the most critical parts of the process because a mis-step here could result in information loss or distortion. For example, filling missing values in YearBuilt with zeros makes no sense, whilst filling WoodenPatioArea with mean values might inadvertently mis-label properties that don't have such a feature.

Content on feature engineering in books and online seems light. Wondering if there's opportunities for improved tooling here too?

This [talk by Andrej Karpathy](https://vimeo.com/274274744) from Tesla, gives an interesting insight on the importance of feature engineering at Tesla.

See my [notebook here](https://www.kaggle.com/alanmartyn/linear-regression)

### 26. 2018-08-22

Today I started the Kaggle competition – [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

To get up and running I:

- read up on Kaggle competitions
- looked through the data descriptions in detail
- setup my Jupyter Notebook/Kernel
- reviewed [Statistical Data](https://en.wikipedia.org/wiki/Statistical_data_type) types 
- reviewed visualisation with Matplotlib
- reviewed data manipulation with Pandas
- Read chapter on [Feature Engineering](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html) from Python Data Science Handbook by Jake VanderPlas

Thoughts:

The Kaggle dataset is not as 'clean' as I imagined. There is plenty of inconsistency in the formatting and labelling that will need to tackled before it can be passed to a model. I'm going to segment real, count, ordinal and categorical data in case I need to process them separately.


### 25. 2018-08-21

Meet with study buddy to revise entire [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning) course. Focus on areas we found most difficult. We cover all items under 'Revisit' in this data log. It was really useful to talk through and test each others understanding, felt like I came away with much improved intution in key areas such as numpy broadcasting, inner products and Taylor Series Approximation.

Our search for suitable London housing datasets was fruitless. The Land Registry datasets include sale prices but not much else, we conclude that best approach would be to scrape archived classified listings. 

We decide to start by tackling the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) Kaggle competition.


### 24. 2018-08-17

Having completed the [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning) course, I met up with my study buddy to do plan next steps and revise the completed course.

We decide to do some applied work next week to predict house prices. Ideally we'd like to predict London house prices, so we decide to look for datasets.

### 23. 2018-08-16

**COMPLETE** [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Principal Component Analysis** week 4 including:

- Programming assignment: 
    - principle component analysis of MNIST dataset
    - benchmarking performance
    - optimisation of PCA for highly dimensional datasets



### 22. 2018-08-15

[Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Principal Component Analysis** week 4 including:

- Vector spaces
- Orthogonal compliments
- Principle Compenent Analysis (PCA) objective
- Multivariate chain rule
- PCA mathematical proof
- Lagrange multipliers revision

Revisit:

-  Chain rule practice q.2



### 21. 2018-08-14

[Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Principal Component Analysis** week 3 including:

- Projection onto 1D subspaces (general case using inner products)
- Projection onto higher dimensional subspaces
- Orthogonal projections
- Programming assignment: 
    - eigenfaces
    - auto-encoder for Olivetti faces dataset
    - linear regression for boston house prices


### 20. 2018-08-13

[Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Principal Component Analysis** week 2 including:

- Basis vectors revision
- Inner product angles and orthogonality
- exam: angles between vectors usin non-standard inner products
- programming assignment: inner products, angels and k-nearest-neighbour

Revisit:

- Programming Assignment: Inner products and angles def pairwise_distance_matrix(X, Y)
- What areas of this programming assignement required inner product generalisations?

### 19. 2018-08-12

- I took the first 8 lectures in Pavel Grinfeld's [inner product course](https://www.youtube.com/playlist?list=PLlXfTHzgMRULZfrNCrrJ7xDcTjGr633mm&disable_polymer=true)
- Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Principal Component Analysis** week 2 including:
    - properties of Inner Products
    - Inner product generalisations
    - 2 exams


Revisit:

- Inner product: distances between vectors: I can't replicate Deisenroth's result @3:00, 12^0.5

### 18. 2018-08-11

Studied course [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Principal Component Analysis** week 1 & 2 including:

- programming assignment to calculate mean and covariance of [The Olivetti Faces](http://scikit-learn.org/stable/datasets/olivetti_faces.html) dataset
    - implemented naive iterative mean and covariance functions in Python
    - benchmarked against equivalent functions in Numpy
- data visualisation: worked through Matplotlib [tutorials](https://matplotlib.org/tutorials/index.html) and a [helpful introduction to matplotlib](http://pbpython.com/effective-matplotlib.html) from Chris Moffit  

- linear algebra: Dot Product review
- linear algebra: Inner Products. Quite an abstract concept not well motivated by M4ML course, I found these lectures particularly useful.
    - [Why inner products?](https://www.youtube.com/watch?v=Ww_aQqWZhz8)
    - [Identifying inner products](https://www.youtube.com/watch?v=8M6eo3j7jO4)


### 17. 2018-08-10

Started course [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Principal Component Analysis** week 1 including:

- mean of high dimensional datasets
- variance of 1D datasets
- variance of high dimensional datasets
- covariance

I read the Principle Compenent Analysis section in Hastie & Tibshirani's Introduction to Statistical Learning textbook. 



### 16. 2018-08-08 

**Completed course:** [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Multivariate Calculus** 

Studied week 6 including:

- Linear regression
- General non-linear least squares
- Least squares regression analysis in practice
- Programming assignment: fitting the distribution of height data

I listened to [The History of Machine Learning from the Inside Out](https://robohub.org/talking-machines-history-of-machine-learning-w-geoffrey-hinton-yoshua-bengio-yann-lecun/) podcast where Geoffrey Hinton, Yoshua Bengio and Yann LeCun discuss motivations for deep-learning before state of the art results. Useful terminology, and interesting to hear their emphsasis on unsupervised learning as the area for future advances.

### 15. 2018-08-07 

Studied [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Multivariate Calculus** week 5 including:

- Gradient descent strategies
- Newton-Raphson revision
- Constrained optimisation
- Lagrange multipliers

Great fun experimenting with gradient descent methods in the exercises. I submitted a proposal to the course forums for a gradient descent method that seems more robust to steep sided gulleys and shallow gradients. Coursework is starting to give me a good orientation for Jupyter notebooks, scipy, and numpy.

I wanted more detail on constrained optimisation so I also took the Khan Academy course, [Lagrange multipliers and constrained optimization](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/lagrange-multipliers-and-constrained-optimization/v/constrained-optimization-introduction), which taught me about The Lagrangian and also steps through the proofs.

I finished the first chapter of Kevin Murphy's Machine Learning book.


### 14. 2018-08-06 

Studied [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Multivariate Calculus** week 4 + 5 including:

- Linearisation revision
- Multivariate Taylor series
- Taylor series exam
- Newton-Raphson iterative optimisation method

Revisit:

- week 4 / 2D Taylor series / q4 improve notes on linearisation, and ask forum question
- week 4 / Taylor Series Assessment / q4, why is f(x) = (x/2)^2​ sin(2x)/2 odd? Why are functions that are same when rotated 180deg about the origin odd?

### 13. 2018-08-05 

Studied [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Multivariate Calculus** week 4 including:

- Taylor Series
- Power Series
- Maclaurin Series
- Linearisation

exercises and exams, at this stage of the course, continue to extend practical abilities in finding derivatives of complex functions. I found this demonstration of the [Binomial Theorem](https://www.youtube.com/watch?v=h_-W4nqy-yY) useful because it provides a neat way to layout working on the page. 

Revisit:

- week 4 / Taylor series - Special cases / Q4 – why is f(x) = 1/(1+x)^2 discontinuous in the complex plane? 

### 12. 2018-08-04

Studied [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Multivariate Calculus** week 4 including:
- Building approximate functions
- Power series

Implemented backpropagation in Python with Numpy. That's the 3rd time I've implemented backprop now, having a detailed knowledge of the chain rule was really useful. This time around I feel like I've grokked it.


### 11. 2018-08-03

Studied [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Multivariate Calculus** week 3 including:

- Multivariate chain rule
- Simple neural networks
- Feedforward
- Backpropagation

Great to revisit the basic neural nets example with the linear algebra and calculus principles built up so far. Satisfying to combine linear algebra with the chain rule. 

I read the article [Google's AutoML: Cutting Through the Hype](http://www.fast.ai/2018/07/23/auto-ml-3/) which offers an interesting perspective on the commercialisation of ML tech, some potential pitfalls for big companies in this area.

### 10. 2018-08-02

Studied [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Multivariate Calculus** week 2 including:

- Partial differentiation
- The Jacobian
- The Hessian

Pretty cool to start combining linear algebra with calculus here. Looking at applications of the Jacobian and the Hessian to gradient descent problems – like the saddle problem – is a great motivator.

I read the paper [Scalable and accurate deep learning with electronic health records](https://www.nature.com/articles/s41746-018-0029-1) that reports on Google's latest work applying ML in healthcare. Looks like they’ve improved upon previous predictions by about 10% accuracy, but the thing that is most interesting is how they've achieved this without need for laborious cleaning and pre-processing of data. They have developed a pipeline that takes raw unstructured data like handwritten notes as input, and outputs some more structure FHIR compliant output. 

There's also a ux that allows doctors to inspect what parts of a given health record were most significant in generating the prediction, a step away from black-box characteristics. Finally, three different models are used (RNN, LSTM, Gradient Boosting), they are able to select the most appropriate model for a particular context using a method called Ensembling.



### 9. 2018-08-01

I met up with my study buddy to discuss progress. We decided to focus primarily on theoretical track for next 2 weeks to get foundational maths in place. Up until now we have found it difficult to split away from that track, and the content feels critical to more applied study.

I started the [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): **Multivariate Calculus** 6 week course.

Revised differentiation basics including Product Rule, Chain Rule, and practice exercises.

I got some advice from friends on best practices for self-directed study. I'm now separating my notes into two notebooks: 1) clean revision notes 2) messy exercise notes. Each morning I now reinforce my understanding by recalling concepts covered the previous day, I then review my notes as reinforcement. I'll also start revisiting the harder exercise questions as further reinforcement.


### 8. 2018-07-31

I completed the [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): Linear algebra course, completing final exam. 

I went back an reviewed some of the principles covered in more detail, by reviewing topics on Khan Academy including: [reduced row echelon form](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/matrices-elimination/v/matrices-reduced-row-echelon-form-1), rank of matrix, calculating determinants in the general case and so on.



**Thoughts:**

There's definitely more to learn about Linear Algebra than could be covered in the Imperial course, but it has given me a good sense of orientation and a desire to continue study in this area. My additional studies on Khan Academy have closed the loop on some areas I was intrigued by such as rref, and rank. 

### 7. 2018-07-30

[Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): Linear algebra course, covered: 
- study of eigenvectors 
- implementation of Google-esque PageRank algorithm which finds largest eigenvector with power method


### 6. 2018-07-29

- [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): Linear algebra. Covered:
    - Einstein summation convention and the symmetry of the dot product
    - Exam: Matrix multiplication
    - Matrices transform in a new basis vector set
    - Exam: Mappings to spaces with different numbers of dimensions
    - Orthogonal matrices
    - Gram-Schmidt process
    - Coursework: implemented Gram-Schmidt algo.
    - Reflecting in a plane
    - Coursework: Reflecting in a plane with Numpy
    - Week 4 complete
    - Eigenvalues and eigenvectors concepts
    - Exam: selecting eigenvectors by inspection
    - Special eigen-cases
    - Calculating eigenvectors
    - Exam: Characteristic polynomials, eigenvalues and eigenvectors

**Thoughts:**

A heads down day studying linear algebra. The exams provide a lot of exercise which really drive things home. Would be good to find ways to keep this knowledge fresh, perhaps longer term exercises?

**Thoughts:**

Great to work on some hands-on examples manipulating information with matrices. Techniques feel generally applicable. Insights on how to measure the information available in a set of vectors, e.g. are they linearly dependant? will be valuable in assessing/designing datasets.

### 5. 2018-07-28

**Today's progress:**

- Met up with a friend who as good applied data science experience. He has built a data science team and mentored students. I learned about the skill level of people working in the field, their day-to-day work and gained insight on best practices and pitfalls.
- [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): Linear algebra. Covered:
    - Solving linear equations using the inverse matrix I 
    - Determinants and inverses
    - Matrix identification programming assignment
    - finished week 3

**Thoughts:**

Data Science practitioners often bring with them a specific technique, perhaps based on past specialisation. It seems as though some practitioners might be inflexible in the range of techniques they will bring to bear on a problem. I wonder if it is realistic to gain a deep – or at least practical – understanding in a broad range of techniques? 

The ability to build a solid data processing pipeline could be an invaluable practical skill – Kafka might be worth investigating here. Maybe this is something I can collaborate with my study partner on.

Possible avenues for applied projects include:
- offering my services to a uni research dept.
- kaggle competitions
- self defined projects (we worked on some ideas here)



### 5. 2018-07-27

**Today's progress:**

- [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): Linear algebra. Covered:
    - transforms matrices
    - matrix composition and combination
    - inverse matrices
    - Gaussian elimination
    - finding inverse matrix with Gaussian elimination

**Thoughts:**

Continued with David Dye's course. Had to fix my front door lock so that was a distraction. The exercises in week 3 require quite laborious calculations with lots of basic arithmetic. It's important to be very careful not to cause cascading errors by dropping a minus sign or other basic arithmetic errors. I mad a few. Now I understand how to derive the inverse of a matrix and , and its given me a good intuition for what goes on when you call inv(A)!


### 4. 2018-07-26

**Today's progress:**

- Foundational: Studied [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning): Linear algebra. Covered:
    - changing basis for vector operations
    - basis, vector space, and linear independence
    - applications of changing basis
    - passed 3 exams on the above
    - (week 2 complete)
- Interview training, reviewed:
    - big O notation
    - data structures: strings, arrays, dynamic lists, linked lists
    - logarithms
    - (interviewcake chap. 0 complete)
- Started reading Murphy's textbook.

**Thoughts:**

It took me much longer to complete MfML week 2 than expected, time was spent working through the exam questions - most of time spent filling out 17 pages in my notebook with workings! I tried to make sure I could derive formula rather than rote memorisation. Cross checking with my study partner revealed a formula I'd derived incorrectly so that was definitely useful, was also good to practice talking the language. David Dye was light on practical method for testing linear dependence – Sal Khan's [More on Linear Independence](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/linear-independence/v/more-on-linear-independence) gave the detail I needed.

Interview training just reading so far. Looking forwards to trying out some exercises.

Murphy's is a very exciting book. Is accessible, and covers some prerequisites like probability. Might be a good companion text for MMfL PCA course. Murphy considers that broadly there are 3 domains of ML; supervised learning, unsupervised learning and reinforcement learning – the latter not being covered in this book. He recognises the similarities between statistics and ML, and suggests that although topics are broadly similar the emphasis is different.

Tomorrow I'd like to get at least week 3 of MMfL done and chap 1 of interview training.

Finally, my grandmother's 2-year old prediction that ethics will increasingly be a defining consideration in ML applications seems to be bolstered [today](https://www.bbc.co.uk/news/technology-44977366) as I read about biases hampering deployment of facial recognition systems. Throws up interesting questions like: How to measure bias in a dataset? How to measure bias in a model? 


### 3. 2018-07-25

**Today's progress:**

- implemented backpropagation algorithm with regularisation from scratch in Matlab to complete MLNG coursework (week 5 complete)
- began study of [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning). covered:
    - The relationship between machine learning, linear algebra, and vectors and matrices
    - Vectors: basic operations, modulus & inner product, cosine & dot product, projection
    - almost finished week2

**Thoughts:** 

The mathematics for machine learning course has been useful, so far I've learned a range of vector operations. The course works up from first principles which means you can get a long way by just recalling the fundamentals and working up from their. Much better suited for me than remembering a load of formulae without context. I also think it is a very good compliment for Andrew Ng's more applied course, certainly goes deeper on the math. I've dug out one of my undergrad textbooks which covers same material, K.A. Stroud's Engineering Mathematics – useful reference for exercise but I'm finding David Dyes lecture's move me along much faster. 

Glad to finish the backprop coursework. It was amazing to see hidden layers visualised for MNIST, I've seen those kinds of visualisations before but now I feel like I have a strong intuition on how to interpret them. 

Kevin Murphy's seminal book just arrived. Linear regression doesn't appear until page 219, and deep learning not until 999, so I'd better get to it.


### 2. 2018-07-24

**Today's progress:** 

- revisited Andrew Ng's ML week 5 lectures on neural nets, feedworward, and backpropagation.
- Implemented feedfoward for MNIST example in Matlab
- reviewed and responded to great feedback on curriculum and motivations from professionals in the field 
- listened to Talking Machines podcast episode 1
- had cursory skim through Google's new [Machine Learning guides](https://developers.google.com/machine-learning/guides/), these look useful, might fold into curriculum
- started building a list of project ideas, i need to publish these
- added [my coursework](https://github.com/a-martyn/ml-sabbatical/tree/master/coursework-ng) from Andrew Ng's Machine Learning course to this repo

**Thoughts:** 

Very useful feedback from mentors. Key points included:

- find a problem that I'm compelled to solve, then use that as project and curriculum benchmark
- find ways to teach what I've learned to others at a similar level
- mentors encouraged me to clarify my thinking on where I want to be in 3 months

This will all be really useful for informing curriculum.

Great to get back up and running with Ng's course. Ng recommends implementing feedforward with a `for` loop, but as I was implementing it I noticed that the loop could be removed by applying linear algebra. I decided to stick with Ng's recommendation, I think he recommends this approach because it allows you to break down the process into very small incremental steps. This proved useful during debugging, and I think it gave me a better intuition than if I'd used abstractions. 

When implementing feedforward I was getting an unexpected value. I was able to debug it by isolating one training example as a test case, and then using intuition to notice that the cost should not be a negative value. I didn't get as far on the coursework as I'd hoped, but feel like I'll be faster today.


### 1. 2018-07-23

**Today's progress:** 

- [x] define [curriculum](https://github.com/a-martyn/ml-sabbatical/tree/master/docs/005-curriculum.md) 
- [x] setup this learning log
- [x] share [curriculum](https://github.com/a-martyn/ml-sabbatical/tree/master/docs/005-curriculum.md) and [motivations]((https://github.com/a-martyn/ml-sabbatical/tree/master/docs/002-curriculum.md)) with friends for feedback
- [x] ordered some textbooks

**Thoughts:** 

It's day 1. Well not quite, I've spent the last few days researching the available learning resources including courses and textbooks. There's a lot out there, I started to feel a bit lost at sea, so I decided to use the UCL MSC in machine learning as a basis, and focused on material that covered prerequisite foundational material as well the core unsupervised learning module, here's my [notes](https://github.com/a-martyn/ml-sabbatical/tree/master/docs/004-books-and-courses.md).

Today I met up with my study partner to compare notes and pin down a [curriculum](https://github.com/a-martyn/ml-sabbatical/tree/master/docs/005-curriculum.md). It is a big relief to have this down on paper. Looking forwards to getting into it tomorrow.

</div>
