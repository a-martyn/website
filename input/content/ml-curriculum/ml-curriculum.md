## Machine Learning: Curriculum

This is the curriculum I've followed with the goal of understanding machine learning from first principles whilst also building practical experience applicable to real-world problems. I'd be thrilled if this record of my path is useful to anyone else who is approaching machine learning from a background in software development or otherwise. If you'd like to discuss please dm me on twitter or email.

### Starting point

- Undergraduate degree in Audio Engineering covering audio signal processing, compression techniques, auditory perception, mathematics and electrical engineering
- 6 years experience working in software development and prototyping for iOS and web
- 2-3 years experience in python development for backend web services and some data analysis
- lots of messing around with Arduino microprocessors
- A-level and undergraduate mathematics in need of refresh

### 1. [Machine Learning](https://www.coursera.org/learn/machine-learning) by Andrew Ng
*(Stanford / Coursera)*


A broad high-level introduction to machine learning techniques. A good motivational course but I stopped after week 5 because I felt it was too high-level, instead I wanted to approach the subject bottom-up from mathematical principles.

My code solutions are on [Github here](https://github.com/a-martyn/ml-sabbatical/tree/master/coursework-ng).

- Linear Regression with One Variable
- Linear Algebra Review
- Linear Regression with Multiple Variables
- Octave/Matlab Tutorial
- Logistic Regression
- Regularisation
- Neural Networks



### 2. [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning)

*(Imperial College / Coursera)*

A good introduction to some of the prerequisite mathematics for machine learning covering Linear Algebra, Multivariate Calculus and statistical methods. I feel that probability theory is missing.

See my code on [Github here](https://github.com/a-martyn/ml-sabbatical/tree/master/coursework-maths4ml).

##### Course 1: Linear Algebra
*Lecturer: David Dye*

- The relationship between machine learning, linear algebra, and vectors and matrices
- Vectors
- Modulus & inner product
- Dot product
- Projection
- Basis, vector space, and linear independence
- Changing basis
- Matrices in linear algebra: operating on vectors
- Matrix Inverses, Gaussian elimination
- Matrix Determinants and Inverses
- Einstein summation
- Non-square matrix multiplication
- Matrix transforms, mapping and changing basis
- Orthogonal matrices
- The Gram-Schmidt process
- Eigenvalues and eigenvectors
- Eigenvector application: implementation of Page Rank algorithm


##### Course 2: Multivariate Calculus 
*Lecturer: Samuel J. Cooper*

- Gradients and derivatives
- Multivariate Product Rule & Chain Rule
- Multivariate calculus
- Partial differentiation
- The Jacobian
- The Hessian
- Multivariate chain rule
- Implementation of back propagation and neural net
- Multivariate Taylor Series approximations
- Optimisation
- Newton-Raphson method
- Implementation and comparison of gradient descent methods
- Constrained optimisation & Lagrange multipliers
- Linear regression
- General non-linear least squares


##### Course 3: Principle Components Analysis
*Lecturer: Marc P. Deisenroth*

- Mean of datasets
- Variances and covariances
- Linear transformation of datasets
- Dot product
- Inner products
- Projections
- Principle Component Analysis derivation
- Principle Component Analysis implementation


##### Supporting study:

- Pavel Grinfeld's [inner product course](https://www.youtube.com/playlist?list=PLlXfTHzgMRULZfrNCrrJ7xDcTjGr633mm&disable_polymer=true)
- [Lagrange multipliers and constrained optimisation](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/lagrange-multipliers-and-constrained-optimization/v/constrained-optimization-introduction) – Khan Academy
- [Reduced row echelon form](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/matrices-elimination/v/matrices-reduced-row-echelon-form-1) – Khan Academy
- [More on Linear Independence](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/linear-independence/v/more-on-linear-independence) - Khan Academy
- Matplotlib [tutorials](https://matplotlib.org/tutorials/index.html) and a [helpful introduction to matplotlib](http://pbpython.com/effective-matplotlib.html) from Chris Moffit  


### 3. [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques): Kaggle

Although all courses included in the curriculum so far include practical exercises they are quite heavily guided so I wanted a more free-form applied setting. For this I turned to Kaggle.

See my [notebook here](https://www.kaggle.com/alanmartyn/linear-regression). Disclaimer – a lot of naive work here which lead me to pick up *An Introduction to Statistical Learning* next.

##### Supporting study:

- [Python Data ScienceHandbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html) by Jake VanderPlas

### 4. [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)

*Trevor Hastie and Robert Tibshirani (Textbook)*

<IMG src='http://www-bcf.usc.edu/%7Egareth/ISL/ISL%20Cover%202.jpg' height=20% width=20%> <P>

To build a conceptual and applied understanding of supervised learning techniques I chose to work through An Introduction to Statistical Learning cover-to-cover, completing the exercise at the end of each chapter. Conceptual exercises cover the mathematics, whilst applied exercises focus on prediction and inference tasks using various datasets. 

The textbook targets the R programming language, but I decided to implement my solutions in Python. Where Python lacked R's functionality I implemented from scratch – I found this to be a useful means to build intution.


Links to my notebooks below: 

[Chapter 2 - Statistical Learning: Conceptual](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch2_statistical_learning_conceptual.ipynb)  
[Chapter 2 - Statistical Learning: Applied](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch2_statistical_learning_applied.ipynb)


[Chapter 3 - Linear Regression: Conceptual](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch3_linear_regression_conceptual.ipynb)  
[Chapter 3 - Linear Regression: Applied](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch3_linear_regression_applied.ipynb)


[Chapter 4 - Classification: Conceptual](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch4_classification_conceptual.ipynb)  
[Chapter 4 - Classification: Applied](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch4_classification_applied.ipynb)


[Chapter 5 - Resampling Methods: Conceptual](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch5_resampling_methods_conceptual.ipynb)  
[Chapter 5 - Resampling Methods: Applied](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch5_resampling_methods_applied.ipynb)


[Chapter 6 - Linear Model Selection and Regularization: Labs](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_labs.ipynb)  
[Chapter 6 - Linear Model Selection and Regularization: Conceptual](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_conceptual.ipynb)  
[Chapter 6 - Linear Model Selection and Regularization: Applied](http://nbviewer.jupyter.org/github.com/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_applied.ipynb)


[Chapter 7 - Moving Beyond Linearity: Labs](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch7_moving_beyond_linearity_labs.ipynb)  
[Chapter 7 - Moving Beyond Linearity: Applied](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch7_moving_beyond_linearity_applied.ipynb)


[Chapter 8 - Tree-Based Methods: Labs](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch8_tree_based_methods_labs.ipynb)  
[Chapter 8 - Tree-Based Methods: Conceptual](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch8_tree_based_methods_conceptual.ipynb)  
[Chapter 8 - Tree-Based Methods: Applied](http://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch8_tree_based_methods_applied.ipynb)


[Chapter 9 - Support Vetor Machines: Labs](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch9_support_vector_machines_labs.ipynb)  
[Chapter 9 - Support Vetor Machines: Conceptual](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch9_support_vector_machines_conceptual.ipynb)  
[Chapter 9 - Support Vetor Machines: Applied](https://nbviewer.jupyter.org/github/a-martyn/ISL-python/blob/master/Notebooks/ch9_support_vector_machines_applied.ipynb)


The source code for these notebooks is available on [Github here](https://github.com/a-martyn/ISL-python).


### [Ideas for future study](#ideas-for-future-study)

Below is an evolving and rough plan for areas of study and projects that I'd like to do dive into next.

##### Probability Theory / Bayesian Statistics

I think we need some introduction to bayesian statistics. Options I've found so far are:

- Curriculum: [Count Bayes](https://www.countbayesie.com/blog/2016/5/1/a-guide-to-bayesian-statistics)
- Lectures: [Aubrey Clayton's Logic of Science lecture series](https://www.youtube.com/user/elfpower/videos)
- Book: [*Probability Theory: The Logic of Science* by E.T. Jaynes](https://books.google.co.uk/books/about/Probability_Theory.html?id=tTN4HuUNXjgC&source=kp_book_description&redir_esc=y), Bayesian Statistics, them seminal work.
- Book: [*Doing Bayesian Data Analysis* by John K. Kruschke](https://www.amazon.com/Doing-Bayesian-Data-Analysis-Second/dp/0124058884/ref=as_li_ss_tl?ie=UTF8&qid=1462141686&sr=8-1&keywords=doing+bayesian+data+analysis&linkCode=sl1&tag=counbaye09-20&linkId=d4059e53b7b13b9daa785421e5bf99a5), practical Bayesian Data Analysis


##### Game Theory

- Course: [Game Theory (Coursera)](https://www.coursera.org/learn/game-theory-1)
- Project: submit at contribution to the [Axlerod project](https://github.com/Axelrod-Python/Axelrod)

##### Reinforcement learning

- Lectures: [Introduction to Reinforcement Learning by David Silver](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- Book: [Reinforcement Learning: An Introduction (2nd Edition)](https://www.amazon.co.uk/Reinforcement-Learning-Introduction-Richard-Sutton/dp/0262039249)
- Project: Try and solve some arcade game, or some other simulated problem


##### Deep learning

- Course: [Andrew Ng's Deep Learning Specialisation](https://www.coursera.org/specializations/deep-learning)
- Book: [*Deep Learning* by Ian Goodfellow and Yoshua Bengio and Aaron Courville](https://www.deeplearningbook.org/)

##### Other references

- This is the best [ML bibliography](https://humancompatible.ai/bibliography) I've found from Centre for Human-Compatible Artificial Intelligence. 
