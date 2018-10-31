# Machine Learning: Curriculum

This is the curriculum I've followed with the goal of understanding machine learning from first principles. I hope that this is useful to others considering exploring machine learning from a background in software development. If you'd like to discuss please dm me on twitter or email.

## Starting point

- Undergraduate degree in Audio Engineering covering audio signal processing, compression techniques, auditory perception, mathematics and electrical engineering
- 6 years experience working in software development and prototyping for iOS and web
- 2-3 years experience in python development for backend web services and some data analysis
- lots of messing around with Arduino microprocessors
- A-level and undergraduate mathematics in need of refresh

## 1. [Machine Learning](https://www.coursera.org/learn/machine-learning) by Andrew Ng (Stanford / Coursera)

A broad high-level introduction to machine learning techniques. A good motivational course but I stopped after week 5 because I felt it was too high-level, instead I wanted to approach the subject bottom-up from mathematical principles.

- Linear Regression with One Variable
- Linear Algebra Review
- Linear Regression with Multiple Variables
- Octave/Matlab Tutorial
- Logistic Regression
- Regularisation
- Neural Networks

## 2. [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning) by David Dye, Samuel J. Cooper, Marc P. Deisenroth (Imperial College / Coursera)

A good introduction to some of the prerequisite mathematics for machine learning covering Linear Algebra, Multivariate Calculus and statistical methods. I feel that probability theory is missing.

#### Course 1: Linear Algebra
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


#### Course 2: Multivariate Calculus 
*Lecturer: Samule J. Cooper*

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


#### Course 3: Principle Components Analysis
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
- [Lagrange multipliers and constrained optimization](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/lagrange-multipliers-and-constrained-optimization/v/constrained-optimization-introduction) – Khan Academy
- [Reduced row echelon form](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/matrices-elimination/v/matrices-reduced-row-echelon-form-1) – Khan Academy
- [More on Linear Independence](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/linear-independence/v/more-on-linear-independence) - Khan Academy
- Matplotlib [tutorials](https://matplotlib.org/tutorials/index.html) and a [helpful introduction to matplotlib](http://pbpython.com/effective-matplotlib.html) from Chris Moffit  


## 3. [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Kaggle)

Although all courses included in the curriculum so far include practical exercises they are quite heavily guided so I a more free-form applied setting is desired. For this I turned to Kaggle. A welcome opportunity to work with a real dataset which opened my eyes to the importance of data preprocessing, exploration and domain expertise.

- Jupyter notebooks
- Pandas, Numpy, Seaborn, Matplotlib, Sklearn
- Feature engineering
- Cross-validation
- Data exploration

See my [notebook here](https://www.kaggle.com/alanmartyn/linear-regression). Disclaimer – a lot of naive work here which lead me to pick up *An Introduction to Statistical Learning* next.

##### Supporting study:

- [Python Data ScienceHandbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html) by Jake VanderPlas

## 4. [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/) by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani (Textbook)

