# Medical_Insurance_Cost_Prediction

This document is written to provide aid in understanding the project.

Contents of the document - 
1. Understanding the problem statement 
2. About the dataset
3. Machine learning 
4. Types of machine learning models with examples 
5. Machine learning algorithm used for the model - linear regression 
6. NumPy library 
7. Pandas library 
8. Scikit-learn library 
9. Exploratory data analysis 
10. Fixing missing values in the dataset 
11. Data visualisation - Seaborn 
12. Categorical encoding 
13. Train-test split 
14. Model evaluation - R squared error  

What is the problem statement for the machine learning algorithm ?

Insurance is a policy that helps to cover up all loss or decrease loss in terms of expenses incurred by various risks. A number of variables affect how much insurance costs. Using machine learning algorithms will help provide a computational intelligence approach for predicting insurance cost.  
Our job is to automate this process of predicting the healthcare insurance cost calculated from the features provided in the dataset. Machine learning provides a computational intelligence approach for predicting healthcare insurance costs

Understanding the dataset - 

Dataset contains the following columns/features - 
1. Age - age of the patient 
2. Sex - female or male 
3. BMI - body mass proportion. BMI is the weight of person in kg divided by square of height in cm 
4. Children - number of children the patient has 
5. Smoker - whether the person is smoker or not. Yes or no 
6. Region - region of residence of patient 
7. Charges - amount of money charged 

The dataset contains 1338 rows (1339 data points) and 7 columns (7 features).

Machine learning - 

Machine learning enables the processing of sonar signals and target detection. Machine Learning is a subset of Artificial Intelligence. This involves the development of computer systems that are able to learn by using algorithms and statistical measures to study data and draw results from it. Machine learning is basically an integration of computer systems, statistical mathematics and data.

Machine Learning is further divided into three classes - Supervised learning, Unsupervised learning and Reinforcement Learning. 

Supervised learning is a machine learning method in which models are trained using labelled data. In supervised learning, models need to find the mapping function and find a relationship between the input and output. In this, the user has a somewhat idea of what the output should look like. It is of two types - regression (predicts results with continuous output. For example, given the picture of a person, we have to predict their age on the basis of the given picture) and classification (predict results in a discrete output. For example, given a patient with a tumor, we have to predict whether the tumor is malignant or benign.) 

Unsupervised learning is a method in which patterns are inferred from the unlabelled input data. It allows us to approach problems with little or no idea what the results should look like. We can derive structure from the data where we don???t necessarily know the effect of variables. We can derive the structure by clustering the data based on relationships among the variables in the data. With unsupervised learning there is no feedback on the prediction results. It is of two types - clustering (model groups input data into groups that are somehow similar or related by different variables. For example, clustering data of thousands of genes into groups) and non-clustering (models identifies individual inputs. It helps us find structure in a chaotic environment. For example, the cocktail party problem where we need to identify different speakers from a given audiotape.)

Reinforcement learning is a feedback-based machine learning technique. It is about taking suitable action to maximise reward in a particular situation. For example, a robotic dog learning the movement of his arms or teaching self-driving cars how to depict the best route for travelling. 

In this case, I use Linear regression model.

Linear regression algorithm shows the linear relationship between a dependent and one or more independent variables. Linear regression model provides the sloped straight line representing the relationship between the variables. 

In linear regression, our aim is to find the best fit line that means the error between the predicted values and actual values should be minimized. The best fit line will have th least error. The different values for weights or the coefficient of lines gives a different line of regression. A cost function is used to calculate the best values of coefficients. 

A cost function optimizes the regression coefficients or weights. It measures how a linear regression model is performing. It finds the accuracy of mapping function, which maps the input variable to the output variable. This mapping function is hypothesis function. 

Python libraries used in the project - 

NumPy  

It is a python library used for working with arrays. It has functions for working in the domain of linear algebra, fourier transform, and matrices. It is the fundamental package for scientific computing with python. NumPy stands for numerical python. 

NumPy is preferred because it is faster than traditional python lists. It has supporting functions that make working with ndarray very easy. Arrays are frequently used where speed and resources are very important. NumPy arrays are faster because it is stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently. This is locality of reference in computer science. 

Pandas - 

Pandas is made for working with relational or labelled data both easily and intuitively. It provides various data structures and operations for manipulating numerical data and time series. 

It has a lot of advantages like - 
1. Fast and efficient for manipulating and analyzing data
2. Data from different file objects can be loaded 
3. Easy handling of missing data in data preprocessing 
4. Size mutability 
5. Easy dataset merging and joining 
6. Flexible reshaping and pivoting of datasets 
7. Gives time-series functionality 

Pandas is built on top of NumPy library. That means that a lot of structures of NumPy are used or replicated in Pandas. The data produced by pandas are often used as input for plotting functions of Matplotlib, statistical analysis in SciPy, and machine learning algorithms in Scikit-learn. 

Scikit-Learn - 

It provides efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction. It has numerous machine learning, pre-processing, cross validation and visualisation algorithms. 

Maplotlib -

Matplotlib is a visualization library in python for 2D plots of arrays. It allows visual access to huge amounts of data in easily digestible visuals and plots like line, bar, scatter, histogram etc. 

Exploratory data analysis - 

Exploratory data analysis is the process of performing initial investigation on the data to discover patterns or spot anomalies. It is done to test the hypothesis and to check assumptions with the help of summary statistics and graphical representations. 

???describe()??? method returns description of data in DataFrame. It tells us the following information for each column - 
- Count - number of non-empty values
- Mean - the average (mean) value  
- Std - standard deviation
- Min - minimum value
- 25% - the 25 percentile 
- 50% - the 50 percentile 
- 75% - the 75 percentile
- Max - maximum value

The info() method prints the information about dataframe. 
It contains the number of columns, column labels, column data types, memory usage, range index, and number of cells in each column. 

Parameters - 
1. verbose - It is used to print the full summary of the dataset.
2. buf - It is a writable buffer, default to sys.stdout.
3. max_cols - It specifies whether a half summary or full summary is to be printed.
4. memory_usage - It specifies whether total memory usage of the DatFrame elements (including index) should be displayed.
5. null_counts - It is used to show the non-null counts.

Missing values - 

Missing values are common when working with real-world datasets. Missing data could result from a human factor, a problem in electrical sensors, missing files, improper management or other factors. Missing values can result in loss of significant information. Missing value can bias the results of model and reduce the accuracy of the model. There are various methods of handling missing data but unfortunately they still introduce some bias such as favoring one class over the other but these methods are useful. 

In Pandas, missing values are represented by NaN. It stands for Not a Number. 

Reasons for missing values - 
1. Past data may be corrupted due to improper maintenance
2. Observations are not recorded for certain fields due to faulty measuring equipments. There might by a failure in recording the values due to human error. 
3. The user has not provided the values intentionally. 

Why we need to handle missing values - 
1. Many machine learning algorithms fail if the dataset contains missing values. 
2. Missing values may result in a biased machine learning model which will lead to incorrect results if the missing values are not handled properly. 
3. Missing data can lead to lack of precision. 

Types of missing data - 

Understanding the different types of missing data will provide insights into how to approach the missing values in the dataset. 
1. Missing Completely at Random (MCAR) 
There is no relationship between the missing data and any other values observed or unobserved within the given dataset. Missing values are completely independent of other data. There is no pattern. The probability of data being missing is the same for all the observations. 
The data may be missing due to human error, some system or equipment failure, loss of sample, or some unsatisfactory technicalities while recording the values.
It should not be assumed as it???s a rare case. The advantage of data with such missing values is that the statistical analysis remains unbiased.   
2. Missing at Random (MAR)
The reason for missing values can be explained by variables on which complete information is provided. There is relationship between the missing data and other values/data. In this case, most of the time, data is not missing for all the observations. It is missing only within sub-samples of the data and there is pattern in missing values. 
In this, the statistical analysis might result in bias. 
3. Not MIssing at Random (NMAR)
Missing values depend on unobserved data. If there is some pattern in missing data and other observed data can not explain it. If the missing data does not fall under the MCAR or MAR then it can be categorized as MNAR. 
It can happen due to the reluctance of people in providing the required information. 
In this case too, statistical analysis might result in bias. 

How to handle missing values - 

isnull().sum() - shows the total number of missing values in each columns 

We need to analyze each column very carefully to understand the reason behind missing values. There are two ways of handling values - 
1. Deleting missing values - this is a simple method. If the missing value belongs to MAR and MCAR then it can be deleted. But if the missing value belongs to MNAR then it should not be deleted. 
The disadvantage of this method is that we might end up deleting useful data. 
You can drop an entire column or an entire row. 
2. Imputing missing values - there are various methods of imputing missing values
3. Replacing with arbitrary value 
4. Replacing with mean - most common method. But in case of outliers, mean will not be appropriate
5. Replacing with mode - mode is most frequently occuring value. It is used in case of categorical features. 
6. Replacing with median - median is middlemost value. It is better to use median in case of outliers. 
7. Replacing with previous value - it is also called a forward fill. Mostly used in time series data. 
8. Replacing with next value - also called backward fill. 
9. Interpolation 

Data visualisation - 

Datasets often come in csv files, spreadsheets, table form etc. Data visualisation provides a good and organized pictorial representation of data which makes it easier to observe, understand and analyze. 

Python provides various libraries that come with different features for visualizing data. All these libraries have different features and can support various types of graphs. 
1. Matplotlib - for 2D array plots. It includes wide range of plots, such as scatter, line, bar, histogram and others that can assist in delving deeper into trends. 
2. Seaborn - it is used for creating statistical representations based on datasets. It is built on top of matplotlib. It is built on top of pandas??? data structures. The library conducts the necessary modelling and aggregation internally to create insightful visuals.
3. Bokeh - it is a modern web browser based interactive visualization library. It can create engaging plots and dashboards with huge streaming data. The library contains many intuitive graphs. It has close relationship with PyData tools. The library is ideal for creating customized visuals.  
4. Plotly - python visualization library that is interactive, accessible, high-level and browser-based. Scientific graphs, 3D charts, statistical plots, and financial charts. Interaction and editing options are available. 

Data visualisation plot used in the project - 

Seaborn - 

Seaborn aims to make visualization the central part of exploring and understanding data. It provides dataset-oriented APIs, so that we can switch between different visual representations for same variables for better understanding of dataset.

Distplot - 

It is used for univariant set of observations and visualizes it through a histogram - only one observation and hence we choose one particular column of the dataset. 

Countplot - 

seaborn.countplot() method is used to show the counts of observations in each categorical bin using bars. 

Parameters - 
1. x, y - This parameter take names of variables in data or vector data, optional, Inputs for plotting long-form data.
2. Hue - (optional) This parameter take column name for colour encoding.
3. Data - (optional) This parameter take DataFrame, array, or list of arrays, Dataset for plotting. If x and y are absent, this is interpreted as wide-form. Otherwise it is expected to be long-form.
4. order, hue_order - (optional) This parameter take lists of strings. Order to plot the categorical levels in, otherwise the levels are inferred from the data objects.
5. Orient - (optional)This parameter take ???v??? | ???h???, Orientation of the plot (vertical or horizontal). This is usually inferred from the dtype of the input variables but can be used to specify when the ???categorical??? variable is a numeric or when plotting wide-form data.
6. Color - (optional) This parameter take matplotlib color, Color for all of the elements, or seed for a gradient palette.
7. Palette - (optional) This parameter take palette name, list, or dict, Colors to use for the different levels of the hue variable. Should be something that can be interpreted by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
8. Saturation - (optional) This parameter take float value, Proportion of the original saturation to draw colors at. Large patches often look better with slightly desaturated colors, but set this to 1 if you want the plot colors to perfectly match the input color spec.
9. Dodge - (optional) This parameter take bool value, When hue nesting is used, whether elements should be shifted along the categorical axis.
10. Ax - (optional) This parameter take matplotlib Axes, Axes object to draw the plot onto, otherwise uses the current Axes.
11. Kwargs - This parameter take key, value mappings, Other keyword arguments are passed through to matplotlib.axes.Axes.bar().
12. Returns - Returns the Axes object with the plot drawn onto it.

Categorical data encoding - 

Datasets contain multiple labels (in the form of words of numbers) in one or more than one columns. Training data is often labelled in words to make the data more human-readable and understandable. Categorical data encoding is coding data to convert it to a machine-readable form. It is an important data preprocessing method. 

There are two types of categorical data 
- Ordinal data - categories have an inherent order
- Nominal data - no inherent order of categories

Label encoding - 

Label encoding refers to converting labels into numeric form to convert it to machine-readable form. It is an important data preprocessing method for structured dataset in supervised learning. 

Limitation - label encoding assigns a unique number to each class of data. However, this can lead to generation of priority issues in the training of data. A label with high value may be considered to have high priority than a label having a lower value. 

Index.value_counts() -

This function returns objects containing counts of unique values. 
Excludes NA values by default. The first element is most frequently occurring element.  

Parameters - 
1. Normalize - f True then the object returned will contain the relative frequencies of the unique values. 
2. Sort - Sort by values 
3. Ascending - Sort in ascending order 
4. Bins 
5. Dropna - Don???t include counts of NaN.
6. Returns - counts : Series 

replace() - 

replace() in Python returns a copy of the string where all occurrences of a substring are replaced with another substring.

Parameters - 
1. Old ??? old substring you want to replace.
2. New ??? new substring which would replace the old substring.
3. Count ??? (Optional ) the number of times you want to replace the old substring with the new substring. 

Return value - It returns a copy of the string where all occurrences of a substring are replaced with another substring.

Train-test split - 

The entire dataset is split into training dataset and testing dataset. Usually, 80-20 or 70-30 split is done. The train-test split is used to prevent the model from overfitting and to estimate the performance of prediction-based algorithms. We need to split the dataset to evaluate how well our machine learning model performs. The train set is used to fit the model, and statistics of training set are known. Test set is for predictions. 

This is done by using scikit-learn library and train_test_split() function. 
Parameters - 
1. *arrays: inputs such as lists, arrays, data frames, or matrices
2. test_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our test size. its default value is none.
3. train_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our train size. its default value is none.
4. random_state: this parameter is used to control the shuffling applied to the data before applying the split. it acts as a seed.
5. shuffle: This parameter is used to shuffle the data before splitting. Its default value is true.
6. stratify: This parameter is used to split the data in a stratified fashion.

Model evaluation - 

Model evaluation is done to test the performance of machine learning model. It is done to determine whether the model is a good fit for the input dataset or not. 

In this case, I use R squared error. 

R-square is comparison of the residual sum of squares with the total sum of squares. The total sum of squares is calculated by sum of squares of perpendicular distance between the data points and the average line. The residual sum of squares is sum of squares of perpendicular distance between the data points and the best fitted line. 
The ideal value for r square is 1. The closer the result is to 1, the better is the model fitted. 
