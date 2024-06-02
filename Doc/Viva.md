# Viva Possible Questions

## General Dump 01

1. **Curse of Dimensionality:**
   - _What is the curse of dimensionality, and how does it affect machine learning algorithms?_
     - The curse of dimensionality refers to the challenges that arise when working with high-dimensional data. As the number of dimensions increases, the data becomes increasingly sparse, making it harder for algorithms to generalize effectively due to the lack of representative samples.
2. **Bias-Variance Tradeoff:**

   - _What is the bias-variance tradeoff, and why is it important in machine learning?_
     - The bias-variance tradeoff refers to the balance between the bias of a model (error due to overly simplistic assumptions) and its variance (sensitivity to fluctuations in the training data). It's crucial because reducing bias often increases variance and vice versa, affecting the model's ability to generalize.

3. **Reducing Overfitting:**

   - _What is overfitting, and why is it a concern in machine learning?_
     - Overfitting occurs when a model learns the training data too well, capturing noise or random fluctuations rather than the underlying patterns. It's concerning because it leads to poor performance on unseen data, compromising the model's ability to generalize.

4. **VIF-Based Reduction of Multicollinearity:**

   - _What is multicollinearity, and why is it problematic in regression analysis?_
     - Multicollinearity refers to the presence of high correlations among predictor variables in a regression model, which can lead to unstable coefficient estimates and inflated standard errors. It complicates the interpretation of individual predictors' effects on the target variable.

5. **VIF Cutoff:**

   - _What is the VIF cutoff, and how is it used in practice?_
     - The VIF cutoff is a threshold value used to identify multicollinearity in regression models. Generally, a VIF value exceeding 10 is considered problematic, indicating high multicollinearity among predictors. Researchers and practitioners use this cutoff to decide whether to address multicollinearity in their analysis.

<div style="page-break-after:always;"></div>

6. **Question: Difference between classifier and regressor?**

   - Classifier:

     - Predicts categorical labels.
     - Used for classification tasks.
     - Output: discrete class label or probability distribution.

   - Regressor:
     - Predicts continuous numerical values.
     - Used for regression tasks.
     - Output: continuous value or range.

7. **What is the purpose of StandardScaler, MinMaxScaler, and PCA in machine learning preprocessing?**

- **StandardScaler:**
  - StandardScaler is used to standardize features by removing the mean and scaling to unit variance.
  - It transforms data such that it has a mean of 0 and a standard deviation of 1.
  - This scaling technique is useful when the features in the dataset have different scales or units.
- **MinMaxScaler:**

  - MinMaxScaler scales features to a given range, usually [0, 1].
  - It preserves the shape of the original distribution and is sensitive to outliers.
  - MinMaxScaler is suitable for algorithms that require features to be on a similar scale or bounded within a specific range.

- **PCA (Principal Component Analysis):**
  - PCA is a dimensionality reduction technique used to reduce the number of features in a dataset while preserving most of its variance.
  - It identifies the directions (principal components) that maximize the variance in the data and projects the original features onto these components.
  - PCA is helpful for reducing computational complexity, removing noise, and visualizing high-dimensional data.
  - PCA performs eigenvalue decomposition or singular value decomposition on the covariance matrix of the dataset to obtain the principal components.

8. **How does StandardScaler differ from MinMaxScaler?**

- **StandardScaler:**
  - Standardizes features to have a mean of 0 and a standard deviation of 1.
  - Centers the data around 0 and scales it to have unit variance.
  - Preserves the shape of the distribution but is sensitive to outliers.

<div style="page-break-after:always;"></div>

- **MinMaxScaler:**
  - Scales features to a specified range, typically [0, 1].
  - Does not center the data; it scales it between the specified minimum and maximum values.
  - Preserves the shape of the distribution and is less sensitive to outliers compared to StandardScaler.

9. **What are the advantages of using PCA?**

- **Dimensionality Reduction:**
  - PCA reduces the number of features in the dataset, making it computationally efficient and easier to visualize.
- **Noise Reduction:**
  - PCA identifies and removes noise and redundant information in the data, leading to simpler and more interpretable models.
- **Feature Compression:**
  - PCA compresses the data by representing it in terms of a smaller number of principal components, while still retaining most of its variability.
- **Visualization:**
  - PCA allows for the visualization of high-dimensional data in lower dimensions, facilitating the exploration and understanding of complex datasets.

10. **How does correlation help in choosing a target variable in machine learning?**

- Correlation analysis reveals the strength of the linear relationship between potential target variables and predictors, aiding in selecting variables with strong predictive power for the target.
- Variables with high correlations with predictors are likely to be more informative and contribute significantly to predicting the target variable, leading to better model performance.

11. **What is the difference between a correlation matrix and a covariance matrix, and their significance in machine learning?**

- **Correlation Matrix:**

  - Measures the linear relationship between variables.
  - Helps identify redundant features and multicollinearity.

- **Covariance Matrix:**
  - Measures the joint variability between variables.
  - Used in dimensionality reduction and regularization techniques.

<div style="page-break-after:always;"></div>

## Based On Models

| Method / Model       | Use Cases                                                      | Pros                                            | Cons                                              | Suitable for               |
| -------------------- | -------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------- | -------------------------- |
| Linear Regression    | Predicting continuous values                                   | Simple, interpretable                           | Assumes linear relationship                       | Regression                 |
| Logistic Regression  | Binary classification                                          | Probabilistic interpretation, easy to implement | Assumes linear decision boundary                  | Classification             |
| Gaussian Naive Bayes | Text classification, simple data with independence assumptions | Efficient, handles high-dimensional data        | Assumes feature independence                      | Classification             |
| Random Forest        | Classification, regression, feature importance                 | High accuracy, handles complex interactions     | May overfit if hyperparameters not tuned properly | Classification, Regression |
| Boosted Trees        | Classification, regression                                     | High accuracy, robust to overfitting            | Computationally expensive, sensitive to noise     | Classification, Regression |
| SVM                  | Classification, regression                                     | Effective in high-dimensional spaces            | Sensitive to choice of kernel function            | Classification, Regression |
| Ensemble Methods     | Classification, regression                                     | Improved performance, robustness                | Complexity, may overfit with correlated models    | Classification, Regression |
| K-Means Clustering   | Unsupervised clustering                                        | Simple, scalable                                | Sensitivity to initial centroids, K parameter     | Clustering                 |
| K Nearest Neighbors  | Classification, regression, pattern recognition                | Simple, non-parametric                          | Computationally expensive during prediction       | Classification, Regression |

<div style="page-break-after:always;"></div>

### Linear Regression:

1. **What is the fundamental assumption of linear regression?**

   - The fundamental assumption of linear regression is that there exists a linear relationship between the independent variables (features) and the dependent variable (target).

2. **How do you interpret the coefficients in linear regression?**

   - The coefficients in linear regression represent the change in the dependent variable for a one-unit change in the corresponding independent variable, holding all other variables constant.

3. **Can you explain the difference between simple linear regression and multiple linear regression?**

   - Simple linear regression involves only one independent variable, whereas multiple linear regression involves two or more independent variables.

4. **What is the purpose of the cost function in linear regression, and how is it minimized?**

   - The purpose of the cost function is to measure the difference between the predicted values and the actual values. In linear regression, the cost function is typically the mean squared error (MSE), and it is minimized using optimization techniques like gradient descent.

5. **How do you assess the goodness of fit of a linear regression model?**

   - The goodness of fit of a linear regression model is often assessed using metrics such as R-squared, adjusted R-squared, and the root mean squared error (RMSE).

6. **What are some techniques to handle multicollinearity in linear regression?**

   - Techniques to handle multicollinearity include removing one of the correlated variables, combining the correlated variables into a single variable, or using regularization techniques such as ridge regression or lasso regression.

7. **Can linear regression be used for classification tasks? If not, why?**
   - No, linear regression is not suitable for classification tasks because it predicts continuous outcomes, not discrete class labels.

<div style="page-break-after:always;"></div>

### Logistic Regression:

1. **What is the difference between linear regression and logistic regression?**

   - Linear regression predicts continuous outcomes, while logistic regression predicts the probability of a binary outcome.

2. **How does logistic regression model the probability of a binary outcome?**

   - Logistic regression models the log-odds of the probability of the positive class using a logistic function.

3. **What is the logistic function, and how is it used in logistic regression?**

   - The logistic function (sigmoid function) maps any real-valued number to the range [0, 1], making it suitable for modeling probabilities in logistic regression.

4. **What is the purpose of the log-odds ratio in logistic regression?**

   - The log-odds ratio in logistic regression represents the linear relationship between the independent variables and the log-odds of the probability of the positive class.

5. **How do you interpret the coefficients in logistic regression?**

   - The coefficients in logistic regression represent the change in the log-odds of the probability of the positive class for a one-unit change in the corresponding independent variable.

6. **What is the likelihood function in logistic regression, and how is it maximized?**

   - The likelihood function measures how well the model predicts the observed data, and it is maximized using optimization techniques such as gradient ascent.

7. **How can you handle multicollinearity in logistic regression?**
   - Similar to linear regression, multicollinearity in logistic regression can be handled by removing correlated variables, combining them, or using regularization techniques.

<div style="page-break-after:always;"></div>

### Gaussian Naive Bayes:

1. **What is the fundamental assumption of Naive Bayes classifiers?**

   - The fundamental assumption of Naive Bayes classifiers is that features are conditionally independent given the class label, even though this assumption may not hold true in reality.

2. **How does the Naive Bayes classifier make predictions?**

   - Naive Bayes calculates the probability of each class given the input features using Bayes' theorem and selects the class with the highest probability.

3. **What is the Gaussian Naive Bayes classifier, and when is it appropriate to use?**

   - Gaussian Naive Bayes assumes that the continuous features follow a Gaussian (normal) distribution. It is appropriate to use when the features have continuous values that approximate a normal distribution.

4. **How do you estimate the parameters of a Gaussian Naive Bayes model?**

   - The parameters of a Gaussian Naive Bayes model, such as mean and variance for each class-feature combination, are estimated from the training data.

5. **Can Naive Bayes handle continuous and categorical features?**

   - Yes, Naive Bayes can handle both continuous and categorical features. Gaussian Naive Bayes is suitable for continuous features, while other variants like Multinomial Naive Bayes and Bernoulli Naive Bayes are suitable for categorical features.

6. **How does Naive Bayes deal with missing data?**

   - Naive Bayes can handle missing data by ignoring the missing values during the probability estimation process.

7. **What are the advantages and disadvantages of Naive Bayes classifiers?**
   - Advantages include simplicity, scalability, and efficiency. However, the main disadvantage is the strong assumption of feature independence, which may not hold true in many real-world scenarios.

<div style="page-break-after:always;"></div>

### Random Forest:

1. **What is the basic idea behind the Random Forest algorithm?**

   - Random Forest is an ensemble learning method that builds multiple decision trees during training and outputs the mode of the classes for classification or the average prediction for regression.

2. **How does a Random Forest differ from a single decision tree?**

   - A Random Forest consists of multiple decision trees trained on random subsets of the data and features, whereas a single decision tree is trained on the entire dataset.

3. **How are decision trees combined in a Random Forest?**

   - In a Random Forest, decision trees are combined through a process called bagging (bootstrap aggregation), where each tree is trained on a bootstrap sample of the data and a random subset of features is considered at each split.

4. **What are the advantages of using a Random Forest?**

   - Random Forests are less prone to overfitting compared to individual decision trees, handle high-dimensional data well, and provide estimates of feature importance.

5. **What is the concept of bagging in Random Forest?**

   - Bagging (bootstrap aggregation) is the process of training each decision tree in the Random Forest on a bootstrap sample of the data, which involves sampling with replacement from the original dataset.

6. **How does Random Forest handle overfitting?**

   - Random Forests reduce overfitting by averaging predictions from multiple trees trained on different subsets of the data, thereby reducing variance.

7. **Can you explain the role of feature importance in Random Forest?**
   - Feature importance in Random Forest indicates the contribution of each feature to the predictive performance of the model. It is computed based on how much each feature reduces impurity across all decision trees in the forest.

<div style="page-break-after:always;"></div>

### Boosted Trees (XGBoost, AdaBoost, CatBoost):

1. **What is the concept of boosting in machine learning?**

   - Boosting is an ensemble learning technique that combines multiple weak learners (usually decision trees) sequentially, where each subsequent model corrects the errors of its predecessor, leading to a strong learner.

2. **How does boosting improve model performance?**

   - Boosting improves model performance by focusing on instances that are difficult to classify. It assigns higher weights to misclassified instances, allowing subsequent models to learn from their mistakes and improve overall accuracy.

3. **Can you explain the difference between AdaBoost, Gradient Boosting (GBDT), and XGBoost?**

   - AdaBoost assigns higher weights to misclassified instances and adjusts subsequent models accordingly. Gradient Boosting (GBDT) builds decision trees sequentially, optimizing a loss function using gradient descent. XGBoost is an optimized implementation of gradient boosting with additional regularization and performance enhancements.

4. **What is the purpose of the learning rate in boosted trees?**

   - The learning rate controls the contribution of each tree to the final ensemble. A lower learning rate makes the model more robust by taking smaller steps during optimization.

5. **How does AdaBoost adjust the weights of misclassified instances?**

   - AdaBoost increases the weights of misclassified instances and decreases the weights of correctly classified instances, allowing subsequent models to focus more on the misclassified instances.

6. **What are some regularization techniques used in boosted trees?**

   - Regularization techniques in boosted trees include controlling the depth of the trees, adding a penalty term to the loss function (such as L1 or L2 regularization), and using subsampling of the data or features.

7. **Can boosted trees handle missing values in data?**
   - Yes, boosted trees can handle missing values by considering them during the split finding process and imputing missing values based on surrogate splits.

<div style="page-break-after:always;"></div>

### SVM (Support Vector Machine):

1. **What is the intuition behind the Support Vector Machine (SVM) algorithm?**

   - The intuition behind SVM is to find the hyperplane that best separates the data points of different classes while maximizing the margin, which is the distance between the hyperplane and the nearest data points (support vectors).

2. **How does SVM find the optimal hyperplane for classification?**

   - SVM finds the optimal hyperplane by maximizing the margin, which is achieved through solving an optimization problem that involves finding the hyperplane with the largest margin while correctly classifying the training data.

3. **What is the role of the kernel function in SVM?**

   - The kernel function allows SVM to operate in a higher-dimensional feature space without explicitly calculating the coordinates of the data points in that space. It maps the input features into a higher-dimensional space, making it easier to find a separating hyperplane.

4. **Can SVM be used for both classification and regression tasks?**

   - Yes, SVM can be used for both classification and regression tasks. For regression, it's known as Support Vector Regression (SVR), which aims to find a hyperplane that best fits the data while minimizing deviations.

5. **How does SVM handle non-linearly separable data?**

   - SVM handles non-linearly separable data by transforming the input features into a higher-dimensional space using a kernel trick. This transformation enables SVM to find a hyperplane that can separate the classes even if they are not linearly separable in the original feature space.

6. **What are the advantages and disadvantages of SVM?**

   - Advantages of SVM include its effectiveness in high-dimensional spaces, ability to handle non-linearly separable data through the kernel trick, and robustness against overfitting. Disadvantages include the high computational cost for large datasets and the difficulty in choosing the appropriate kernel and hyperparameters.

7. **What are the key parameters in SVM, and how do they affect the model?**
   - Key parameters in SVM include the choice of kernel function, regularization parameter (C), and kernel parameters (such as gamma for RBF kernel). These parameters affect the complexity of the decision boundary, the trade-off between margin and classification error, and the model's generalization ability.

<div style="page-break-after:always;"></div>

### Ensemble Methods:

1. **What is the idea behind ensemble learning?**

   - Ensemble learning combines multiple individual models (learners) to improve predictive performance over any single model. The diversity among the individual models allows the ensemble to capture different aspects of the data and make more accurate predictions.

2. **Can you explain the difference between bagging and boosting?**

   - Bagging (Bootstrap Aggregating) involves training multiple models independently on different subsets of the data and combining their predictions through averaging or voting. Boosting, on the other hand, builds models sequentially, where each subsequent model corrects the errors of its predecessor by assigning higher weights to misclassified instances.

3. **How do ensemble methods combine multiple base models to make predictions?**

   - Ensemble methods combine multiple base models either by averaging their predictions (e.g., bagging) or by sequentially training models where each subsequent model corrects the errors of the previous ones (e.g., boosting).

4. **What are the advantages of ensemble methods over individual models?**

   - Ensemble methods can often achieve better predictive performance than individual models by reducing variance, mitigating overfitting, and improving robustness against noise in the data.

5. **Can you compare and contrast Gradient Boosting, Random Forest, and AdaBoost?**

   - Gradient Boosting builds models sequentially, optimizing a differentiable loss function using gradient descent. Random Forest builds multiple decision trees independently and combines their predictions through voting or averaging. AdaBoost builds models sequentially, adjusting the weights of misclassified instances at each iteration.

6. **What is the concept of feature importance in ensemble methods?**

   - Feature importance in ensemble methods measures the contribution of each feature to the predictive performance of the model. It is computed based on how much each feature reduces impurity (for decision tree-based models) or improves predictive performance across all models in the ensemble.

7. **How do you select the base models and meta-model in stacked generalization?**
   - In stacked generalization (stacking), multiple diverse base models are trained on the dataset, and their predictions are used as features for a meta-model (or blender), which combines their predictions to make the final prediction. The choice of base models and the meta-model depends on the dataset and problem domain, often selected based on experimentation and performance evaluation through cross-validation.

<div style="page-break-after:always;"></div>

### K-Means Clustering (Unsupervised):

1. **What is the objective of K-Means clustering?**

   - The objective of K-Means clustering is to partition a dataset into K clusters, where each data point belongs to the cluster with the nearest mean, minimizing the within-cluster variance.

2. **How does the K-Means algorithm assign data points to clusters?**

   - The K-Means algorithm assigns data points to clusters by iteratively updating cluster centroids and assigning each data point to the nearest centroid until convergence.

3. **What is the significance of the number of clusters (K) in K-Means?**

   - The number of clusters (K) in K-Means determines the granularity of the clustering solution. Selecting an appropriate K value is crucial as it directly impacts the interpretation and utility of the clustering results.

4. **How do you initialize the centroids in K-Means?**

   - Centroid initialization in K-Means can be performed randomly or using more sophisticated techniques like k-means++ to ensure a good starting point and improve convergence speed.

5. **Can K-Means handle categorical data?**

   - K-Means is primarily designed for numerical data and Euclidean distance calculation, so categorical data needs to be preprocessed into numerical form or transformed using techniques like one-hot encoding before applying K-Means.

6. **What are the limitations of K-Means clustering?**

   - Limitations of K-Means clustering include its sensitivity to the initial centroids, the need to specify the number of clusters beforehand, and its inability to handle non-convex clusters or clusters of varying sizes and densities effectively.

7. **How do you evaluate the quality of K-Means clusters?**
   - The quality of K-Means clusters can be evaluated using metrics such as the within-cluster sum of squares (WCSS), silhouette score, Davies–Bouldin index, or visual inspection of cluster separations and compactness.
