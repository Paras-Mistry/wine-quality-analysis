# Wine-Quality-Analysis

The code performs exploratory data analysis on the wine quality dataset, normalizes the features, splits the dataset into training and testing sets, and evaluates the performance of five different machine learning models on the dataset. The machine learning models evaluated include Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and Support Vector Machine (SVM). The evaluation metrics used include accuracy score, F1 score, and classification report. The performance of each model is compared and presented in a tabular format.

# Steps Performed

1. Import the required libraries, namely numpy, pandas, matplotlib, seaborn, and several models from scikit-learn.
2. Read the CSV file containing the wine quality dataset using pandas.
3. View the top 5 rows of the dataset to get a general idea of the data.
4. Check the information and statistics of the dataset using the info() and describe() functions respectively.
5. Visualize the distribution of the target variable using a countplot using seaborn library.
6. Check for correlation between the features using a heatmap visualization from seaborn.
7. Explore the relationship between two features using scatterplot, violinplot, and boxplot visualizations.
8. Normalize the feature values by dividing each value with its respective maximum value.
9. Split the dataset into training and testing sets using train_test_split() function.
10. Initialize an empty dictionary to store the performance metrics of the models.
11. Create a logistic regression model using scikit-learn's LogisticRegression() class and fit it to the training data.
12. Predict the target variable using the trained model on the testing set and evaluate the accuracy, f1 score, and classification report using accuracy_score(), f1_score(), and classification_report() functions respectively.
13. Store the performance metrics of the model in the model_comp dictionary created in step 10.
14. Create a decision tree model using scikit-learn's DecisionTreeClassifier() class and fit it to the training data.
15. Predict the target variable using the trained model on the testing set and evaluate the accuracy, f1 score, and classification report using accuracy_score(), f1_score(), and classification_report() functions respectively.
16. Store the performance metrics of the model in the model_comp dictionary created in step 10.
17. Create a random forest model using scikit-learn's RandomForestClassifier() class and fit it to the training data.
18. Predict the target variable using the trained model on the testing set and evaluate the accuracy, f1 score, and classification report using accuracy_score(), f1_score(), and classification_report() functions respectively.
19. Store the performance metrics of the model in the model_comp dictionary created in step 10.
20. Create a K-Nearest Neighbors model using scikit-learn's KNeighborsClassifier() class and fit it to the training data.
21. Predict the target variable using the trained model on the testing set and evaluate the accuracy, f1 score, and classification report using accuracy_score(), f1_score(), and classification_report() functions respectively.
22. Store the performance metrics of the model in the model_comp dictionary created in step 10.
23. Create a support vector machine (SVM) model using scikit-learn's SVC() class and fit it to the training data.
24. Predict the target variable using the trained model on the testing set and evaluate the accuracy, f1 score, and classification report using accuracy_score(), f1_score(), and classification_report() functions respectively.
25. Store the performance metrics of the model in the model_comp dictionary created in step 10.
26. Create a pandas DataFrame from the model_comp dictionary and sort the values by the f1_score column.
27. Display the resulting DataFrame with a green background gradient using the style.background_gradient() function.
