# %%
#! /bin/python
# -*- coding: utf-8 -*-
# --------- Imports and load the dataset into a dataframe --------- #
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import for data manipulation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
# imports for models
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
# Imports to calculate accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Informations for reproductibility 
random_seed = 21

def data_generator(data):
    #  this script is to generate a random dataset from the originial one
    """

    @param data:
    @param seed:
    @return:
    """
    y = data['outcome']
    X = data.drop(['outcome'],axis=1)
    indexes_cols_drop = np.random.randint(len(X.columns),size=10)
    columns = list(X.columns)
    X = X.drop([columns[i] for i in range(len(columns)) if i in indexes_cols_drop],axis=1)
    nb_rows = np.random.randint(7000,len(X))
    indexes_rows = np.random.randint(len(X),size = nb_rows)
    data_for_project = pd.concat([X,y],axis=1)
    data_for_project = data_for_project.loc[indexes_rows,:]
    return data_for_project

# Generate a random dataset
np.random.seed(random_seed)
data = pd.read_csv("./Datasets/Dataset_project_RS.csv", index_col=0)
df = data_generator(data)
print("Job finshed")

# %%
# --------- Data profiling --------- #
GenDataProfile = False
if GenDataProfile:
    # Generate the profiling report
    profile = ProfileReport(df, title="Dataset Profiling Report", explorative=True)
    profile.to_file("Dataset_profile_report.html")
    
# --------- Data preprocessing --------- #
# drop the constant column based on the report ('num_outboud_cmds')
df = df.drop(columns=['num_outbound_cmds'], axis=1)
df_2 = df.copy()
df

# %%
# --------- Parametering for models training and tests --------- #
# Label Encoding for the target variable (y=outcome)
target_le = LabelEncoder()  
df['outcome'] = target_le.fit_transform(df['outcome'])

# Split the data into features (X) and target (y)
X_base = df.drop('outcome', axis=1)
X = X_base.copy()
X_BASE_2 = X_base.copy()
X_2ndROUND=X_base.copy()
y = df['outcome']

# Identify non-numeric columns
non_numeric_cols = X.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)

# Apply One-Hot Encoding to non-numeric columns
X = pd.get_dummies(X, columns=non_numeric_cols)

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

# Apply StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Job finshed")

# %%
# --------- Model : Multilayer Perception model --------- #
# train the model and make predictions
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=random_seed)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# Evaluation metrics : Classification Report and Confusion Matrix
# Evaluation for MLP
print("\nMLP Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Classification Report:\n", classification_report(y_test, y_pred_mlp, target_names=target_le.classes_, labels=target_le.transform(target_le.classes_),zero_division=0))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_mlp)
# Set figure size for a larger plot
plt.figure(figsize=(12, 10)) 
# Plot heatmap with larger font sizes for better readability
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', annot_kws={"size": 10})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.show()

# %%
# ------------------- HyperParameter Tuning -------------------
DEBUG=False
if DEBUG:
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50, 25)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    # Initialize MLPClassifier
    mlp = MLPClassifier(max_iter=300, random_state=random_seed)
    # Perform grid search
    grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='recall', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    # Print the best parameters found
    print(f"Best parameters: {grid_search.best_params_}")
    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
else :
    # We know the best case by running it previously 
    best_model = MLPClassifier(max_iter=300, hidden_layer_sizes=(50, ), alpha=0.0001, solver='adam',activation='relu',learning_rate='constant',random_state=random_seed)
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    
# Evaluation metrics : Classification Report and Confusion Matrix
# Evaluation for MLP
print("\nBest MLP Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best, target_names=target_le.classes_, labels=target_le.transform(target_le.classes_),zero_division=0))
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
# Set figure size for a larger plot
plt.figure(figsize=(12, 10)) 
# Plot heatmap with larger font sizes for better readability
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', annot_kws={"size": 10})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.show()

# %%
# ----- Feature selection : drop unimportant features----- #
# --------- Model : Decision Tree --------- #
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn import tree

# Apply Label Encoding to non-numeric columns for Decision Tree
label_encoder_tree = LabelEncoder()
for col in non_numeric_cols:
    X_base[col] = label_encoder_tree.fit_transform(X_base[col])

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_base, y, test_size=0.3, random_state=random_seed)

# Initialize and train the model
clf_1 = DecisionTreeClassifier(criterion='gini', max_depth=10000, random_state=random_seed)
clf_1.fit(X_train_tree, y_train_tree)

# Get feature importance
importances = clf_1.feature_importances_
feat_importances = pd.DataFrame({'Feature': X_base.columns, 'Importance': importances})
feat_importances = feat_importances.sort_values(by='Importance', ascending=False)
# Display the most important features
print('Most Important :')
print(feat_importances.head(10))
print('Least Important :')
print(feat_importances.tail(10))
# DROP all the features with an importance of 0
zero_importance_features = feat_importances[feat_importances['Importance'] == 0]['Feature'].tolist()
print(f"Features with zero importance: {zero_importance_features} -> {len(zero_importance_features)}")
X_2ndROUND=X_2ndROUND.drop(columns=zero_importance_features)

# Identify non-numeric columns
non_numeric_cols = X_2ndROUND.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)

# Apply One-Hot Encoding to non-numeric columns
X_2ndROUND = pd.get_dummies(X_2ndROUND, columns=non_numeric_cols)

# Perform the train-test split
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2ndROUND, y, test_size=0.3, random_state=random_seed)

# Apply StandardScaler
scaler = StandardScaler()
X_train_2 = scaler.fit_transform(X_train_2)
X_test_2 = scaler.transform(X_test_2)

print("Job finished")

# %%
# Second round of grid search
DEBUG = False
if DEBUG :
    param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
    # Initialize MLPClassifier
    mlpGRID2 = MLPClassifier(max_iter=300, random_state=random_seed)
    
    # Perform grid search
    grid_search = GridSearchCV(mlpGRID2, param_grid, cv=3, scoring='recall', n_jobs=-1)
    grid_search.fit(X_train_2, y_train_2)
    # Print the best parameters found
    print(f"Best parameters: {grid_search.best_params_}")
    # Evaluate the best model on the test set
    best_model_2 = grid_search.best_estimator_
    y_pred_best_2 = best_model_2.predict(X_test_2)

else :
    # We know the best case by running it previously 
    best_model_2 = MLPClassifier(max_iter=300, hidden_layer_sizes=(50,), alpha=0.0001, solver='adam',activation='relu',learning_rate='constant',random_state=random_seed)
    best_model_2.fit(X_train_2, y_train_2)
    y_pred_best_2 = best_model_2.predict(X_test_2)
    
# Evaluation metrics : Classification Report and Confusion Matrix
# Evaluation for MLP
print("\nBest MLP Results 2:")
print("Accuracy:", accuracy_score(y_test_2, y_pred_best_2))
print("Classification Report:\n", classification_report(y_test_2, y_pred_best_2, target_names=target_le.classes_, labels=target_le.transform(target_le.classes_),zero_division=0))
# Compute confusion matrix
cm = confusion_matrix(y_test_2, y_pred_best_2)
# Set figure size for a larger plot
plt.figure(figsize=(12, 10)) 
# Plot heatmap with larger font sizes for better readability
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', annot_kws={"size": 10})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.show()

# %%
# --------- data change : Combine Rare Classes --------- #
# Find the frequency of each class in 'outcome'
outcome_counts = df_2['outcome'].value_counts()

# Define a threshold below which classes will be combined
threshold = 10

# Modify the existing 'outcome' column, replacing rare classes with "other malicious"
df_2['outcome'] = df_2['outcome'].apply(lambda x: x if outcome_counts[x] >= threshold else 'other malicious')

# Print class distribution after combining rare classes
print("Classes after combining rare ones:\n", df_2['outcome'].value_counts())

# --------- Decision Tree (for Feature Importance) --------- #
# Create a copy of the dataframe for the Decision Tree
X_tree_2 = df_2.copy()

# Identify non-numeric columns for Label Encoding in Decision Tree
non_numeric_cols = X_tree_2.select_dtypes(include=['object']).columns

# Apply Label Encoding to non-numeric columns for the Decision Tree
label_encoder_tree = LabelEncoder()
for col in non_numeric_cols:
    X_tree_2[col] = label_encoder_tree.fit_transform(X_tree_2[col])

# Split the data into features (X_tree_2) and target (y_tree)
X_tree_2 = X_tree_2.drop('outcome', axis=1)
y_tree = df_2['outcome']

# Perform the train-test split
X_train_tree_2, X_test_tree_2, y_train_tree_2, y_test_tree_2 = train_test_split(X_tree_2, y_tree, test_size=0.3, random_state=random_seed)

# Initialize and train the Decision Tree model
clf_2 = DecisionTreeClassifier(criterion='gini', max_depth=10000, random_state=random_seed)
clf_2.fit(X_train_tree_2, y_train_tree_2)

# Get feature importance
importances = clf_2.feature_importances_
feat_importances = pd.DataFrame({'Feature': X_tree_2.columns, 'Importance': importances})
feat_importances = feat_importances.sort_values(by='Importance', ascending=False)

# Display the most important features
print('Most Important :')
print(feat_importances.head(10))
print('Least Important :')
print(feat_importances.tail(10))

# DROP all the features with an importance of 0
zero_importance_features = feat_importances[feat_importances['Importance'] == 0]['Feature'].tolist()
print(f"Features with zero importance: {zero_importance_features} -> {len(zero_importance_features)}")

# Update X by dropping zero-importance features
df_2 = df_2.drop(columns=zero_importance_features)

# --------- Parametering for models training and tests --------- #
# Label Encoding for the target variable (y=outcome)
target_le_combined = LabelEncoder()  
df_2['outcome'] = target_le_combined.fit_transform(df_2['outcome'])

# Split the data into features (X) and target (y)
X_3 = df_2.drop('outcome', axis=1)
y_3 = df_2['outcome']

# Identify non-numeric columns for One-Hot Encoding
non_numeric_cols = X_3.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)

# Apply One-Hot Encoding to non-numeric columns
X_3 = pd.get_dummies(X_3, columns=non_numeric_cols)

# Perform the train-test split with stratification
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.3, stratify=y_3, random_state=random_seed)

# Apply StandardScaler to normalize features
scaler = StandardScaler()
X_train_3 = scaler.fit_transform(X_train_3)
X_test_3 = scaler.transform(X_test_3)

print("Job finished")

# %%
# Third round of grid search
DEBUG = False

if DEBUG:
    param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        } 
    # Initialize MLPClassifier
    mlp_3 = MLPClassifier(max_iter=300, random_state=random_seed)
    # Perform grid search
    grid_search = GridSearchCV(mlp_3, param_grid, cv=3, scoring='recall', n_jobs=-1)
    grid_search.fit(X_train_3, y_train_3)
    # Print the best parameters found
    print(f"Best parameters: {grid_search.best_params_}")
    # Evaluate the best model on the test set
    best_model_3 = grid_search.best_estimator_
    y_pred_best_3 = best_model_3.predict(X_test_3)
else:
    # We know the best case by running it previously 
    best_model_3 = MLPClassifier(max_iter=300, hidden_layer_sizes=(50,), alpha=0.0001, solver='adam',activation='relu',learning_rate='constant',random_state=random_seed)
    best_model_3.fit(X_train_3, y_train_3)
    y_pred_best_3 = best_model_3.predict(X_test_3)

# Evaluation metrics : Classification Report and Confusion Matrix
# Evaluation for MLP
print("\nBest MLP Results 3:")
print("Accuracy:", accuracy_score(y_test_3, y_pred_best_3))
print("Classification Report:\n", classification_report(y_test_3, y_pred_best_3, target_names=target_le_combined.classes_, labels=target_le_combined.transform(target_le_combined.classes_),zero_division=0))

# Compute confusion matrix
cm = confusion_matrix(y_test_3, y_pred_best_3)

# Set figure size for a larger plot
plt.figure(figsize=(12, 10)) 

# Plot heatmap with larger font sizes for better readability
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', annot_kws={"size": 10})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.show()

# %%
# ----------------- BONUS : performance of other models with initial dataset-----------------

# --------- Model : Logistic Regression model --------- #

# train the model
lr_model = LogisticRegression(random_state=random_seed, max_iter=10000)
lr_model.fit(X_train, y_train)
# Make predictions
y_pred_lr = lr_model.predict(X_test)

#  --------- Model : KNN --------- 
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = knn_model.predict(X_test)

# --------- Model : Decision Tree  --------- #
# Create a copy of the dataframe for the Decision Tree
X_tree = df.copy()

# Identify non-numeric columns for Label Encoding in Decision Tree
non_numeric_cols = X_tree.select_dtypes(include=['object']).columns

# Apply Label Encoding to non-numeric columns for the Decision Tree
label_encoder_tree = LabelEncoder()
for col in non_numeric_cols:
    X_tree[col] = label_encoder_tree.fit_transform(X_tree[col])

# Split the data into features (X_tree) and target (y_tree)
X_tree = X_tree.drop('outcome', axis=1)
y_tree = df['outcome']

# Perform the train-test split
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.3, random_state=random_seed)

# Initialize and train the Decision Tree model
clf = DecisionTreeClassifier(criterion='gini', max_depth=10000, random_state=random_seed)
clf.fit(X_train_tree, y_train_tree)

# Make predictions
y_pred_clf = clf.predict(X_test_tree)

#  --------- Evalutations --------- 

# Evaluation for LR
print("\nLR Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr, target_names=target_le.classes_, labels=target_le.transform(target_le.classes_), zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Evaluation for Decision Tree
print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_clf))
print("Classification Report:\n", classification_report(y_test, y_pred_clf, target_names=target_le.classes_, labels=target_le.transform(target_le.classes_), zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_clf))

# Evaluation for KNN
print("\nKNN Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn, target_names=target_le.classes_, labels=target_le.transform(target_le.classes_), zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# Accuracy Comparaison
print("\n RECAP : \n")
print(f"Decision tree Accuracy: {accuracy_score(y_test, y_pred_clf):.2f}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")
print(f"MLP Accuracy: {accuracy_score(y_test, y_pred_mlp):.2f}")
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")


