# By Malak Mohamed Osman 2205059  Fadya Hesham ELOraby 2205177  Abdelrahman Ayman Saad 2205033
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# arff file mt2asm l 7agten header feha metadata (column names) w l actual data
data, meta = arff.loadarff('KDDTrain+_20Percent.arff') # store data and columns name 3shan hst3mlha wna b3ml l df
columns = meta.names() 
df = pd.DataFrame(data, columns=columns) # creates a df from the data array and assigns the extracted column names to it.
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x) # decoding 3shan l data kant bytes 3yza a7wlha l string (a3ml decode lw kant byte)

print("A sample of data before: ") 
print(df.head())

# count the number of normal and anomaly instances
normal_count = df[df['class'] == 'normal'].shape[0] # shape[0] returns the number of rows 
anomaly_count = df[df['class'] == 'anomaly'].shape[0]
print(normal_count)
print(anomaly_count)

labels = ['Normal', 'Anomaly']
counts = [normal_count, anomaly_count]
plt.figure(figsize=(6, 4))
plt.bar(labels, counts, color=['blue', 'pink'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

class_labels = df['class'].unique().tolist() # convert anomaly and normal to 0 & 1
class_labels.sort()
class_dict = {label: idx for idx, label in enumerate(class_labels)}
df['class'] = df['class'].map(class_dict)
print(class_dict)

# convert any categorical column to numerical values.
categorical_columns = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
label_encoder = LabelEncoder() 
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column]) # scans the unique values in the column and assigns each a unique integer then replace it in the original df.

# Scaling
x = df.iloc[:, :-1]
y = df.iloc[:, -1:]
scaler = MinMaxScaler()
scaler.fit(x.values)
x_scaled = scaler.transform(x.values)
new_x = pd.DataFrame(data=x_scaled, columns=x.columns)
scaled_df = pd.concat([new_x, y.reset_index(drop=True)], axis=1)

print("A sample of data after: ")
print(scaled_df.head())


x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)

n_estimators = [10, 50, 100, 200, 500] # number of trees 
results = {}
# trying different values ( evaluate the performance of a Random Forest Classifier with different numbers of trees )
# bdwar 3la a7sn 3dd trees ydeny a3la accuracy
for n_estimators in n_estimators:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42) # initialize the model
    rf.fit(x_train, y_train) # train el model 3la l data
    y_pred = rf.predict(x_test) # Predicts the labels for the test dataset 
    accuracy = accuracy_score(y_test, y_pred) # ykarn ben l actual w l predicted w y7sb accuracy
    report = classification_report(y_test, y_pred) 
    results[n_estimators] = {'accuracy': accuracy, 'classification_report': report, 'model': rf}

best_accuracy = 0
best_n_estimators = None
best_classification_report = None
best_model = None

for n_estimators, result in results.items():
    accuracy = round(result['accuracy'], 5)
    print(f"\nResults for {n_estimators} Estimators:")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(result['classification_report'])
    if result['accuracy'] > best_accuracy:
        best_accuracy = result['accuracy']
        best_n_estimators = n_estimators
        best_classification_report = result['classification_report']
        best_model = result['model']

print("\nBest Model:")
best_accuracy = round(best_accuracy, 5)
print(f"Best Accuracy: {best_accuracy} with {best_n_estimators} Estimators")
print("Best Classification Report:")
print(best_classification_report)

estimators_list = list(results.keys())
accuracies = [results[n_estimators]['accuracy'] for n_estimators in estimators_list]

plt.figure(figsize=(8, 6))
plt.plot(estimators_list, accuracies, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
plt.title('Accuracy vs Number of Estimators in Random Forest', fontsize=14)
plt.xlabel('Number of Estimators', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(estimators_list)
plt.grid(True)
plt.show()

# feature importance (measuring how much each feature reduces the impurity) yshof anhy features b t influence l model's decision aktar 7aga
if best_model is not None:
    feature_importances = best_model.feature_importances_
    feature_names = x.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_n = 20
    top_features = feature_importance_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
    plt.title('Top 20 Features by Importance', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.gca().invert_yaxis()
    plt.show()

# Classification Report ex:
'''
 precision    recall  f1-score   support

           0       1.00      1.00      1.00      2349
           1       1.00      1.00      1.00      

    accuracy                           1.00      5039
   macro avg       1.00      1.00      1.00      5039
weighted avg       1.00      1.00      1.00      5039

0 (anomaly)
1 (normal)
precision: percentage of predictions for this class that are correct. tp/tp+fp
Recall:  The percentage of actual instances of this class that are correctly identified by the model. tp/tp+fn 
support: represent how many instances of each class are in your test dataset.
'''