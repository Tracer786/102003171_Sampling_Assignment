from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE ,ADASYN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd
from imblearn.under_sampling import ClusterCentroids

# Load the data
url = 'https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv'
data = pd.read_csv(url)

# Split the data into features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Define the z-value and margin of error for each sampling technique
z = 2  
m = 0.075

n1 = int(np.ceil((z**2 * 0.5 * 0.5) / (m**2)))
n2 = int(np.ceil((z**2 * 0.05 * (1-0.05)) / (m**2)))
n3 = int(np.ceil((z**2 * 0.05 * (1-0.05)) / (m**2)))
n4 = int(np.ceil((z**2 * 0.05 * (1-0.05)) / (m**2)))
n5 = int(np.ceil((z**2 * 0.05 * (1-0.05)) / (m**2)))

# Define the sampling techniques and models
sampler1 = RandomUnderSampler(sampling_strategy='majority', random_state=12)
sampler2 = RandomOverSampler(sampling_strategy='minority', random_state=12)
sampler3 = SMOTE(sampling_strategy='minority', random_state=12)
sampler4 = TomekLinks(sampling_strategy='majority')
sampler5 = ClusterCentroids(random_state=42)

model1 = LinearDiscriminantAnalysis()
model2 = LogisticRegression(random_state=42,max_iter=500)
model3 = RandomForestClassifier(random_state=42)
model4 = SVC(random_state=42)
model5 = ExtraTreesClassifier(random_state=42)



# Define a dictionary to hold the sampling techniques and models
samplers = {
    'Sampling1': sampler1,
    'Sampling2': sampler2,
    'Sampling3': sampler3,
    'Sampling4': sampler4,
    'Sampling5': sampler5,
}
models = {
    'M1': model1,
    'M2': model2,
    'M3': model3,
    'M4': model4,
    'M5': model5,
}

# Evaluate each model on each sampling technique
results = {}
for sampler_name, sampler in samplers.items():
    if sampler_name == 'Sampling1':
        n = n1
    elif sampler_name == 'Sampling2':
        n = n2
    elif sampler_name == 'Sampling3':
        n = n3
    elif sampler_name == 'Sampling4':
        n = n4
    else:
        n = n5

    # Undersample or oversample the training data
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    # Limit the resampled data to the sample size
    if len(X_resampled) > n:
        X_resampled = X_resampled[:n]
        y_resampled = y_resampled[:n]
    
    for model_name, model in models.items():
        # Train the model on the resampled data
        model.fit(X_resampled, y_resampled)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        
        # Add the accuracy score to the results dictionary
        if model_name in results:
            results[model_name][sampler_name] = accuracy
        else:
            results[model_name] = {sampler_name: accuracy}

# Print the results
print('Results:')
print('        Sampling1   Sampling2   Sampling3   Sampling4   Sampling5')
for model_name, model_results in results.items():
    print(model_name, end='')
    for sampler_name in samplers.keys():
        if sampler_name in model_results:
            print(f'    {model_results[sampler_name]:.4f}   ', end='')
        else:
            print('              ', end='')
    print()         