# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

def make_pickle_object():
# Loading the dataset
    df = pd.read_csv('diabetes.csv')

    # Renaming DiabetesPedigreeFunction as DPF
    df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

    # Replacing the 0 values from ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] by NaN
    df_copy = df.copy(deep=True)
    df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

    # Replacing NaN values by mean or median
    df_copy['Glucose'] = df_copy['Glucose'].fillna(df_copy['Glucose'].mean())
    df_copy['BloodPressure'] = df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean())
    df_copy['SkinThickness'] = df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median())
    df_copy['Insulin'] = df_copy['Insulin'].fillna(df_copy['Insulin'].median())
    df_copy['BMI'] = df_copy['BMI'].fillna(df_copy['BMI'].median())

    # Model Building
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X = df_copy.drop(columns='Outcome')
    y = df_copy['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Creating Random Forest Model
    classifier = RandomForestClassifier(n_estimators=20)
    classifier.fit(X_train, y_train)

    # Saving the model using pickle.dump()
    filename = 'diabetes-prediction-rfc-model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(classifier, file)  # Correct usage of pickle.dump()
    print("Model has successfully created")
    
    