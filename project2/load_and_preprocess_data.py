import numpy as np
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .models import PenguinObservation


def load_and_preprocess_data():
    df = load_penguins().dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    feature_names = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    X = df[feature_names].copy()
    y = df['species'].copy()
    master_classes = sorted(list(y.unique()))
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Compute Median Absolute Deviation (MAD) safely
    mads = np.median(np.abs(X_train - np.median(X_train, axis=0)), axis=0) + 1e-6
    
    # Scale features for the Logistic Regression model 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Check if we already have records to avoid duplicates in the PenguinObservation database table model 
    if not PenguinObservation.objects.exists():
        observations = []
        for i in range(len(X_test_scaled)):
            obs = PenguinObservation(
                bill_length_mm=X_test_scaled[i][0], 
                bill_depth_mm=X_test_scaled[i][1], 
                flipper_length_mm=X_test_scaled[i][2],
                body_mass_g=X_test_scaled[i][3]
            )
            observations.append(obs)
    
        # Save all to the database at once, this populates the id field for each instance
        PenguinObservation.objects.bulk_create(observations)
    
    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test,
        'X_train_scaled': X_train_scaled, 
        'X_test_scaled': X_test_scaled,
        'scaler': scaler, 
        'mads': mads,
        'feature_names': feature_names, 
        'master_classes': master_classes
    }