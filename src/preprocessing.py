import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Basic data cleaning.
    """
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Handle missing values (simple imputation or drop)
    # The dataset description says "14+ features". 
    # For simplicity in this assignment, we might drop rows with NaNs or fill them.
    # Let's fill with median for numerical and mode for categorical if any.
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Missing values found. Imputing...")
        for col in df.columns:
            if df[col].dtype == 'object':
                 df[col] = df[col].fillna(df[col].mode()[0])
            else:
                 df[col] = df[col].fillna(df[col].median())
    
    return df

def preprocess_data(df, target_column='num'):
    """
    Prepare features and target.
    Scaling is often done inside the pipeline, but here we can do it or return split data.
    The target 'num' usually indicates presence (values 1,2,3,4) or absence (0).
    We need binary classification: 0 vs >0.
    """
    df = clean_data(df)
    
    # Convert target to binary (0 = no disease, 1 = disease)
    # The UCI dataset often has 'num' as the target where 0 is healthy, 1-4 is disease.
    if target_column in df.columns:
        df['target'] = df[target_column].apply(lambda x: 1 if x > 0 else 0)
        df = df.drop(columns=[target_column])
    else:
        # Check if 'target' already exists (some versions might have it)
        if 'target' not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
            
    X = df.drop(columns=['target'])
    y = df['target']
    
    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
