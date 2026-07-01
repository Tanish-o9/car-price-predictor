import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load and clean data
df = pd.read_csv('quikr_car.csv', on_bad_lines='skip')
df = df.dropna()
df = df[df['Price'] != 'Ask For Price']

# Clean Price column
df['Price'] = df['Price'].str.replace(',', '', regex=False)
df['Price'] = df['Price'].astype(int)

# Convert year to integer
df['year'] = df['year'].astype(int)

# Clean kms_driven column
df['kms_driven'] = df['kms_driven'].str.replace(' kms', '', regex=False)
df['kms_driven'] = df['kms_driven'].str.replace(',', '', regex=False)
df['kms_driven'] = df['kms_driven'].astype(int)

# Remove rows with missing fuel type and uncommon fuel type
df = df[df['fuel_type'].notna()]
df = df[df['fuel_type'] != 'Petrol + LPG']
df.reset_index(drop=True, inplace=True)

# Prepare features and target
X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessor with fixed OneHotEncoder
categorical_features = ['name', 'company', 'fuel_type']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# Create and fit pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.4f}")

# Test prediction
test_car = pd.DataFrame(
    [['Tata Manza', 'Tata', 2015, 10000, 'Diesel']],
    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
)
prediction = pipe.predict(test_car)
print(f"Predicted price for Tata Manza: Rs.{prediction[0]:,.0f}")