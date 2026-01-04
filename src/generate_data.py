
import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
locations = ['urban', 'suburban', 'rural']

data = {
    'area': np.random.randint(800, 4000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'age': np.random.randint(1, 50, n_samples),
    'location': np.random.choice(locations, n_samples)
}

df = pd.DataFrame(data)

# Create a synthetic price based on the features
df['price'] = (
    150 * df['area'] +
    50000 * df['bedrooms'] -
    2000 * df['age'] +
    np.random.normal(0, 50000, n_samples)  # Adding some noise
)

# Add a location-based price adjustment
location_adjustment = {
    'urban': 100000,
    'suburban': 50000,
    'rural': -50000
}
df['price'] += df['location'].map(location_adjustment)

# Ensure price is non-negative
df['price'] = df['price'].clip(lower=0)

# Save the dataset
df.to_csv('data/house_prices.csv', index=False)

print("Synthetic dataset 'house_prices.csv' created successfully.")
