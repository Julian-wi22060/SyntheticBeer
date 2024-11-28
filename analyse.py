import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to load and preprocess data
def load_and_preprocess_data(beer_csv, beer_excel):
    logging.info("Loading and preprocessing data.")
    # Load datasets
    beer_data_set = pd.read_csv(beer_csv)
    beer_descriptors = pd.ExcelFile(beer_excel)

    # Handle missing values
    beer_data_set['Name'].fillna('Unknown', inplace=True)
    logging.info("Handled missing values.")

    # Encode categorical variables
    label_encoders = {}
    for column in ['Style', 'Brewery']:
        encoder = LabelEncoder()
        beer_data_set[column] = encoder.fit_transform(beer_data_set[column])
        label_encoders[column] = encoder
    logging.info("Encoded categorical variables.")

    # Select numeric columns for modeling
    numeric_columns = [
        'ABV', 'Ave Rating', 'Min IBU', 'Max IBU',
        'Astringency', 'Body', 'Alcohol', 'Bitter',
        'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy',
        'Spices', 'Malty'
    ]

    model_data = beer_data_set[numeric_columns]
    logging.info("Data preprocessing complete.")
    return model_data, label_encoders


# Function to train a Gaussian Mixture Model and generate synthetic data
def train_gmm_and_generate_large_data(model_data, n_components=5, target_size_mb=10):
    logging.info(f"Training Gaussian Mixture Model with {n_components} components.")
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(model_data)

    # Estimate the number of samples needed for the target size
    sample_size = int((target_size_mb * 1024 * 1024) / (model_data.shape[1] * 8))  # Approximation
    logging.info(f"Generating approximately {sample_size} samples to match target size of {target_size_mb} MB.")

    synthetic_data, _ = gmm.sample(n_samples=sample_size)
    synthetic_data_df = pd.DataFrame(synthetic_data, columns=model_data.columns)
    logging.info("Large synthetic data generation complete.")
    return synthetic_data_df


# Function to clean negative values in synthetic data
def clean_synthetic_data(synthetic_data_df):
    logging.info("Cleaning negative values in synthetic data.")
    return synthetic_data_df.clip(lower=0)


# Function to visualize data distributions
def visualize_distributions(original_data, synthetic_data, columns_to_plot):
    logging.info("Visualizing data distributions.")
    for column in columns_to_plot:
        plt.figure(figsize=(8, 4))
        plt.hist(original_data[column], bins=30, alpha=0.5, label='Original Data', density=True)
        plt.hist(synthetic_data[column], bins=30, alpha=0.5, label='Synthetic Data', density=True)
        plt.title(f'Distribution Comparison for {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()
        plt.show()
    logging.info("Visualization complete.")


# Main execution workflow
def main():
    logging.info("Starting main workflow.")
    # File paths
    beer_csv = 'data/beer_data_set.csv'
    beer_excel = 'data/Beer Descriptors Simplified.xlsx'

    # Step 1: Load and preprocess data
    model_data, label_encoders = load_and_preprocess_data(beer_csv, beer_excel)

    # Step 2: Train GMM and generate synthetic data
    synthetic_data_df = train_gmm_and_generate_large_data(model_data, n_components=5, target_size_mb=10)

    # Step 3: Clean synthetic data
    cleaned_synthetic_data_df = clean_synthetic_data(synthetic_data_df)

    # Step 4: Visualize distributions
    columns_to_plot = ['ABV', 'Ave Rating', 'Min IBU', 'Max IBU', 'Bitter', 'Sweet', 'Sour']
    visualize_distributions(model_data, cleaned_synthetic_data_df, columns_to_plot)

    # Step 5: Export cleaned synthetic data
    export_path = 'data/large_cleaned_synthetic_beer_data.csv'
    cleaned_synthetic_data_df.to_csv(export_path, index=False)
    logging.info(f"Cleaned synthetic data exported to: {export_path}")


# Run the main function
if __name__ == "__main__":
    main()
