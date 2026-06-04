# ============================================================
# FILE: src/data_loader.py
# PURPOSE: Load all 4 CSV files and give us a first look at the data
#
# BEGINNER CONCEPT: What is a DataFrame?
#   A DataFrame is like an Excel spreadsheet inside Python.
#   It has rows (records) and columns (features/attributes).
#   Pandas is the library that creates and manages DataFrames.
# ============================================================

import pandas as pd   # pandas = Python Data Analysis Library
import os             # os = Operating System module for file paths
import sys            # sys = System module for path manipulation

# Add the parent directory to path so we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_all_data(data_dir='data/raw'):
    """
    Load all 4 CSV files from the dataset.

    WHAT THIS FUNCTION DOES:
    - Reads all CSV files using pandas
    - Checks if files exist (if not, generates them)
    - Returns all 4 DataFrames for further processing

    Parameters:
        data_dir (str): Path to the folder containing CSV files

    Returns:
        tuple: (destinations_df, users_df, reviews_df, history_df)
    """

    # Define file paths for all 4 dataset files
    dest_path    = os.path.join(data_dir, 'destinations.csv')
    users_path   = os.path.join(data_dir, 'users.csv')
    reviews_path = os.path.join(data_dir, 'reviews.csv')
    hist_path    = os.path.join(data_dir, 'user_history.csv')

    # Check if data exists — if not, generate it automatically
    if not os.path.exists(dest_path):
        print("⚠️  Dataset not found. Generating sample data...")
        from src.data_generator import create_dataset
        create_dataset(data_dir)

    # ── Load each CSV file using pd.read_csv() ──────────────────────────
    # pd.read_csv() reads a CSV file and converts it into a DataFrame
    print("📂 Loading dataset files...")

    destinations_df = pd.read_csv(dest_path)
    print(f"   ✓ Destinations: {destinations_df.shape[0]} places, {destinations_df.shape[1]} columns")

    users_df = pd.read_csv(users_path)
    print(f"   ✓ Users:        {users_df.shape[0]} users, {users_df.shape[1]} columns")

    reviews_df = pd.read_csv(reviews_path)
    print(f"   ✓ Reviews:      {reviews_df.shape[0]} reviews, {reviews_df.shape[1]} columns")

    history_df = pd.read_csv(hist_path)
    print(f"   ✓ History:      {history_df.shape[0]} records, {history_df.shape[1]} columns")

    return destinations_df, users_df, reviews_df, history_df


def explore_data(destinations_df, reviews_df):
    """
    Print a summary of the dataset so we understand what we're working with.

    BEGINNER CONCEPT: EDA (Exploratory Data Analysis)
    Before building any ML model, ALWAYS explore your data first!
    Just like a doctor examines a patient before prescribing medicine,
    a data scientist explores data before applying algorithms.

    Key questions to answer during EDA:
    - How many records do we have?
    - What columns exist?
    - Are there missing values?
    - What does the data distribution look like?
    """

    print("\n" + "="*60)
    print("  📊 DATASET EXPLORATION SUMMARY")
    print("="*60)

    print("\n🗺️  DESTINATIONS OVERVIEW:")
    print(f"   Total destinations: {len(destinations_df)}")
    print(f"   States covered: {destinations_df['state'].nunique()}")
    print(f"   Destination types: {', '.join(destinations_df['type'].unique()[:5])}...")
    print(f"   Average rating: {destinations_df['average_rating'].mean():.2f}")

    print("\n📈 RATING DISTRIBUTION:")
    rating_counts = destinations_df['average_rating'].value_counts().sort_index()
    for rating, count in rating_counts.head(5).items():
        print(f"   Rating {rating}: {'█' * count} ({count} places)")

    print("\n🏷️  TOP DESTINATION TYPES:")
    type_counts = destinations_df['type'].value_counts()
    for dest_type, count in type_counts.head(6).items():
        print(f"   {dest_type:20s}: {count} destinations")

    print("\n💰 BUDGET LEVEL DISTRIBUTION:")
    budget_counts = destinations_df['budget_level'].value_counts()
    for level, count in budget_counts.items():
        print(f"   {level:10s}: {count} destinations")

    print("\n⭐ TOP 5 HIGHEST RATED DESTINATIONS:")
    top_rated = destinations_df.nlargest(5, 'average_rating')[['name', 'state', 'average_rating', 'type']]
    for _, row in top_rated.iterrows():
        print(f"   {row['name']:35s} | {row['state']:20s} | ⭐ {row['average_rating']}")

    print("\n✅ Data exploration complete!\n")


# Run this file directly to see the data summary
if __name__ == '__main__':
    dest_df, users_df, reviews_df, history_df = load_all_data()
    explore_data(dest_df, reviews_df)
