# ============================================================
# FILE: src/preprocessor.py
# PURPOSE: Clean the raw data to make it ready for Machine Learning
#
# BEGINNER CONCEPT: Why do we preprocess data?
#
#   Raw data is MESSY — just like real life!
#   Imagine you collected 1000 survey forms, but:
#   - Some people left fields blank (missing values)
#   - Some wrote "Beach" and others wrote "beach" (inconsistent case)
#   - Ratings are on a 1-5 scale but distances are in kilometers
#     (different scales confuse ML algorithms)
#
#   PREPROCESSING = cleaning and standardizing data BEFORE ML
#
#   Real-world analogy:
#   A chef doesn't cook raw, dirty vegetables.
#   They wash, peel, and chop them first.
#   Preprocessing is the "washing and chopping" of data.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# MinMaxScaler: Scales numbers to a 0-1 range (normalisation)
# LabelEncoder: Converts text categories to numbers (e.g., "Beach" → 2)
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def handle_missing_values(df):
    """
    Handle missing (NaN) values in the dataset.

    BEGINNER CONCEPT: What are missing values?
    Missing values = empty cells in your dataset (shown as NaN = Not a Number)

    Strategies to handle missing values:
    1. DROP  → Delete rows/columns with missing values (risky if too many)
    2. FILL  → Replace with mean/median/mode or a fixed value
    3. IMPUTE → Use ML to predict the missing values (advanced)

    We use strategy 2 (FILL) because dropping would lose real data.
    """

    df = df.copy()  # Always work on a copy, never modify original data!

    missing_before = df.isnull().sum().sum()
    print(f"   Missing values found: {missing_before}")

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            # For NUMERIC columns → fill with the MEDIAN
            # We use MEDIAN (not mean) because it's less affected by outliers
            # Example: If ratings are [4, 4, 4, 1], mean=3.25 but median=4
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        elif df[col].dtype == 'object':
            # For TEXT/CATEGORY columns → fill with the MODE (most frequent value)
            # Example: If most places are "Historical", fill blanks with "Historical"
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    missing_after = df.isnull().sum().sum()
    print(f"   Missing values after cleaning: {missing_after} ✓")

    return df


def normalize_ratings(df, rating_col='average_rating'):
    """
    Normalize ratings to a 0-1 scale using Min-Max Scaling.

    BEGINNER CONCEPT: Why normalize?
    Imagine two features:
    - Rating: values between 1 and 5
    - Number of reviews: values between 0 and 10,000

    If we use these directly in ML, the model will think "10,000 reviews"
    is more important than "rating=5" just because the number is bigger.
    Normalization puts EVERYTHING on the same scale (0 to 1).

    Formula: (value - min) / (max - min)
    Example: rating 4.5, min=1, max=5 → (4.5-1)/(5-1) = 0.875
    """

    if rating_col in df.columns:
        scaler = MinMaxScaler()  # Creates a scaler object
        # reshape(-1, 1) converts a 1D array to a 2D column for the scaler
        df[f'{rating_col}_normalized'] = scaler.fit_transform(
            df[[rating_col]]
        )
    return df


def encode_categories(df):
    """
    Convert text categories to numbers so ML algorithms can understand them.

    BEGINNER CONCEPT: Why encode categories?
    Computers only understand NUMBERS, not text!
    ML algorithms cannot process "Beach" or "Mountain" directly.
    We convert them:
    - "Beach"      → 0
    - "Historical" → 1
    - "Nature"     → 2
    - "Spiritual"  → 3
    ... and so on.

    This is called LABEL ENCODING.

    Another method is ONE-HOT ENCODING (creates binary columns for each category)
    We use label encoding for ML models but keep the original text for display.
    """

    df = df.copy()
    le = LabelEncoder()

    # Encode 'type' column (destination category)
    if 'type' in df.columns:
        df['type_encoded'] = le.fit_transform(df['type'].astype(str))

    # Encode 'state' column
    if 'state' in df.columns:
        df['state_encoded'] = le.fit_transform(df['state'].astype(str))

    # Encode 'budget_level' column
    if 'budget_level' in df.columns:
        budget_map = {'Free': 0, 'Low': 1, 'Medium': 2, 'High': 3}
        df['budget_encoded'] = df['budget_level'].map(budget_map).fillna(1)

    return df


def clean_text(text):
    """
    Clean text data (descriptions, reviews) for text processing.

    WHY CLEAN TEXT?
    Raw text has lots of noise:
    - "The BEACH is amazing!!!" → "the beach is amazing"
    - "Great  place " (extra spaces) → "great place"
    Cleaned text gives better results in TF-IDF vectorization.
    """

    if pd.isna(text) or text == '':
        return ''

    # Convert to lowercase (so "Beach" and "beach" are treated the same)
    text = str(text).lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def normalize_reviews(df, reviews_df):
    """
    Enrich destination data with computed review statistics.
    Merges the average ratings from the reviews file into destinations.
    """

    if reviews_df is not None and len(reviews_df) > 0:
        # Calculate average rating per destination from reviews
        avg_ratings = reviews_df.groupby('destination_id')['rating'].agg(
            ['mean', 'count']
        ).reset_index()
        avg_ratings.columns = ['destination_id', 'review_avg_rating', 'review_count']

        # Merge with destinations
        df = df.merge(avg_ratings, on='destination_id', how='left')

        # Fill missing (destinations with no reviews) with the overall average
        df['review_avg_rating'] = df['review_avg_rating'].fillna(df['average_rating'])
        df['review_count'] = df['review_count'].fillna(0)

    return df


def preprocess_all(destinations_df, users_df=None, reviews_df=None):
    """
    Main preprocessing pipeline — runs all cleaning steps in order.

    BEGINNER CONCEPT: What is a Pipeline?
    A pipeline is a series of steps that data flows through,
    like water through pipes: Raw Data → Step1 → Step2 → Clean Data

    Our pipeline:
    1. Handle missing values
    2. Normalize numerical ratings
    3. Encode text categories to numbers
    4. Clean description text
    5. Merge review statistics
    """

    print("\n🧹 Starting Data Preprocessing Pipeline...")
    print("-" * 45)

    # Step 1: Handle missing values
    print("Step 1: Handling missing values...")
    destinations_df = handle_missing_values(destinations_df)

    # Step 2: Enrich with review data
    if reviews_df is not None:
        print("Step 2: Merging review statistics...")
        destinations_df = normalize_reviews(destinations_df, reviews_df)

    # Step 3: Normalize ratings to 0-1 scale
    print("Step 3: Normalizing ratings...")
    destinations_df = normalize_ratings(destinations_df)

    # Step 4: Encode text categories to numbers
    print("Step 4: Encoding categories...")
    destinations_df = encode_categories(destinations_df)

    # Step 5: Clean description text
    print("Step 5: Cleaning text descriptions...")
    destinations_df['description_clean'] = destinations_df['description'].apply(clean_text)
    destinations_df['tags'] = (
        destinations_df['type'].astype(str) + ' ' +
        destinations_df['state'].astype(str) + ' ' +
        destinations_df['best_time_to_visit'].astype(str) + ' ' +
        destinations_df['budget_level'].astype(str)
    ).apply(clean_text)

    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    destinations_df.to_csv('data/processed/destinations_processed.csv', index=False)
    print("   ✓ Saved processed data to data/processed/")

    print("\n✅ Preprocessing complete!")
    print(f"   Final shape: {destinations_df.shape[0]} rows × {destinations_df.shape[1]} columns")

    return destinations_df


# Run directly to test preprocessing
if __name__ == '__main__':
    from src.data_loader import load_all_data
    dest_df, users_df, reviews_df, history_df = load_all_data()
    processed_df = preprocess_all(dest_df, users_df, reviews_df)
    print(processed_df[['name', 'type', 'type_encoded', 'average_rating',
                         'average_rating_normalized', 'budget_encoded']].head(10))
