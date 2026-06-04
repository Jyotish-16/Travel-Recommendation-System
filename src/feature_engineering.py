# ============================================================
# FILE: src/feature_engineering.py
# PURPOSE: Convert raw data into FEATURE VECTORS for ML algorithms
#
# BEGINNER CONCEPT: What is Feature Engineering?
#
#   Machines understand NUMBERS, not descriptions like
#   "Taj Mahal is a beautiful marble monument near Agra."
#
#   Feature Engineering = Converting text & data into numbers
#   that capture the MEANING and SIMILARITY between destinations.
#
#   Real-world analogy:
#   Think of each destination as a person on a dating app.
#   Their "profile" (features) = interests, location, budget.
#   We convert the profile into numbers so the app can find
#   compatible matches. That's exactly what we do here!
#
# KEY ML CONCEPTS TAUGHT HERE:
#   1. TF-IDF Vectorization  → Converts text to numbers
#   2. Cosine Similarity     → Measures how similar two places are
#   3. Feature Matrix        → A big table of all features as numbers
# ============================================================

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer: Converts text into a matrix of TF-IDF features
# TF-IDF = Term Frequency-Inverse Document Frequency
# It gives HIGH weight to words that are UNIQUE to a document
# and LOW weight to common words like "the", "is", "a"

from sklearn.metrics.pairwise import cosine_similarity
# cosine_similarity: Measures the angle between two feature vectors
# If angle = 0°  → similarity = 1.0 (identical)
# If angle = 90° → similarity = 0.0 (completely different)

from sklearn.preprocessing import MinMaxScaler
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# STEP 1: TF-IDF VECTORIZATION
# ============================================================

def build_tfidf_matrix(destinations_df):
    """
    Convert text descriptions into TF-IDF feature vectors.

    BEGINNER CONCEPT: What is TF-IDF?

    TF  = Term Frequency     → How often a word appears in THIS document
    IDF = Inverse Doc Freq   → How rare the word is across ALL documents

    Example:
    - "beach" appears in 50 out of 100 destination descriptions → LOW IDF score
      (it's common, not very informative)
    - "houseboat" appears in only 3 descriptions → HIGH IDF score
      (it's unique, very informative for Kerala backwaters!)

    TF-IDF = TF × IDF → Rewards UNIQUE, MEANINGFUL words

    HOW NETFLIX USES THIS:
    Netflix converts movie descriptions into TF-IDF vectors.
    "Action thriller with car chases" → [0.8, 0.0, 0.9, ...]
    Then finds other movies with similar vectors → similar movies!

    Parameters:
        destinations_df: DataFrame with 'description_clean' and 'tags' columns

    Returns:
        tfidf_matrix: numpy matrix of shape (n_destinations, n_features)
        vectorizer: fitted TfidfVectorizer (saved for transforming new queries)
    """

    # Combine description + tags into one rich text field for vectorization
    # This gives the model more context about each destination
    destinations_df['combined_text'] = (
        destinations_df['description_clean'].fillna('') + ' ' +
        destinations_df['tags'].fillna('') + ' ' +
        destinations_df['type'].fillna('').str.lower() + ' ' +
        destinations_df['state'].fillna('').str.lower()
    )

    # Create TF-IDF Vectorizer
    # max_features=500: Only keep the top 500 most important words
    # stop_words='english': Remove common words like "the", "is", "a"
    # ngram_range=(1,2): Consider both single words AND word pairs
    #   e.g., "wild life" as a pair gives more meaning than "wild" alone
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),   # unigrams and bigrams
        min_df=1,             # word must appear in at least 1 document
        sublinear_tf=True     # apply log normalization to TF (reduces impact of very frequent words)
    )

    # fit_transform does TWO things:
    # 1. FIT: Learn the vocabulary from our data
    # 2. TRANSFORM: Convert each description into a vector
    tfidf_matrix = vectorizer.fit_transform(destinations_df['combined_text'])

    print(f"   ✓ TF-IDF Matrix shape: {tfidf_matrix.shape}")
    print(f"     ({tfidf_matrix.shape[0]} destinations × {tfidf_matrix.shape[1]} text features)")

    return tfidf_matrix, vectorizer


# ============================================================
# STEP 2: NUMERIC FEATURE MATRIX
# ============================================================

def build_numeric_features(destinations_df):
    """
    Build a matrix of numerical features for each destination.

    WHY USE NUMERIC FEATURES IN ADDITION TO TEXT?
    Text alone misses important attributes:
    - Budget level (expensive vs cheap)
    - Rating (quality)
    - Popularity (how many people visit)
    - Entry fee

    We combine text + numeric features for a RICHER representation.

    Parameters:
        destinations_df: preprocessed DataFrame

    Returns:
        numeric_matrix: numpy array of normalized numeric features
    """

    # Select the most informative numeric columns
    numeric_cols = []

    if 'average_rating_normalized' in destinations_df.columns:
        numeric_cols.append('average_rating_normalized')
    elif 'average_rating' in destinations_df.columns:
        numeric_cols.append('average_rating')

    if 'popularity' in destinations_df.columns:
        numeric_cols.append('popularity')

    if 'budget_encoded' in destinations_df.columns:
        numeric_cols.append('budget_encoded')

    if 'type_encoded' in destinations_df.columns:
        numeric_cols.append('type_encoded')

    if 'state_encoded' in destinations_df.columns:
        numeric_cols.append('state_encoded')

    if 'entry_fee' in destinations_df.columns:
        numeric_cols.append('entry_fee')

    if len(numeric_cols) == 0:
        return np.zeros((len(destinations_df), 1))

    numeric_df = destinations_df[numeric_cols].fillna(0)

    # Normalize all numeric features to 0-1 scale
    # This ensures no single feature dominates (e.g., entry_fee could be 500
    # which is much larger than rating 4.5 — normalization fixes this)
    scaler = MinMaxScaler()
    numeric_matrix = scaler.fit_transform(numeric_df)

    print(f"   ✓ Numeric Feature Matrix: {numeric_matrix.shape}")

    return numeric_matrix


# ============================================================
# STEP 3: COMBINED FEATURE MATRIX
# ============================================================

def build_feature_matrix(destinations_df):
    """
    Combine TF-IDF text features AND numeric features into ONE matrix.

    WHY COMBINE?
    Think of it like judging a restaurant:
    - Description/reviews = text features (taste, ambiance description)
    - Price, rating, location = numeric features
    The BEST recommendation uses BOTH!

    We use a weighted combination:
    - Text features: 70% weight (descriptions are very informative)
    - Numeric features: 30% weight (ratings and budget also matter)

    Returns:
        feature_matrix: Final combined feature matrix
        tfidf_matrix: TF-IDF portion (for text similarity)
        vectorizer: Saved vectorizer (for new user queries)
    """

    print("\n🔧 Building Feature Matrix...")
    print("-" * 45)

    # Build TF-IDF text features
    print("   Building TF-IDF text vectors...")
    tfidf_matrix, vectorizer = build_tfidf_matrix(destinations_df)

    # Build numeric features
    print("   Building numeric feature vectors...")
    numeric_matrix = build_numeric_features(destinations_df)

    # Convert sparse TF-IDF matrix to dense numpy array for combining
    # Sparse matrix = stores only non-zero values (memory efficient)
    # Dense matrix = stores all values (needed for numpy operations)
    tfidf_dense = tfidf_matrix.toarray()

    # WEIGHTED COMBINATION: text=70%, numeric=30%
    # We multiply by weights so both contribute appropriately
    text_weight    = 0.70
    numeric_weight = 0.30

    # Resize numeric_matrix to match dimensions if needed
    # (pad with zeros if numeric has fewer columns)
    combined = np.hstack([
        tfidf_dense * text_weight,
        numeric_matrix * numeric_weight
    ])

    print(f"   ✓ Combined Feature Matrix: {combined.shape}")
    print(f"     (text {tfidf_dense.shape[1]} dims × 70%) + (numeric {numeric_matrix.shape[1]} dims × 30%)")

    return combined, tfidf_matrix, vectorizer


# ============================================================
# STEP 4: COSINE SIMILARITY MATRIX
# ============================================================

def compute_similarity_matrix(feature_matrix):
    """
    Compute Cosine Similarity between ALL pairs of destinations.

    BEGINNER CONCEPT: What is Cosine Similarity?

    Imagine each destination is a DIRECTION in space (a vector/arrow).
    Two destinations pointing in similar directions = similar places!

    The formula measures the ANGLE between two vectors:
    - cosine_similarity = cos(θ)
    - If θ = 0°  → cos(0)  = 1.0 → IDENTICAL places
    - If θ = 90° → cos(90) = 0.0 → COMPLETELY DIFFERENT places
    - If θ = 180°→ cos(180)= -1.0→ OPPOSITE places

    Why cosine and not Euclidean distance?
    Cosine ignores the LENGTH of the vector (how much text there is)
    and only looks at the DIRECTION (what topics are discussed).
    This is better for text data!

    HOW AMAZON USES THIS:
    "Customers who viewed this also viewed..."
    → Amazon computes product similarity using cosine similarity on
      product descriptions, categories, and purchase history!

    Returns:
        similarity_matrix: n×n matrix where similarity_matrix[i][j]
                          = similarity score between destination i and j
    """

    print("\n📐 Computing Cosine Similarity Matrix...")

    # cosine_similarity computes similarity between ALL pairs at once
    # This creates an n×n matrix (100×100 for our 100 destinations)
    similarity_matrix = cosine_similarity(feature_matrix)

    print(f"   ✓ Similarity Matrix shape: {similarity_matrix.shape}")
    print(f"     Each cell = similarity score between 2 destinations (0 to 1)")

    # Show a sample: similarity between first 3 destinations
    print(f"\n   Sample (first 3×3 corner of similarity matrix):")
    print(f"   {np.round(similarity_matrix[:3, :3], 3)}")

    return similarity_matrix


# ============================================================
# MAIN: Build everything
# ============================================================

def build_all_features(destinations_df):
    """
    Master function that builds ALL features and returns everything needed
    for the recommendation engine.

    Returns:
        feature_matrix:    Combined text + numeric feature matrix
        similarity_matrix: n×n cosine similarity matrix
        tfidf_matrix:      Raw TF-IDF matrix (for query matching)
        vectorizer:        Fitted TF-IDF vectorizer
    """

    feature_matrix, tfidf_matrix, vectorizer = build_feature_matrix(destinations_df)
    similarity_matrix = compute_similarity_matrix(feature_matrix)

    return feature_matrix, similarity_matrix, tfidf_matrix, vectorizer


# Test when run directly
if __name__ == '__main__':
    from src.data_loader import load_all_data
    from src.preprocessor import preprocess_all

    dest_df, users_df, reviews_df, history_df = load_all_data()
    processed_df = preprocess_all(dest_df, users_df, reviews_df)
    feature_matrix, similarity_matrix, tfidf_matrix, vectorizer = build_all_features(processed_df)

    print(f"\n🎉 Feature Engineering Complete!")
    print(f"   Feature Matrix:    {feature_matrix.shape}")
    print(f"   Similarity Matrix: {similarity_matrix.shape}")
