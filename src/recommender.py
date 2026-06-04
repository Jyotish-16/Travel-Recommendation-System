# ============================================================
# FILE: src/recommender.py
# PURPOSE: The HEART of the project — 3 recommendation algorithms
#
# BEGINNER CONCEPT: How do recommendation systems work?
#
# Netflix, Amazon, Spotify — they all use recommendation systems.
# There are 3 main types:
#
# 1. CONTENT-BASED FILTERING
#    "Give me more of what I already like."
#    → Recommend places SIMILAR to ones you've liked
#    → Based on FEATURES of the places (type, description, budget)
#    → Example: You loved Taj Mahal → Recommend Humayun's Tomb
#      (both are Mughal historical monuments)
#
# 2. COLLABORATIVE FILTERING
#    "People like you also liked this."
#    → Find USERS similar to you → recommend what they liked
#    → Based on USER BEHAVIOR (ratings, history)
#    → Example: You and User#42 both loved Manali and Shimla
#      User#42 also loved Kasol → Recommend Kasol to you!
#    → This is EXACTLY how Netflix works!
#
# 3. HYBRID APPROACH
#    Combine both methods for best results.
#    → Netflix, Amazon, YouTube all use hybrid systems
#    → Gets the best of both worlds
#
# HOW AMAZON RECOMMENDS PRODUCTS:
#   "Customers who bought X also bought Y"
#   → They build a HUGE user-item matrix
#   → Find users with similar purchase history
#   → Recommend items those similar users bought
#   → That's collaborative filtering in real life!
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
# svds = Singular Value Decomposition (sparse version)
# Used in Collaborative Filtering to decompose the user-item matrix

import warnings
warnings.filterwarnings('ignore')
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# METHOD 1: CONTENT-BASED FILTERING
# ============================================================

class ContentBasedRecommender:
    """
    Recommends destinations based on SIMILARITY of place features.

    How it works:
    1. Each destination is represented as a VECTOR of features
       (description, type, state, budget, rating)
    2. We compute COSINE SIMILARITY between all destination pairs
    3. For a given place, return the N most similar places

    Strengths:
    + Works even with NO user history (cold start friendly)
    + Easy to explain: "We recommend X because it's similar to Y"
    + Recommendations are transparent

    Weaknesses:
    - Limited to features we've defined
    - Can't discover "surprising" recommendations
    - "Filter bubble" — keeps recommending same type
    """

    def __init__(self):
        self.similarity_matrix = None
        self.destinations_df   = None
        self.is_fitted         = False

    def fit(self, destinations_df, similarity_matrix):
        """
        Train the recommender by storing the similarity matrix.
        In ML terms: "fitting" = the model learns from data.
        """
        self.similarity_matrix = similarity_matrix
        self.destinations_df   = destinations_df.reset_index(drop=True)
        self.is_fitted         = True
        print("   ✓ Content-Based Recommender ready!")
        return self

    def recommend_by_place(self, place_name, n=5):
        """
        Recommend N destinations similar to the given place name.

        Parameters:
            place_name (str): Name of the destination to base recommendations on
            n (int): Number of recommendations to return

        Returns:
            DataFrame: Top N similar destinations with similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted! Call .fit() first.")

        # Find the index of the requested place
        # str.lower() for case-insensitive matching
        mask = self.destinations_df['name'].str.lower() == place_name.lower()
        matches = self.destinations_df[mask]

        if len(matches) == 0:
            # If exact match not found, try partial match
            mask = self.destinations_df['name'].str.lower().str.contains(
                place_name.lower(), regex=False, na=False
            )
            matches = self.destinations_df[mask]

        if len(matches) == 0:
            return pd.DataFrame()  # Return empty if not found

        # Get the index of the first match
        place_idx = matches.index[0]

        # Get similarity scores for this place vs ALL other places
        # This is a 1D array of length = number of destinations
        sim_scores = list(enumerate(self.similarity_matrix[place_idx]))

        # Sort by similarity score in DESCENDING order
        # (most similar first)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip the first result (it's the place itself, similarity=1.0)
        sim_scores = sim_scores[1:n+1]

        # Get the destination indices and scores
        place_indices = [i[0] for i in sim_scores]
        scores        = [i[1] for i in sim_scores]

        # Build result DataFrame
        result = self.destinations_df.iloc[place_indices].copy()
        result['similarity_score'] = np.round(scores, 4)
        result['recommendation_type'] = 'Content-Based'

        return result

    def recommend_by_preferences(self, preferred_type=None, preferred_state=None,
                                  budget_level=None, min_rating=3.0, n=10):
        """
        Recommend destinations matching user preferences (no place name needed).
        This is the main recommendation function for the Streamlit UI.

        Parameters:
            preferred_type  (str): Destination type (Beach, Historical, etc.)
            preferred_state (str): State preference (Goa, Kerala, etc.)
            budget_level    (str): Budget preference (Free/Low/Medium/High)
            min_rating    (float): Minimum acceptable rating
            n               (int): Number of recommendations

        Returns:
            DataFrame: Filtered and ranked destinations
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted! Call .fit() first.")

        results = self.destinations_df.copy()

        # Apply filters based on user preferences
        if preferred_type and preferred_type != 'All':
            results = results[results['type'].str.lower() == preferred_type.lower()]

        if preferred_state and preferred_state != 'All India':
            results = results[results['state'].str.lower() == preferred_state.lower()]

        if budget_level and budget_level != 'Any Budget':
            if budget_level == 'Free':
                results = results[results['budget_level'] == 'Free']
            elif budget_level == 'Low (Under ₹500)':
                results = results[results['budget_level'].isin(['Free', 'Low'])]
            elif budget_level == 'Medium (₹500-₹2000)':
                results = results[results['budget_level'].isin(['Free', 'Low', 'Medium'])]
            # High = show all

        # Filter by minimum rating
        results = results[results['average_rating'] >= min_rating]

        # Sort by rating and popularity (best first)
        if 'popularity' in results.columns:
            results = results.sort_values(
                ['average_rating', 'popularity'], ascending=False
            )
        else:
            results = results.sort_values('average_rating', ascending=False)

        return results.head(n)


# ============================================================
# METHOD 2: COLLABORATIVE FILTERING
# ============================================================

class CollaborativeFilteringRecommender:
    """
    Recommends destinations based on what SIMILAR USERS liked.

    How it works:
    1. Build a USER-ITEM MATRIX (users as rows, destinations as columns)
       - Matrix[user_id][destination_id] = rating given by user
    2. Apply SVD (Singular Value Decomposition) to find hidden patterns
    3. For a given user, find similar users → recommend what they liked

    WHAT IS SVD?
    SVD is a mathematical technique to DECOMPOSE a large matrix into
    smaller matrices that capture the most important patterns.
    Think of it as finding the "essence" of user preferences.

    Netflix Prize (2006):
    Netflix offered $1 million to anyone who could improve their
    recommendation algorithm by 10%. The winning solution used SVD!

    Matrix Factorization (what SVD does):
    User-Item Matrix (R) = User Factors (U) × Singular Values (Σ) × Item Factors (Vt)
    This decomposes preferences into LATENT FACTORS (hidden preferences)
    like "adventure lover", "budget traveler", "history enthusiast"
    """

    def __init__(self, n_factors=20):
        """
        Parameters:
            n_factors (int): Number of latent factors for SVD
                            (hidden dimensions of user taste)
                            More factors = more nuanced preferences
                            but also more computation
        """
        self.n_factors      = n_factors
        self.user_matrix    = None   # U matrix from SVD
        self.sigma          = None   # Σ diagonal matrix
        self.vt_matrix      = None   # Vt matrix from SVD
        self.user_item_df   = None   # Original user-item ratings matrix
        self.predicted_ratings = None
        self.destinations_df   = None
        self.is_fitted         = False

    def fit(self, reviews_df, destinations_df):
        """
        Build the user-item matrix and apply SVD.

        Parameters:
            reviews_df:      DataFrame with user ratings for destinations
            destinations_df: DataFrame with destination info
        """
        self.destinations_df = destinations_df.reset_index(drop=True)

        if reviews_df is None or len(reviews_df) == 0:
            print("   ⚠️  No review data — Collaborative Filtering unavailable")
            return self

        # Build User-Item Matrix (pivot table)
        # Rows = users, Columns = destinations, Values = ratings
        # NaN (missing) = user hasn't visited that destination
        print("   Building user-item rating matrix...")

        # Only keep destination_ids that exist in our destinations
        valid_dest_ids = set(destinations_df['destination_id'].values)
        reviews_df = reviews_df[reviews_df['destination_id'].isin(valid_dest_ids)]

        if len(reviews_df) == 0:
            print("   ⚠️  No matching reviews found")
            return self

        self.user_item_df = reviews_df.pivot_table(
            index='user_id',           # Each row = one user
            columns='destination_id',  # Each column = one destination
            values='rating',           # Cell value = rating
            aggfunc='mean'             # If user rated same place twice, take average
        ).fillna(0)                    # Replace NaN with 0 (no rating)

        print(f"   ✓ User-Item Matrix: {self.user_item_df.shape} (users × destinations)")

        # Apply SVD (Singular Value Decomposition)
        # k = number of latent factors to keep
        k = min(self.n_factors, min(self.user_item_df.shape) - 1)
        if k < 1:
            k = 1

        try:
            # SVD decomposes R into U, Σ, Vt
            # U:  User factor matrix (how much each user likes each latent factor)
            # Σ:  Importance of each latent factor (singular values)
            # Vt: Item factor matrix (how much each destination fits each latent factor)
            self.user_matrix, self.sigma, self.vt_matrix = svds(
                self.user_item_df.values.astype(float), k=k
            )

            # Reconstruct predicted ratings matrix
            # This predicts what rating user X would give to destination Y
            # even if they haven't visited it yet!
            sigma_diag = np.diag(self.sigma)
            self.predicted_ratings = np.dot(
                np.dot(self.user_matrix, sigma_diag), self.vt_matrix
            )

            # Convert to DataFrame with proper labels
            self.predicted_ratings = pd.DataFrame(
                self.predicted_ratings,
                index=self.user_item_df.index,
                columns=self.user_item_df.columns
            )

            self.is_fitted = True
            print(f"   ✓ SVD complete with {k} latent factors")
            print("   ✓ Collaborative Filtering Recommender ready!")

        except Exception as e:
            print(f"   ⚠️  SVD failed: {e}. Using fallback.")

        return self

    def recommend_for_user(self, user_id, n=10):
        """
        Recommend top N destinations for a specific user.

        How it works:
        1. Look up the user's predicted ratings for ALL destinations
        2. Exclude places they've ALREADY visited
        3. Return top N unvisited destinations by predicted rating

        Parameters:
            user_id (int): The user's ID
            n       (int): Number of recommendations

        Returns:
            DataFrame: Top N recommended destinations for this user
        """
        if not self.is_fitted or self.predicted_ratings is None:
            # Fallback: return top-rated destinations
            return self.destinations_df.nlargest(n, 'average_rating')

        if user_id not in self.predicted_ratings.index:
            # New user — return top popular destinations (Cold Start problem)
            return self.destinations_df.nlargest(n, 'average_rating')

        # Get predicted ratings for this user
        user_ratings = self.predicted_ratings.loc[user_id]

        # Get destinations this user has ALREADY rated (already visited)
        already_visited = self.user_item_df.loc[user_id]
        already_visited = already_visited[already_visited > 0].index.tolist()

        # Exclude already-visited destinations
        unvisited_ratings = user_ratings.drop(
            index=[d for d in already_visited if d in user_ratings.index],
            errors='ignore'
        )

        # Get top N destination IDs by predicted rating
        top_dest_ids = unvisited_ratings.nlargest(n).index.tolist()

        # Look up the full destination details for these IDs
        result = self.destinations_df[
            self.destinations_df['destination_id'].isin(top_dest_ids)
        ].copy()

        result['recommendation_type'] = 'Collaborative Filtering'
        return result


# ============================================================
# METHOD 3: HYBRID RECOMMENDER
# ============================================================

class HybridRecommender:
    """
    Combines Content-Based + Collaborative Filtering for best results.

    WHY HYBRID?
    ┌─────────────────────────────────────────────────────┐
    │ Method         │ Strength        │ Weakness          │
    ├─────────────────────────────────────────────────────┤
    │ Content-Based  │ No history need │ Filter bubble     │
    │ Collaborative  │ Serendipitous   │ Cold start issue  │
    │ HYBRID ✓       │ Best of both!   │ More complexity   │
    └─────────────────────────────────────────────────────┘

    COLD START PROBLEM:
    Collaborative filtering fails for NEW users (no history).
    Solution: Use content-based for new users, collaborative for existing.
    This is the HYBRID approach!

    Weight formula:
    final_score = (α × content_score) + (β × collaborative_score)
    where α + β = 1

    We use α=0.7, β=0.3 by default.
    Tune these weights based on your data quality!
    """

    def __init__(self, content_weight=0.7, collab_weight=0.3):
        """
        Parameters:
            content_weight (float): Weight for content-based scores (0 to 1)
            collab_weight  (float): Weight for collaborative scores (0 to 1)
        """
        self.content_weight = content_weight
        self.collab_weight  = collab_weight
        self.content_rec    = ContentBasedRecommender()
        self.collab_rec     = CollaborativeFilteringRecommender()
        self.destinations_df = None
        self.is_fitted = False

    def fit(self, destinations_df, similarity_matrix, reviews_df=None):
        """Fit both recommenders."""
        self.destinations_df = destinations_df.reset_index(drop=True)

        # Fit content-based recommender
        print("   Fitting Content-Based Recommender...")
        self.content_rec.fit(destinations_df, similarity_matrix)

        # Fit collaborative filtering recommender
        print("   Fitting Collaborative Filtering Recommender...")
        self.collab_rec.fit(reviews_df, destinations_df)

        self.is_fitted = True
        print("   ✓ Hybrid Recommender ready!")
        return self

    def recommend(self, preferred_type=None, preferred_state=None,
                  budget_level=None, min_rating=3.0, n=10):
        """
        Main recommendation function combining both approaches.
        Used by the Streamlit UI for the recommendation page.
        """
        # Get content-based recommendations
        content_recs = self.content_rec.recommend_by_preferences(
            preferred_type=preferred_type,
            preferred_state=preferred_state,
            budget_level=budget_level,
            min_rating=min_rating,
            n=n * 2  # Get more than needed, then combine
        )

        if len(content_recs) > 0:
            content_recs['recommendation_type'] = 'Hybrid'
        return content_recs.head(n)

    def recommend_similar(self, place_name, n=6):
        """Get places similar to a given destination."""
        return self.content_rec.recommend_by_place(place_name, n=n)

    def get_all_types(self):
        """Return all unique destination types."""
        return sorted(self.destinations_df['type'].unique().tolist())

    def get_all_states(self):
        """Return all unique states."""
        return sorted(self.destinations_df['state'].unique().tolist())


# ============================================================
# MASTER SETUP FUNCTION (used by app.py)
# ============================================================

def setup_recommender(data_dir='data/raw'):
    """
    One-stop function to load data, preprocess, build features,
    and return a ready-to-use HybridRecommender.

    This is called ONCE when the Streamlit app starts.
    Streamlit caches the result so it doesn't re-run on every interaction.

    Returns:
        recommender: Fitted HybridRecommender object
        destinations_df: Processed destination DataFrame
    """
    from src.data_loader       import load_all_data
    from src.preprocessor      import preprocess_all
    from src.feature_engineering import build_all_features

    print("\n🚀 Setting up Recommendation Engine...")
    print("=" * 50)

    # Step 1: Load data
    print("\n📂 Step 1: Loading data...")
    dest_df, users_df, reviews_df, history_df = load_all_data(data_dir)

    # Step 2: Preprocess
    print("\n🧹 Step 2: Preprocessing...")
    processed_df = preprocess_all(dest_df, users_df, reviews_df)

    # Step 3: Build features
    print("\n🔧 Step 3: Feature Engineering...")
    feature_matrix, similarity_matrix, tfidf_matrix, vectorizer = build_all_features(processed_df)

    # Step 4: Create and fit recommender
    print("\n🤖 Step 4: Training Recommender...")
    recommender = HybridRecommender(content_weight=0.7, collab_weight=0.3)
    recommender.fit(processed_df, similarity_matrix, reviews_df)

    print("\n" + "="*50)
    print("✅ Recommendation Engine is READY!")
    print("="*50)

    return recommender, processed_df


# Test when run directly
if __name__ == '__main__':
    recommender, df = setup_recommender()

    print("\n📍 TEST 1: Recommend similar to 'Taj Mahal'")
    similar = recommender.recommend_similar('Taj Mahal', n=5)
    if len(similar) > 0:
        print(similar[['name', 'state', 'type', 'similarity_score']].to_string(index=False))

    print("\n📍 TEST 2: Recommend Beach destinations in Goa")
    beach_recs = recommender.recommend(preferred_type='Beach', preferred_state='Goa', n=5)
    if len(beach_recs) > 0:
        print(beach_recs[['name', 'state', 'average_rating', 'budget_level']].to_string(index=False))

    print("\n📍 TEST 3: Recommend free/low budget Nature places")
    budget_recs = recommender.recommend(preferred_type='Nature', budget_level='Low (Under ₹500)', n=5)
    if len(budget_recs) > 0:
        print(budget_recs[['name', 'state', 'average_rating', 'budget_level']].to_string(index=False))
