# ============================================================
# FILE: src/evaluator.py
# PURPOSE: Measure how GOOD our recommendation system is
#
# BEGINNER CONCEPT: Why evaluate ML models?
#
#   Building a model is only HALF the work.
#   You must MEASURE its performance — otherwise how do you
#   know if it's actually working?
#
#   Real-world analogy:
#   A doctor doesn't just prescribe medicine — they follow up
#   to check if the patient is getting better (evaluation).
#
# KEY METRICS WE USE:
#
#   1. PRECISION@K
#      "Out of K recommendations, how many were actually relevant?"
#      Example: We recommend 5 beaches. User liked 4 of them.
#      Precision@5 = 4/5 = 0.80 (80% accurate)
#
#   2. COVERAGE
#      "What % of the total catalogue can our system recommend?"
#      If we only ever recommend 10 places out of 100, coverage = 10%
#      Low coverage = filter bubble (bad!)
#
#   3. DIVERSITY
#      "Are we recommending a VARIETY of places or just the same type?"
#      Good recommender = mix of beaches, mountains, temples, etc.
#
#   4. INTRA-LIST SIMILARITY
#      "How similar are the recommendations to each other?"
#      Low similarity = HIGH diversity (better!)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def precision_at_k(recommended_types, preferred_type, k=5):
    """
    Calculate Precision@K for category-based recommendations.

    FORMULA: Precision@K = (Relevant items in top K) / K

    Parameters:
        recommended_types (list): List of types of recommended destinations
        preferred_type    (str):  The type the user preferred
        k                 (int):  Number of recommendations to evaluate

    Returns:
        float: Precision score between 0 and 1
    """
    if not recommended_types or preferred_type == 'All':
        return 1.0  # If no filter applied, all are relevant

    top_k   = recommended_types[:k]
    relevant = sum(1 for t in top_k if t.lower() == preferred_type.lower())

    return relevant / k if k > 0 else 0.0


def calculate_coverage(recommended_ids, total_destinations):
    """
    Calculate what percentage of the catalogue we can recommend.

    FORMULA: Coverage = Unique recommended items / Total items

    Parameters:
        recommended_ids    (set): Set of destination IDs ever recommended
        total_destinations (int): Total number of destinations in catalogue

    Returns:
        float: Coverage score between 0 and 1 (higher = better)
    """
    if total_destinations == 0:
        return 0.0

    coverage = len(set(recommended_ids)) / total_destinations
    return round(coverage, 4)


def calculate_diversity(recommendations_df):
    """
    Calculate how DIVERSE the recommendations are.
    Diversity = variety of destination types recommended.

    FORMULA: Diversity = Unique types / Total recommendations

    High diversity means we're recommending a good mix:
    → "Beach, Mountain, Historical, Wildlife, Spiritual"
    is MORE diverse than:
    → "Beach, Beach, Beach, Beach, Beach"

    Parameters:
        recommendations_df (DataFrame): Recommended destinations

    Returns:
        float: Diversity score between 0 and 1 (higher = more diverse)
    """
    if len(recommendations_df) == 0:
        return 0.0

    if 'type' not in recommendations_df.columns:
        return 0.0

    unique_types = recommendations_df['type'].nunique()
    total_recs   = len(recommendations_df)

    diversity = unique_types / total_recs
    return round(diversity, 4)


def calculate_average_rating(recommendations_df):
    """
    Calculate the average rating of recommended destinations.
    Higher = we're recommending better-quality places.

    Parameters:
        recommendations_df (DataFrame): Recommended destinations

    Returns:
        float: Mean rating of all recommended places
    """
    if len(recommendations_df) == 0:
        return 0.0

    if 'average_rating' in recommendations_df.columns:
        return round(recommendations_df['average_rating'].mean(), 2)

    return 0.0


def evaluate_recommender(recommender, test_preferences=None):
    """
    Run a full evaluation of the recommendation system.

    This function:
    1. Tests different user preference combinations
    2. Calculates all metrics
    3. Returns a nice summary report

    Parameters:
        recommender: Fitted HybridRecommender object
        test_preferences: List of preference dicts to test

    Returns:
        dict: Evaluation report with all metrics
    """

    if test_preferences is None:
        # Default test cases — covering different scenarios
        test_preferences = [
            {'preferred_type': 'Beach',      'min_rating': 3.5},
            {'preferred_type': 'Historical', 'min_rating': 3.5},
            {'preferred_type': 'Nature',     'min_rating': 3.0},
            {'preferred_type': 'Spiritual',  'min_rating': 4.0},
            {'preferred_type': 'Adventure',  'min_rating': 3.0},
            {'preferred_type': 'Wildlife',   'min_rating': 3.5},
            {'preferred_type': 'Hill Station','min_rating': 3.5},
            {'preferred_type': 'Heritage',   'min_rating': 4.0},
        ]

    print("\n📊 EVALUATING RECOMMENDATION SYSTEM")
    print("=" * 55)
    print(f"{'Test Case':<20} {'Precision@5':>12} {'Diversity':>10} {'Avg Rating':>11}")
    print("-" * 55)

    all_recommended_ids = set()
    total_precision  = 0.0
    total_diversity  = 0.0
    total_avg_rating = 0.0
    valid_tests      = 0

    results = []

    for pref in test_preferences:
        try:
            recs = recommender.recommend(
                preferred_type=pref.get('preferred_type'),
                min_rating=pref.get('min_rating', 3.0),
                n=10
            )

            if len(recs) == 0:
                continue

            # Track all recommended IDs for coverage calculation
            if 'destination_id' in recs.columns:
                all_recommended_ids.update(recs['destination_id'].tolist())

            # Calculate metrics
            rec_types  = recs['type'].tolist() if 'type' in recs.columns else []
            precision  = precision_at_k(rec_types, pref.get('preferred_type', 'All'), k=5)
            diversity  = calculate_diversity(recs)
            avg_rating = calculate_average_rating(recs)

            total_precision  += precision
            total_diversity  += diversity
            total_avg_rating += avg_rating
            valid_tests      += 1

            test_name = pref.get('preferred_type', 'Mixed')[:18]
            print(f"{test_name:<20} {precision:>12.2%} {diversity:>10.2%} {avg_rating:>11.2f}")

            results.append({
                'test_case':  pref.get('preferred_type', 'Mixed'),
                'precision':  precision,
                'diversity':  diversity,
                'avg_rating': avg_rating,
                'num_recs':   len(recs)
            })

        except Exception as e:
            print(f"   ⚠️  Test failed for {pref}: {e}")

    # Calculate overall metrics
    if valid_tests > 0:
        avg_precision  = total_precision  / valid_tests
        avg_diversity  = total_diversity  / valid_tests
        avg_rating_all = total_avg_rating / valid_tests
    else:
        avg_precision = avg_diversity = avg_rating_all = 0.0

    total_destinations = len(recommender.destinations_df)
    coverage = calculate_coverage(all_recommended_ids, total_destinations)

    print("-" * 55)
    print(f"{'AVERAGE':<20} {avg_precision:>12.2%} {avg_diversity:>10.2%} {avg_rating_all:>11.2f}")
    print(f"\n📈 CATALOGUE COVERAGE: {coverage:.1%} ({len(all_recommended_ids)}/{total_destinations} destinations)")
    print("=" * 55)

    # Grade the system
    overall_score = (avg_precision * 0.4) + (avg_diversity * 0.3) + (coverage * 0.3)
    if overall_score >= 0.7:
        grade = "🏆 Excellent"
    elif overall_score >= 0.5:
        grade = "👍 Good"
    elif overall_score >= 0.3:
        grade = "📈 Fair — needs improvement"
    else:
        grade = "⚠️  Poor — check your data"

    print(f"\n🎯 OVERALL SYSTEM GRADE: {grade}")
    print(f"   Score: {overall_score:.2%}")

    return {
        'avg_precision':  avg_precision,
        'avg_diversity':  avg_diversity,
        'avg_rating':     avg_rating_all,
        'coverage':       coverage,
        'overall_score':  overall_score,
        'grade':          grade,
        'results':        results
    }


# Run directly to evaluate the system
if __name__ == '__main__':
    from src.recommender import setup_recommender

    recommender, df = setup_recommender()
    report = evaluate_recommender(recommender)

    print("\n📋 Detailed Results Table:")
    results_df = pd.DataFrame(report['results'])
    print(results_df.to_string(index=False))
