# ============================================================
# FILE: app.py
# PURPOSE: Main Streamlit web application for the
#          Intelligent Travel Recommendation System
#
# HOW TO RUN:
#   streamlit run app.py
#
# WHAT IS STREAMLIT?
#   Streamlit is a Python library that converts Python scripts
#   into beautiful, interactive web applications — NO HTML/CSS/JS needed!
#   It's the easiest way to build ML-powered web apps.
#
# STREAMLIT BASICS:
#   st.title()       → Big heading
#   st.write()       → Write text or data
#   st.sidebar.*     → Left sidebar elements
#   st.selectbox()   → Dropdown menu
#   st.slider()      → Slider for numbers
#   st.columns()     → Side-by-side layout
#   st.metric()      → Display a number with label
#   st.spinner()     → Loading animation
#   @st.cache_data   → Cache slow operations (don't re-run on every click)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys
import warnings
import re

warnings.filterwarnings('ignore')

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── PAGE CONFIGURATION (must be the FIRST Streamlit command) ────────────
st.set_page_config(
    page_title="Yatra AI — India Travel Recommender",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS STYLING ──────────────────────────────────────────────────
# We inject custom CSS to make the app look beautiful and premium
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700&display=swap');

/* Global styles */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero gradient title */
.hero-title {
    font-family: 'Inter',sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #2563EB;
    text-align: center;
    line-height: 1.2;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}

.hero-subtitle {
    text-align: center;
    color: #6B7280;
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 1rem;
}

/* Destination cards */
.dest-card {
    background: #1F2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    min-height: 260px;
}

.dest-card:hover {
    border-color: #2563EB;
    transform: translateY(-2px);
    transition: all 0.2s ease;
}

.card-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #FAFAFA;
    margin-bottom: 0.2rem;
}

.card-location {
    color: #9CA3AF;
    font-size: 0.85rem;
    margin-bottom: 0.8rem;
}

.card-desc {
    color: #D1D5DB;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 5px;
    margin-bottom: 4px;
}
.badge-type{
background: #DBEAFE;
color: #1D4ED8;
border: 1px solid #93C5FD;
}   
.badge-budget{
background: #DCFCE7;
color: #15803D;
border: 1px solid #86EFAC;
}
.badge-time{
background : #F3F4F6;
color: #4B5563;
border: 1px solid #D1D5DB;
}
.badge-free{
  background: #DCFCE7;
  color: #15803D;
  border: 1px solid #86EFAC;
}

/* Rating stars */
.rating-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 0.5rem;
}
.rating-num {
    font-size: 1rem;
    font-weight: 700;
    color: #FBBF24;
}
.rating-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    overflow: hidden;
}
.rating-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: #FBBF24;
}

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #E5E7EB;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.4rem
    border-bottom: 1px solid #374151;
}

/* Stats row */
.stat-box {
    background: #1F2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-num {
    font-size: 2rem;
    font-weight: 800;
    color: #2563EB;
}
.stat-label {
    font-size: 0.8rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Info box */
.info-box {
    background: #1F2937;
    border: 1px solid #374151;
    border-left: 4px solid #2563EB;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #D1D5DB;
    font-size: 0.9rem;
    line-height: 1.7;
}



/* Sidebar styling */
.sidebar-header {
    font-size: 1rem;
    font-weight: 600;
    color: #E5E7EB;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #374151;
}

/* Tab styling fix */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #9CA3AF;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
}

/* Button override */
.stButton > button {
    background: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover {
   background: #1D4ED8
    transform: none;
    box-shadow: none;
}

/* Force arrow cursor on selectbox instead of text cursor */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] input {
    cursor: default !important;
}
</style>
""", unsafe_allow_html=True)


# ── CACHED DATA LOADING ──────────────────────────────────────────────────
# @st.cache_resource: Cache the model so it loads ONCE, not on every interaction
# This is critical for performance — loading ML models is slow!
@st.cache_resource(show_spinner=False)
def load_recommender():
    """
    Load and initialize the recommendation engine.
    This function runs ONCE and the result is cached in memory.

    WHY CACHING?
    Without caching, the entire ML pipeline would re-run every time
    the user changes a slider or clicks a button. That would take 10-20 seconds!
    With caching, it loads once (~3 seconds) and then responds instantly.
    """
    from src.recommender import setup_recommender
    return setup_recommender()


# ── HELPER FUNCTIONS ────────────────────────────────────────────────────

def get_star_display(rating, max_stars=5):
    """Convert numeric rating to star emoji string."""
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = max_stars - full - half
    return '⭐' * full + ('✨' if half else '') + '☆' * empty


def get_budget_emoji(budget_level):
    """Return emoji + label for budget level."""
    mapping = {
        'Free':   ('🆓', 'Free Entry'),
        'Low':    ('💚', 'Budget Friendly'),
        'Medium': ('💛', 'Mid-Range'),
        'High':   ('💰', 'Premium'),
    }
    return mapping.get(budget_level, ('💰', budget_level))


def get_type_emoji(dest_type):
    """Return emoji for destination type."""
    mapping = {
        'Beach':       '🏖️',
        'Historical':  '🏛️',
        'Nature':      '🌿',
        'Spiritual':   '🕌',
        'Wildlife':    '🐯',
        'Hill Station':'⛰️',
        'Adventure':   '🧗',
        'Heritage':    '🏰',
        'Cultural':    '🎭',
        'Monument':    '🗿',
    }
    return mapping.get(dest_type, '📍')


def render_destination_card(row, show_similarity=False):
    """
    Render a beautiful destination card using HTML.
    Each card shows: name, location, description, badges, rating.
    """
    name        = row.get('name', 'Unknown Place')
    state       = row.get('state', '')
    dest_type   = row.get('type', '')

    raw_description = str(row.get('description', ''))

    # Remove HTML tags
    clean_description = re.sub(r'<.*?>', '', raw_description)

    description = clean_description[:200] + '...' if len(clean_description) > 200 else clean_description

    rating      = float(row.get('average_rating', 0))
    budget      = row.get('budget_level', 'Low')
    best_time   = row.get('best_time_to_visit', '')
    entry_fee   = row.get('entry_fee', 0)
    reviews     = int(row.get('total_reviews', row.get('review_count', 0)))

    type_emoji   = get_type_emoji(dest_type)
    budget_emoji, budget_label = get_budget_emoji(budget)
    rating_pct   = (rating / 5.0) * 100

    # Build entry fee badge
    if entry_fee == 0:
        fee_badge = '<span class="badge badge-free">Free Entry</span>'
    else:
        fee_badge = f'<span class="badge badge-budget">₹{int(entry_fee)} entry</span>'

    # Optional similarity score
    sim_badge = ''
    if show_similarity and 'similarity_score' in row:
        sim_pct = int(row['similarity_score'] * 100)
        sim_badge = f'<span class="badge badge-type">{sim_pct}% Match</span>'

    # Using dedent-like structure to prevent Markdown from interpreting spaces as code blocks
    card_html = f"""<div class="dest-card">
<div class="card-title">{type_emoji} {name}</div>
<div class="card-location">📍 {state}</div>
<div class="card-desc">{description}</div>
<div style="margin-bottom: 0.6rem;">
<span class="badge badge-type">{dest_type}</span>
<span class="badge badge-budget">{budget_emoji} {budget_label}</span>
<span class="badge badge-time">🗓️ {best_time}</span>
{fee_badge} {sim_badge}
</div>
<div class="rating-row">
<span class="rating-num">⭐ {rating}</span>
<div class="rating-bar-bg">
<div class="rating-bar-fill" style="width:{rating_pct}%;"></div>
</div>
<span style="color:#6B7280;font-size:0.8rem;">{reviews} reviews</span>
</div>
</div>"""
    st.markdown(card_html, unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────────────────

def render_sidebar(recommender, df):
    """Render the left sidebar with all user preference controls."""

    # Initialize session state variables to manage mutual exclusivity
    if 'pref_type' not in st.session_state: st.session_state.pref_type = 'All'
    if 'pref_state' not in st.session_state: st.session_state.pref_state = 'All India'
    if 'budget_level' not in st.session_state: st.session_state.budget_level = 'Any Budget'
    if 'search_query' not in st.session_state: st.session_state.search_query = ''
    if 'similar_to' not in st.session_state: st.session_state.similar_to = 'Select a place...'

    def on_dropdown_change():
        """When a dropdown changes, clear the search inputs."""
        st.session_state.search_query = ''
        st.session_state.similar_to = 'Select a place...'

    def on_search_change():
        """When a search input changes, clear the dropdown filters."""
        st.session_state.pref_type = 'All'
        st.session_state.pref_state = 'All India'
        st.session_state.budget_level = 'Any Budget'

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <div style="font-size:3rem;">🗺️</div>
            <div style="font-size:1.2rem; font-weight:800; color:#FF6B35;">Yatra AI</div>
            <div style="font-size:0.75rem; color:#6B7280; letter-spacing:0.1em;">TRAVEL RECOMMENDER</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(255,107,53,0.3);">', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-header">🎯 Your Travel Preferences</div>', unsafe_allow_html=True)

        # ── Destination Type ────────────────────────────────────────
        all_types = ['All'] + sorted(df['type'].unique().tolist())
        preferred_type = st.selectbox(
            "🏷️ Destination Type",
            options=all_types,
            key='pref_type',
            on_change=on_dropdown_change,
            help="What kind of destination do you prefer? Beach, Historical sites, Nature, etc."
        )

        # ── State ───────────────────────────────────────────────────
        all_states = ['All India'] + sorted(df['state'].unique().tolist())
        preferred_state = st.selectbox(
            "🗾 Preferred State",
            options=all_states,
            key='pref_state',
            on_change=on_dropdown_change,
            help="Filter destinations by Indian state"
        )

        # ── Budget ──────────────────────────────────────────────────
        budget_options = [
            'Any Budget',
            'Free',
            'Low (Under ₹500)',
            'Medium (₹500-₹2000)',
            'High (₹2000+)'
        ]
        budget_level = st.selectbox(
            "💰 Budget Level",
            options=budget_options,
            key='budget_level',
            on_change=on_dropdown_change,
            help="Choose your budget comfort level for entry fees and travel"
        )

        # ── Minimum Rating ──────────────────────────────────────────
        min_rating = st.slider(
            "⭐ Minimum Rating",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.5,
            help="Only show destinations with at least this rating"
        )

        # ── Number of Recommendations ───────────────────────────────
        n_recs = st.slider(
            "🔢 Number of Recommendations",
            min_value=3,
            max_value=21,
            value=9,
            step=3,
            help="How many destinations do you want to see?"
        )

        st.markdown('<hr style="border-color:rgba(255,107,53,0.3);">', unsafe_allow_html=True)

        # ── Search ──────────────────────────────────────────────────
        st.markdown('<div class="sidebar-header">🔍 Search Destinations</div>', unsafe_allow_html=True)
        search_query = st.text_input(
            "Search by name or keyword",
            key='search_query',
            on_change=on_search_change,
            placeholder="e.g. Taj Mahal, Goa, beach...",
            help="Search for a specific destination"
        )

        # ── Similar to ──────────────────────────────────────────────
        st.markdown('<div class="sidebar-header">🔗 Find Similar Places</div>', unsafe_allow_html=True)
        all_place_names = ['Select a place...'] + sorted(df['name'].tolist())
        similar_to = st.selectbox(
            "Similar to:",
            options=all_place_names,
            key='similar_to',
            on_change=on_search_change,
            help="Get destinations similar to a place you love"
        )

        st.markdown('<hr style="border-color:rgba(255,107,53,0.3);">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; color:#6B7280; font-size:0.75rem; padding: 0.5rem;">
            🤖 Powered by ML<br>
            Content-Based + Collaborative Filtering<br>
            Cosine Similarity · SVD · TF-IDF
        </div>
        """, unsafe_allow_html=True)

    return preferred_type, preferred_state, budget_level, min_rating, n_recs, search_query, similar_to


# ── TAB 1: RECOMMENDATIONS ───────────────────────────────────────────────

def render_recommendations_tab(recommender, df, preferred_type, preferred_state,
                                 budget_level, min_rating, n_recs,
                                 search_query, similar_to):
    """Main recommendations tab content."""

    # ── Hero Section ────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-title">Yatra AI</div>
    <div class="hero-subtitle">Personalized Travel Recommendations Across India</div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Quick Stats ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{len(df)}</div>
            <div class="stat-label">Destinations</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{df['state'].nunique()}</div>
            <div class="stat-label">States Covered</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{df['type'].nunique()}</div>
            <div class="stat-label">Destination Types</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        avg_r = round(df['average_rating'].mean(), 1)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{avg_r}⭐</div>
            <div class="stat-label">Average Rating</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Search Results ───────────────────────────────────────────────
    if search_query and search_query.strip():
        st.markdown(f'<div class="section-header">🔍 Search Results for "{search_query}"</div>',
                    unsafe_allow_html=True)

        search_results = df[
            df['name'].str.lower().str.contains(search_query.lower(), regex=False, na=False) |
            df['state'].str.lower().str.contains(search_query.lower(), regex=False, na=False) |
            df['type'].str.lower().str.contains(search_query.lower(), regex=False, na=False) |
            df['description'].str.lower().str.contains(search_query.lower(), regex=False, na=False)
        ]

        if len(search_results) > 0:
            st.success(f"Found {len(search_results)} destination(s) matching '{search_query}'")
            cols = st.columns(3)
            for i, (_, row) in enumerate(search_results.head(9).iterrows()):
                with cols[i % 3]:
                    render_destination_card(row)
        else:
            st.warning(f"No destinations found for '{search_query}'. Try different keywords!")

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Similar Places Section ───────────────────────────────────────
    if similar_to and similar_to != 'Select a place...':
        st.markdown(f'<div class="section-header">🔗 Places Similar to {similar_to}</div>',
                    unsafe_allow_html=True)

        with st.spinner(f"Finding places similar to {similar_to}..."):
            similar_recs = recommender.recommend_similar(similar_to, n=6)

        if len(similar_recs) > 0:
            st.markdown(f"""
            <div class="info-box">
                💡 <strong>How this works:</strong> We computed <strong>Cosine Similarity</strong>
                between <em>{similar_to}</em> and all other destinations using TF-IDF feature vectors
                of descriptions, types, and location data. The places below have the highest
                similarity scores — they share similar characteristics!
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(3)
            for i, (_, row) in enumerate(similar_recs.iterrows()):
                with cols[i % 3]:
                    render_destination_card(row, show_similarity=True)
        else:
            st.info(f"Could not find similar places for '{similar_to}'. Try another destination.")

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Main Recommendations ─────────────────────────────────────────
    # Build a label for what we're showing
    filter_parts = []
    if preferred_type  != 'All':         filter_parts.append(preferred_type)
    if preferred_state != 'All India':   filter_parts.append(preferred_state)
    if budget_level    != 'Any Budget':  filter_parts.append(budget_level)
    filter_label = ' · '.join(filter_parts) if filter_parts else 'All India'

    st.markdown(f'<div class="section-header">✨ Recommended for You — {filter_label}</div>',
                unsafe_allow_html=True)

    with st.spinner("🤖 AI is finding the best destinations for you..."):
        recommendations = recommender.recommend(
            preferred_type  = None if preferred_type  == 'All'        else preferred_type,
            preferred_state = None if preferred_state == 'All India'   else preferred_state,
            budget_level    = None if budget_level    == 'Any Budget'  else budget_level,
            min_rating      = min_rating,
            n               = n_recs
        )

    if len(recommendations) == 0:
        st.warning("😕 No destinations match your current filters. Try relaxing some criteria!")
        st.info("💡 Tip: Lower the minimum rating or change the destination type.")
    else:
        st.success(f"✅ Found {len(recommendations)} destinations matching your preferences!")

        # Show recommendation method explanation
        with st.expander("🤖 How did AI choose these recommendations?", expanded=False):
            st.markdown("""
            **Our Hybrid Recommendation Engine works in 3 steps:**

            1. **Content-Based Filtering (70% weight)**
               - Each destination is converted to a numerical vector using **TF-IDF**
               - TF-IDF captures unique keywords from descriptions (e.g., "houseboat", "tiger", "minaret")
               - **Cosine Similarity** finds destinations with similar feature vectors

            2. **Collaborative Filtering (30% weight)**
               - Builds a **User-Item Matrix** from 1,000 user reviews
               - Applies **SVD (Matrix Factorization)** to find hidden user preferences
               - "Users who liked X also liked Y" — predicts unvisited destinations

            3. **Hybrid Blending**
               - Final score = 70% × Content Score + 30% × Collaborative Score
               - Your filters (type, state, budget, rating) are applied on top

            This is exactly how **Netflix** and **Amazon** work! 🎯
            """)

        # Display in 3-column grid
        cols = st.columns(3)
        for i, (_, row) in enumerate(recommendations.iterrows()):
            with cols[i % 3]:
                render_destination_card(row)


# ── TAB 2: EXPLORE MAP & CHARTS ──────────────────────────────────────────

def render_charts_tab(df):
    """Charts and visualizations tab."""

    st.markdown('<div class="section-header">📊 Explore India\'s Travel Landscape</div>',
                unsafe_allow_html=True)

    # Set dark style for matplotlib
    plt.rcParams['figure.facecolor'] = '#0F1117'
    plt.rcParams['axes.facecolor']   = '#1A1D2E'
    plt.rcParams['text.color']       = '#FAFAFA'
    plt.rcParams['axes.labelcolor']  = '#9CA3AF'
    plt.rcParams['xtick.color']      = '#9CA3AF'
    plt.rcParams['ytick.color']      = '#9CA3AF'
    plt.rcParams['axes.edgecolor']   = '#374151'
    plt.rcParams['grid.color']       = '#1F2937'

    orange_palette = ['#FF6B35', '#FF8C5A', '#FFB347', '#FFC06B', '#FFD190',
                      '#E85D2B', '#D44F20', '#C04418', '#A83910', '#922D08']

    row1_col1, row1_col2 = st.columns(2)

    # Chart 1: Destination Types Distribution
    with row1_col1:
        st.markdown("**🏷️ Destination Types Distribution**")
        type_counts = df['type'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        bar_colors = [orange_palette[i % len(orange_palette)] for i in range(len(type_counts))]
        bars = ax.barh(type_counts.index, type_counts.values,
                       color=bar_colors, edgecolor='none')
        for bar, val in zip(bars, type_counts.values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color='#FAFAFA', fontsize=9, fontweight='600')
        ax.set_xlabel('Number of Destinations')
        ax.set_title('Types of Destinations', color='#FAFAFA', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 2: Top States by Destination Count
    with row1_col2:
        st.markdown("**🗾 Top States by Destination Count**")
        state_counts = df['state'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(range(len(state_counts)), state_counts.values,
                      color=orange_palette[:len(state_counts)], edgecolor='none', width=0.65)
        for bar, val in zip(bars, state_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', color='#FAFAFA', fontsize=9, fontweight='600')
        ax.set_xticks(range(len(state_counts)))
        ax.set_xticklabels(state_counts.index, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel('Number of Destinations')
        ax.set_title('Top 10 States', color='#FAFAFA', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    row2_col1, row2_col2 = st.columns(2)

    # Chart 3: Rating Distribution
    with row2_col1:
        st.markdown("**⭐ Rating Distribution**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['average_rating'], bins=15, color='#FF6B35',
                edgecolor='#1A1D2E', alpha=0.85, rwidth=0.85)
        ax.axvline(df['average_rating'].mean(), color='#FFB347',
                   linestyle='--', linewidth=2,
                   label=f"Mean: {df['average_rating'].mean():.2f}")
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('How Destinations Are Rated', color='#FAFAFA', pad=10)
        ax.legend(facecolor='#1A1D2E', edgecolor='#FF6B35', labelcolor='#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 4: Budget Level Pie Chart
    with row2_col2:
        st.markdown("**💰 Budget Level Distribution**")
        budget_order  = ['Free', 'Low', 'Medium', 'High']
        budget_counts = df['budget_level'].value_counts().reindex(budget_order).dropna()
        colors        = ['#34D399', '#60A5FA', '#FFB347', '#F87171']
        fig, ax       = plt.subplots(figsize=(7, 4))
        wedges, texts, autotexts = ax.pie(
            budget_counts.values,
            labels=budget_counts.index,
            autopct='%1.1f%%',
            colors=colors[:len(budget_counts)],
            startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(width=0.6, edgecolor='#0F1117', linewidth=2)
        )
        for t in texts:
            t.set_color('#FAFAFA')
            t.set_fontsize(10)
        for a in autotexts:
            a.set_color('#0F1117')
            a.set_fontsize(9)
            a.set_fontweight('bold')
        ax.set_title('Destination Budget Levels', color='#FAFAFA', pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 5: Top rated destinations
    st.markdown("**🏆 Top 10 Highest Rated Destinations**")
    top10 = df.nlargest(10, 'average_rating')[['name', 'state', 'type', 'average_rating']]
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = [orange_palette[i % len(orange_palette)] for i in range(len(top10))]
    bars = ax.bar(range(len(top10)), top10['average_rating'],
                  color=bar_colors, edgecolor='none', width=0.7)
    for bar, val in zip(bars, top10['average_rating']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1f}⭐', ha='center', color='#FAFAFA', fontsize=9, fontweight='700')
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels(
        [f"{r['name']}\n({r['state']})" for _, r in top10.iterrows()],
        rotation=30, ha='right', fontsize=8
    )
    ax.set_ylabel('Average Rating')
    ax.set_ylim(0, 5.5)
    ax.set_title('Top 10 Highest Rated Indian Destinations', color='#FAFAFA', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Data Table
    st.markdown('<div class="section-header">📋 Full Destination Dataset</div>', unsafe_allow_html=True)
    display_cols = ['name', 'state', 'type', 'average_rating', 'budget_level',
                    'best_time_to_visit', 'entry_fee']
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].sort_values('average_rating', ascending=False),
        use_container_width=True,
        height=400
    )


# ── TAB 3: LEARN ML ─────────────────────────────────────────────────────

def render_ml_tab():
    """Educational tab explaining ML concepts for beginners."""

    st.markdown('<div class="hero-title" style="font-size:2rem;">🎓 How AI Recommends Travel Destinations</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">A beginner-friendly guide to Recommendation Systems</div>',
                unsafe_allow_html=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # Netflix analogy
    st.markdown("""
    <div class="info-box">
        🎬 <strong>The Netflix Analogy</strong><br><br>
        Have you ever wondered how Netflix always seems to know what show you'll love next?
        Or how Amazon says "Customers who bought this also bought..."?<br><br>
        <strong>That's a Recommendation System!</strong> It's a type of AI that predicts
        what you'll like based on your preferences and what similar users enjoy.
        <strong>This project builds the same technology — but for Indian travel destinations!</strong>
    </div>
    """, unsafe_allow_html=True)

    # Three methods
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="dest-card">
            <div class="card-title">📝 Content-Based Filtering</div>
            <br>
            <strong style="color:#FF6B35;">What it does:</strong>
            <div class="card-desc">
                Recommends places <em>similar</em> to what you've liked before,
                based on the features (type, description, location).
            </div>
            <strong style="color:#FF6B35;">Real-life example:</strong>
            <div class="card-desc">
                You loved Taj Mahal (Mughal historical monument) →
                System recommends Humayun's Tomb (also Mughal historical).
            </div>
            <strong style="color:#FF6B35;">Algorithm:</strong>
            <div class="card-desc">TF-IDF + Cosine Similarity</div>
            <span class="badge badge-type">✅ No user history needed</span>
            <span class="badge badge-budget">⚠️ Filter bubble risk</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="dest-card">
            <div class="card-title">👥 Collaborative Filtering</div>
            <br>
            <strong style="color:#FF6B35;">What it does:</strong>
            <div class="card-desc">
                Finds users with <em>similar tastes</em> to you,
                then recommends what they enjoyed.
            </div>
            <strong style="color:#FF6B35;">Real-life example:</strong>
            <div class="card-desc">
                You & User#42 both loved Manali + Shimla.
                User#42 also loved Kasol → You'll probably love Kasol too!
            </div>
            <strong style="color:#FF6B35;">Algorithm:</strong>
            <div class="card-desc">SVD Matrix Factorization</div>
            <span class="badge badge-type">✅ Serendipitous discoveries</span>
            <span class="badge badge-budget">⚠️ Needs user history</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="dest-card">
            <div class="card-title">🔀 Hybrid Recommender</div>
            <br>
            <strong style="color:#FF6B35;">What it does:</strong>
            <div class="card-desc">
                Combines <em>both methods</em> for the best of both worlds.
                70% content-based + 30% collaborative.
            </div>
            <strong style="color:#FF6B35;">Real-life example:</strong>
            <div class="card-desc">
                Netflix, Spotify, YouTube ALL use hybrid systems!
                Pure content or pure collaborative alone isn't enough.
            </div>
            <strong style="color:#FF6B35;">Algorithm:</strong>
            <div class="card-desc">Weighted Hybrid Blend</div>
            <span class="badge badge-type">✅ Best accuracy</span>
            <span class="badge badge-budget">✅ Handles cold start</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step-by-step ML pipeline
    with st.expander("🔧 Step-by-Step: How our ML Pipeline works", expanded=True):
        steps = [
            ("1️⃣ Data Loading", "pandas.read_csv()",
             "Load 4 CSV files: destinations, users, reviews, user_history. Each file is a DataFrame (like an Excel table in Python)."),
            ("2️⃣ Data Preprocessing", "MinMaxScaler, LabelEncoder",
             "Clean missing values, normalize ratings to 0-1 scale, encode text categories to numbers. Computers only understand numbers!"),
            ("3️⃣ TF-IDF Vectorization", "TfidfVectorizer",
             "Convert place descriptions into numerical vectors. 'Houseboat' gets high score for Kerala backwaters. This captures the MEANING of each destination."),
            ("4️⃣ Cosine Similarity", "cosine_similarity()",
             "Compute the angle between every pair of destination vectors. Angle ≈ 0° means very similar places. Creates a 100×100 similarity matrix."),
            ("5️⃣ User-Item Matrix", "pivot_table()",
             "Build a matrix where rows=users, columns=destinations, values=ratings. This is the foundation of collaborative filtering."),
            ("6️⃣ SVD Decomposition", "scipy.sparse.linalg.svds",
             "Decompose the user-item matrix into hidden 'taste factors'. Reveals patterns like 'adventure lover' or 'heritage enthusiast'. Predicts ratings for unvisited places!"),
            ("7️⃣ Hybrid Blending", "Weighted average",
             "Final score = 0.7 × content_score + 0.3 × collaborative_score. Apply user filters (type, state, budget, rating). Return top N results."),
        ]
        for step_name, library, explanation in steps:
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.markdown(f"**{step_name}**")
                st.code(library, language='python')
            with col_b:
                st.markdown(f"""
                <div class="info-box" style="margin: 0.3rem 0;">{explanation}</div>
                """, unsafe_allow_html=True)
            st.markdown("")

    # Key concepts
    with st.expander("📐 Key ML Concept: Cosine Similarity Explained Visually"):
        st.markdown("""
        **Imagine each destination as an ARROW in space:**

        ```
        Taj Mahal  →  [historical=0.9, architecture=0.8, mughal=0.95, beach=0.0, wildlife=0.0]
        Humayun's  →  [historical=0.85, architecture=0.75, mughal=0.9, beach=0.0, wildlife=0.0]
        Goa Beach  →  [historical=0.1, architecture=0.0, mughal=0.0, beach=0.95, wildlife=0.0]
        ```

        **Cosine Similarity = cos(angle between two arrows)**

        - Taj Mahal vs Humayun's Tomb → angle ≈ 5°  → similarity ≈ 0.996 (very similar! ✅)
        - Taj Mahal vs Goa Beach      → angle ≈ 89° → similarity ≈ 0.02  (very different! ❌)

        **Why COSINE and not just distance?**
        Cosine ignores the LENGTH of the vector (how long the description is)
        and only compares the DIRECTION (what topics are discussed). This works
        much better for text data!
        """)

    with st.expander("🎯 Interview Questions & Answers"):
        qa = [
            ("Q: What is a recommendation system?",
             "A system that predicts user preferences and suggests relevant items. Used by Netflix (movies), Amazon (products), Spotify (music), and now our travel app (destinations)."),
            ("Q: What is Content-Based Filtering?",
             "It recommends items similar to what the user has liked in the past, based on item features. Example: If you liked Taj Mahal (historical, Mughal), recommend Humayun's Tomb (also historical, Mughal)."),
            ("Q: What is Collaborative Filtering?",
             "It finds users with similar preferences and recommends what those users liked. 'People like you also liked...' Netflix uses this to recommend movies watched by similar users."),
            ("Q: What is the Cold Start Problem?",
             "New users have no history, so collaborative filtering can't work. Solution: Use content-based filtering for new users until enough history is collected. That's why Streamlit/Spotify asks for preferences on signup!"),
            ("Q: What is TF-IDF?",
             "Term Frequency-Inverse Document Frequency. Converts text to numbers. Gives HIGH scores to words unique to a document (e.g., 'houseboat' for Kerala) and LOW scores to common words (e.g., 'the', 'is')."),
            ("Q: What is Cosine Similarity?",
             "A measure of similarity between two vectors based on the cosine of the angle between them. Value of 1.0 = identical, 0.0 = completely different. Used to find similar destinations."),
            ("Q: What is SVD?",
             "Singular Value Decomposition. Decomposes a large user-item matrix into smaller matrices that capture hidden 'taste factors'. The Netflix Prize winning algorithm used SVD!"),
            ("Q: How do you evaluate a recommendation system?",
             "Precision@K (how many of top-K are relevant), Coverage (% of catalogue recommended), Diversity (variety of types), RMSE (error in predicted ratings). We implement all these in evaluator.py."),
        ]
        for q, a in qa:
            st.markdown(f"**{q}**")
            st.markdown(f"""<div class="info-box">{a}</div>""", unsafe_allow_html=True)
            st.markdown("")


# ── TAB 4: EVALUATE ─────────────────────────────────────────────────────

def render_evaluate_tab(recommender):
    """Model evaluation tab."""

    st.markdown('<div class="section-header">📈 Model Performance Evaluation</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        🔬 <strong>Why evaluate?</strong> Building a model is only half the work.
        We must MEASURE its performance — just like a doctor checks if treatment works!
        Click the button below to run a full evaluation across 8 test scenarios.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Run Evaluation Now", key="eval_btn"):
        from src.evaluator import evaluate_recommender
        with st.spinner("Running evaluation tests... this takes a few seconds"):
            report = evaluate_recommender(recommender)

        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("Avg Precision@5", f"{report['avg_precision']:.0%}", col1),
            ("Avg Diversity",   f"{report['avg_diversity']:.0%}",  col2),
            ("Catalogue Coverage", f"{report['coverage']:.0%}",   col3),
            ("Avg Rating",     f"{report['avg_rating']:.2f}⭐",    col4),
        ]
        for label, value, col in metrics:
            with col:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-num">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
            🏆 <strong>System Grade: {report['grade']}</strong><br>
            Overall Score: {report['overall_score']:.1%}
        </div>
        """, unsafe_allow_html=True)

        if report['results']:
            st.markdown("**📋 Detailed Results by Test Case:**")
            results_df = pd.DataFrame(report['results'])
            results_df['precision']  = results_df['precision'].apply(lambda x: f"{x:.0%}")
            results_df['diversity']  = results_df['diversity'].apply(lambda x: f"{x:.0%}")
            results_df['avg_rating'] = results_df['avg_rating'].apply(lambda x: f"{x:.2f}⭐")
            st.dataframe(results_df, use_container_width=True)
    else:
        st.info("👆 Click the button above to evaluate the recommendation system!")


# ── MAIN APP ─────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""

    # Load the recommendation engine (cached after first run)
    with st.spinner("🚀 Loading AI Recommendation Engine... (first load takes ~5 seconds)"):
        recommender, df = load_recommender()

    # Render sidebar and get user preferences
    (preferred_type, preferred_state, budget_level,
     min_rating, n_recs, search_query, similar_to) = render_sidebar(recommender, df)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "✨ Recommendations",
        "📊 Explore & Charts",
        "🎓 Learn ML",
        "📈 Evaluate Model"
    ])

    with tab1:
        render_recommendations_tab(
            recommender, df,
            preferred_type, preferred_state, budget_level,
            min_rating, n_recs, search_query, similar_to
        )

    with tab2:
        render_charts_tab(df)

    with tab3:
        render_ml_tab()

    with tab4:
        render_evaluate_tab(recommender)


# Entry point
if __name__ == '__main__':
    main()
