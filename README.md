# 🗺️ Intelligent Travel Recommendation System

> **An AI-powered travel recommendation engine for Indian destinations — built with Python, Scikit-learn & Streamlit**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 What This Project Does

This project builds a **Machine Learning recommendation system** that suggests personalized Indian tourist destinations based on:
- Your preferred travel type (Beach, Historical, Nature, Wildlife, etc.)
- Budget level (Free → Luxury)
- State preference (All India or specific state)
- Minimum rating threshold
- Similarity to places you already love

**Inspired by how Netflix and Amazon build their recommendation engines!**

---

## 🧠 ML Concepts Implemented

| Concept | Where Used | Library |
|---|---|---|
| **Content-Based Filtering** | `src/recommender.py` | scikit-learn |
| **Collaborative Filtering** | `src/recommender.py` | scipy (SVD) |
| **Hybrid Recommendation** | `src/recommender.py` | Custom |
| **TF-IDF Vectorization** | `src/feature_engineering.py` | scikit-learn |
| **Cosine Similarity** | `src/feature_engineering.py` | scikit-learn |
| **SVD Matrix Factorization** | `src/recommender.py` | scipy |
| **Feature Engineering** | `src/feature_engineering.py` | numpy, pandas |
| **Data Preprocessing** | `src/preprocessor.py` | scikit-learn |
| **Model Evaluation** | `src/evaluator.py` | Custom metrics |

---

## 📁 Project Structure

```
Intelligent Travel Recommend System/
│
├── 📄 app.py                      ← 🚀 Main Streamlit web application
├── 📄 setup.py                    ← One-click setup & initialization
├── 📄 requirements.txt            ← All Python dependencies
│
├── 📁 src/                        ← ML source code (the brain!)
│   ├── __init__.py
│   ├── data_generator.py          ← Creates realistic India tourism dataset
│   ├── data_loader.py             ← Loads & explores the CSV files
│   ├── preprocessor.py            ← Cleans & encodes data for ML
│   ├── feature_engineering.py     ← TF-IDF, feature matrix, cosine similarity
│   ├── recommender.py             ← Content-based + Collaborative + Hybrid ML
│   └── evaluator.py               ← Precision@K, Coverage, Diversity metrics
│
├── 📁 data/
│   ├── raw/                       ← Original CSV files (auto-generated)
│   │   ├── destinations.csv       ← 100 Indian tourist places
│   │   ├── users.csv              ← 200 user profiles
│   │   ├── reviews.csv            ← 1,000 user ratings & reviews
│   │   └── user_history.csv       ← 500 travel history records
│   └── processed/                 ← Cleaned data after preprocessing
│
├── 📁 assets/                     ← Images and static files
│
└── 📁 .streamlit/
    └── config.toml                ← App theme (dark, orange accents)
```

---

## ⚡ Quick Start (3 Steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run setup (generates dataset + tests everything)
```bash
python setup.py
```

### Step 3 — Launch the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

---

## 🗄️ Dataset

This project uses a dataset modeled after the **[India Travel Recommender System Dataset](https://www.kaggle.com/datasets/amanmehra23/travel-recommendation-dataset)** on Kaggle.

**4 CSV files with the same schema:**

| File | Records | Description |
|---|---|---|
| `destinations.csv` | 100 | Indian tourist places with type, state, rating, budget |
| `users.csv` | 200 | User profiles with preferences and demographics |
| `reviews.csv` | 1,000 | User ratings (1–5) with review text |
| `user_history.csv` | 500 | Past travel records with satisfaction scores |

**Destinations cover all major Indian states:**
Delhi · Rajasthan · Kerala · Goa · Himachal Pradesh · Uttarakhand · Karnataka · Tamil Nadu · Maharashtra · Northeast India · Andaman & Nicobar

**Destination types:**
Beach · Historical · Nature · Spiritual · Wildlife · Hill Station · Adventure · Heritage · Cultural · Monument

---

## 🤖 How the Recommendation Engine Works

```
User Preferences
      │
      ▼
┌─────────────────────────────────────────┐
│         HYBRID RECOMMENDER              │
│                                         │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │ Content-Based│  │  Collaborative  │  │
│  │   (70%)      │  │  Filtering(30%) │  │
│  │              │  │                 │  │
│  │  TF-IDF      │  │  User-Item      │  │
│  │  Vectorize   │  │  Matrix         │  │
│  │  Descriptions│  │  SVD            │  │
│  │  Cosine Sim  │  │  Decomposition  │  │
│  └──────┬───────┘  └───────┬─────────┘  │
│         └────────┬─────────┘            │
│                  ▼                      │
│         Weighted Combination            │
│         + User Filters Applied          │
└──────────────────┬──────────────────────┘
                   ▼
         Top N Recommendations
```

### Content-Based Filtering
1. Each destination description → **TF-IDF vector** (500 dimensions)
2. Combine with numeric features (rating, popularity, budget, type)
3. Compute **Cosine Similarity** matrix (100×100)
4. For user preferences → find highest-scoring matching destinations

### Collaborative Filtering
1. Build **User-Item Rating Matrix** (200 users × 100 destinations)
2. Apply **SVD** (Singular Value Decomposition) with 20 latent factors
3. Reconstruct predicted ratings for all user-destination pairs
4. Recommend highest predicted-rating unvisited destinations

### Hybrid Blend
```python
final_score = (0.7 × content_score) + (0.3 × collaborative_score)
```

---

## 📊 Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| **Precision@5** | % of top-5 recs that match user preference | > 70% |
| **Coverage** | % of catalogue we can recommend | > 50% |
| **Diversity** | Variety of types in recommendations | > 40% |
| **Avg Rating** | Mean rating of recommended places | > 4.0 |

Run evaluation inside the app → **"📈 Evaluate Model"** tab.

---

## 🚀 Deployment (Streamlit Cloud)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Intelligent Travel Recommendation System"
git remote add origin https://github.com/YOUR_USERNAME/travel-recommender.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository → `app.py` as main file
5. Click **"Deploy"** → your app is live in 2 minutes! 🎉

### Step 3: Share your link
Your app URL will be: `https://YOUR_USERNAME-travel-recommender.streamlit.app`

---

## 📝 Resume Description

```
Intelligent Travel Recommendation System | Python, ML, Streamlit
• Built an AI-powered recommendation engine for 100+ Indian tourist destinations
  using Content-Based and Collaborative Filtering algorithms
• Implemented TF-IDF vectorization and Cosine Similarity for destination matching;
  applied SVD matrix factorization (Netflix Prize technique) for user-based filtering
• Created interactive Streamlit web app with preference filters, search, and
  real-time recommendations updating without page reload
• Dataset: 100 destinations, 200 users, 1,000 reviews across 15 Indian states
• Evaluated model using Precision@K, Coverage, and Diversity metrics
• Deployed live on Streamlit Cloud | GitHub: [your-link]
```

---

## 💼 LinkedIn Description

```
🗺️ Excited to share my latest project: Intelligent Travel Recommendation System!

Built an AI-powered travel app that recommends Indian tourist destinations
using the same ML techniques Netflix uses for movies.

🔧 Tech Stack: Python · Scikit-learn · Streamlit · Pandas · NumPy · SciPy

🤖 ML Techniques:
✅ Content-Based Filtering (TF-IDF + Cosine Similarity)
✅ Collaborative Filtering (SVD Matrix Factorization)
✅ Hybrid Recommendation Engine
✅ Model Evaluation (Precision@K, Coverage, Diversity)

📊 100 destinations · 15 states · 1,000 reviews
🚀 Live demo: [your-streamlit-link]
💻 GitHub: [your-github-link]

#MachineLearning #Python #Streamlit #RecommendationSystem #DataScience #AI
```

---

## 🎓 Viva/Interview Q&A

**Q: What is a recommendation system?**
A: A system that predicts user preferences and suggests relevant items. Examples: Netflix (movies), Amazon (products), Spotify (songs), our app (travel destinations).

**Q: Difference between Content-Based and Collaborative Filtering?**
A: Content-based uses item features to find similar items. Collaborative uses user behavior to find similar users, then recommends what they liked. Our hybrid combines both.

**Q: What is the Cold Start Problem?**
A: New users have no history → collaborative filtering fails. Solution: Use content-based filtering initially, switch to hybrid as history builds up.

**Q: What is TF-IDF?**
A: Term Frequency × Inverse Document Frequency. Converts text to numbers, giving high scores to unique meaningful words and low scores to common words like "the", "is".

**Q: Why Cosine Similarity over Euclidean distance?**
A: Cosine ignores vector length (text volume) and only measures direction (topic similarity). Better for text data where document length varies.

**Q: What is SVD?**
A: Singular Value Decomposition — decomposes the user-item matrix into latent factor matrices. Reveals hidden preferences like "adventure lover" or "budget traveler". Won the Netflix Prize!

**Q: How do you evaluate your recommendation system?**
A: Precision@K (relevance accuracy), Coverage (catalogue coverage), Diversity (variety), Average Rating (quality). All implemented in `src/evaluator.py`.

---

## 🔮 Future Enhancements

- [ ] **Weather API** — Show current weather at recommended destinations
- [ ] **Google Maps Integration** — Interactive map with destination pins
- [ ] **Hotel Recommendations** — Link with hotel booking APIs
- [ ] **AI Chatbot** — "Plan my 5-day Rajasthan trip" using Gemini API
- [ ] **Personalized Itinerary** — Day-by-day travel plan generator
- [ ] **Real Kaggle Data** — Plug in actual Kaggle dataset via kaggle API
- [ ] **User Authentication** — Save preferences across sessions
- [ ] **Sentiment Analysis** — Analyze review text for deeper insights
- [ ] **Deep Learning** — Neural Collaborative Filtering (NCF)
- [ ] **A/B Testing** — Compare recommendation algorithms live

---

## 🛠️ Tech Stack Explained

| Library | Why We Use It |
|---|---|
| **Pandas** | Load, clean, and manipulate tabular data (CSV files) |
| **NumPy** | Fast numerical operations and matrix math |
| **Scikit-learn** | TF-IDF, Cosine Similarity, MinMaxScaler, LabelEncoder |
| **SciPy** | Sparse SVD for collaborative filtering (memory efficient) |
| **Streamlit** | Build interactive web UI with pure Python — no HTML needed! |
| **Matplotlib** | Create charts and plots for data visualization |
| **Seaborn** | Beautiful statistical visualizations (built on matplotlib) |

---

## 📜 License

MIT License — Free to use for educational and personal projects.

---

*Built with ❤️ for learning ML · India Tourism Data · Powered by Streamlit*
