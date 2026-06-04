# ============================================================
# FILE: setup.py
# PURPOSE: One-click project setup script
#
# HOW TO RUN:
#   python setup.py
#
# WHAT IT DOES:
#   1. Checks if required packages are installed
#   2. Generates the dataset (matching Kaggle schema)
#   3. Runs a quick test of the recommendation engine
#   4. Tells you how to launch the app
# ============================================================

import subprocess
import sys
import os


def check_and_install_packages():
    """Check if all required packages are installed."""
    print("📦 Checking required packages...")

    required = [
        'pandas', 'numpy', 'sklearn', 'streamlit',
        'matplotlib', 'seaborn', 'scipy', 'PIL'
    ]
    import_names = {
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow'
    }

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"   ✓ {import_names.get(pkg, pkg)}")
        except ImportError:
            pkg_name = import_names.get(pkg, pkg)
            missing.append(pkg_name)
            print(f"   ✗ {pkg_name} — NOT FOUND")

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Installing now...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install'] + missing
        )
        print("✅ All packages installed!")
    else:
        print("✅ All packages already installed!")


def generate_data():
    """Generate the dataset."""
    print("\n📊 Generating dataset...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.data_generator import create_dataset
    create_dataset('data/raw')


def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🧪 Running quick test...")

    from src.recommender import setup_recommender

    recommender, df = setup_recommender()

    # Test 1: Basic recommendation
    recs = recommender.recommend(preferred_type='Beach', n=3)
    assert len(recs) > 0, "Recommendation returned 0 results!"
    print(f"   ✓ Recommendation test passed ({len(recs)} results)")

    # Test 2: Similar places
    similar = recommender.recommend_similar('Taj Mahal', n=3)
    print(f"   ✓ Similarity test passed ({len(similar)} similar places found)")

    # Test 3: Search
    search_results = df[df['name'].str.lower().str.contains('goa', na=False)]
    print(f"   ✓ Search test passed ({len(search_results)} results for 'goa')")

    print("   ✅ All tests passed!")
    return True


def main():
    print("=" * 60)
    print("  🗺️  INTELLIGENT TRAVEL RECOMMENDATION SYSTEM")
    print("  Setup & Initialization Script")
    print("=" * 60)

    # Step 1: Check packages
    check_and_install_packages()

    # Step 2: Generate data
    generate_data()

    # Step 3: Quick test
    try:
        run_quick_test()
    except Exception as e:
        print(f"\n⚠️  Test warning: {e}")
        print("   App may still work — try launching it anyway.")

    # Done!
    print("\n" + "=" * 60)
    print("  ✅ SETUP COMPLETE!")
    print("=" * 60)
    print("""
  🚀 HOW TO LAUNCH THE APP:
     streamlit run app.py

  🌐 The app will open at: http://localhost:8501

  📁 PROJECT STRUCTURE:
     app.py              ← Main Streamlit application
     src/
       data_generator.py ← Dataset creation
       data_loader.py    ← Data loading & EDA
       preprocessor.py   ← Data cleaning & encoding
       feature_engineering.py ← TF-IDF & similarity
       recommender.py    ← ML recommendation engine
       evaluator.py      ← Model evaluation metrics
     data/
       raw/              ← CSV dataset files
       processed/        ← Cleaned data
     .streamlit/
       config.toml       ← App theme settings
""")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
