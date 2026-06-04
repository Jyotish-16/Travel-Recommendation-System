# ============================================================
# FILE: src/data_generator.py
# PURPOSE: Generate realistic data matching the Kaggle dataset schema:
#   https://www.kaggle.com/datasets/amanmehra23/travel-recommendation-dataset
#
# The Kaggle dataset has 4 CSV files:
#   1. destinations.csv  - Indian tourist places
#   2. users.csv         - User profiles & preferences
#   3. reviews.csv       - Ratings & written reviews
#   4. user_history.csv  - Past travel history
#
# WHY WE GENERATE DATA:
#   The Kaggle API requires authentication tokens. To make this project
#   work for EVERYONE without setup hassles, we generate realistic data
#   that has the exact same structure as the real Kaggle dataset.
#   Once you get the real data, just replace the files in data/raw/
# ============================================================

import pandas as pd      # For creating DataFrames (like Excel tables in Python)
import numpy as np       # For random number generation
import os                # For creating folders and checking file paths
import random            # For picking random items from lists

# Set a "seed" so we get the same random data every time we run this script
# This makes results reproducible (same output each time)
np.random.seed(42)
random.seed(42)


# ============================================================
# DESTINATION DATA
# These are real Indian tourist places with realistic attributes
# ============================================================

DESTINATIONS_DATA = {
    'destination_id': list(range(1, 101)),
    'name': [
        # North India
        'Taj Mahal', 'Qutub Minar', 'India Gate', 'Red Fort', 'Lotus Temple',
        'Jama Masjid', 'Humayun Tomb', 'Akshardham Temple', 'Lal Qila', 'Chandni Chowk',
        # Rajasthan
        'Amber Fort', 'City Palace Jaipur', 'Hawa Mahal', 'Mehrangarh Fort', 'Jaisalmer Fort',
        'Lake Palace Udaipur', 'Chittorgarh Fort', 'Ranthambore Tiger Reserve', 'Pushkar Lake', 'Jantar Mantar',
        # South India
        'Hampi', 'Mysore Palace', 'Brihadeeswarar Temple', 'Meenakshi Amman Temple', 'Shore Temple Mahabalipuram',
        'Coorg', 'Ooty', 'Munnar', 'Kodaikanal', 'Wayanad',
        # Kerala
        'Alleppey Backwaters', 'Kovalam Beach', 'Periyar Wildlife Sanctuary', 'Varkala Beach', 'Thekkady',
        'Fort Kochi', 'Athirapally Falls', 'Kalpetta', 'Bekal Fort', 'Marari Beach',
        # Goa
        'Calangute Beach', 'Baga Beach', 'Anjuna Beach', 'Palolem Beach', 'Basilica of Bom Jesus',
        'Dudhsagar Falls', 'Chapora Fort', 'Colva Beach', 'Candolim Beach', 'Vagator Beach',
        # Himachal Pradesh
        'Shimla', 'Manali', 'Dharamsala', 'Rohtang Pass', 'Solang Valley',
        'Kasol', 'Spiti Valley', 'Kullu', 'Dalhousie', 'Khajjiar',
        # Uttarakhand
        'Rishikesh', 'Haridwar', 'Mussoorie', 'Nainital', 'Jim Corbett National Park',
        'Auli', 'Valley of Flowers', 'Kedarnath Temple', 'Badrinath Temple', 'Chopta',
        # Mumbai & Maharashtra
        'Gateway of India', 'Marine Drive Mumbai', 'Elephanta Caves', 'Ajanta Caves', 'Ellora Caves',
        'Lonavala', 'Mahabaleshwar', 'Aurangabad', 'Shirdi', 'Pune',
        # Northeast India
        'Kaziranga National Park', 'Cherrapunji', 'Shillong', 'Tawang Monastery', 'Ziro Valley',
        'Majuli Island', 'Dzukou Valley', 'Loktak Lake', 'Kamakhya Temple', 'Haflong',
        # Andaman & Other
        'Radhanagar Beach Andaman', 'Havelock Island', 'Cellular Jail Andaman', 'Ross Island', 'Neil Island',
        'Lakshadweep', 'Pondicherry', 'Mahabalipuram', 'Rameshwaram', 'Kanyakumari'
    ],
    'type': [
        'Historical', 'Historical', 'Monument', 'Historical', 'Spiritual',
        'Spiritual', 'Historical', 'Spiritual', 'Historical', 'Cultural',
        'Historical', 'Historical', 'Historical', 'Historical', 'Historical',
        'Heritage', 'Historical', 'Wildlife', 'Spiritual', 'Historical',
        'Heritage', 'Historical', 'Spiritual', 'Spiritual', 'Historical',
        'Nature', 'Nature', 'Nature', 'Nature', 'Nature',
        'Nature', 'Beach', 'Wildlife', 'Beach', 'Wildlife',
        'Heritage', 'Nature', 'Nature', 'Historical', 'Beach',
        'Beach', 'Beach', 'Beach', 'Beach', 'Spiritual',
        'Nature', 'Historical', 'Beach', 'Beach', 'Beach',
        'Hill Station', 'Hill Station', 'Cultural', 'Adventure', 'Adventure',
        'Nature', 'Adventure', 'Hill Station', 'Hill Station', 'Nature',
        'Spiritual', 'Spiritual', 'Hill Station', 'Hill Station', 'Wildlife',
        'Adventure', 'Nature', 'Spiritual', 'Spiritual', 'Nature',
        'Historical', 'Beach', 'Heritage', 'Heritage', 'Heritage',
        'Nature', 'Hill Station', 'Historical', 'Spiritual', 'Cultural',
        'Wildlife', 'Nature', 'Hill Station', 'Spiritual', 'Nature',
        'Nature', 'Nature', 'Nature', 'Spiritual', 'Nature',
        'Beach', 'Beach', 'Historical', 'Historical', 'Beach',
        'Beach', 'Cultural', 'Historical', 'Spiritual', 'Nature'
    ],
    'state': [
        'Uttar Pradesh', 'Delhi', 'Delhi', 'Delhi', 'Delhi',
        'Delhi', 'Delhi', 'Delhi', 'Delhi', 'Delhi',
        'Rajasthan', 'Rajasthan', 'Rajasthan', 'Rajasthan', 'Rajasthan',
        'Rajasthan', 'Rajasthan', 'Rajasthan', 'Rajasthan', 'Rajasthan',
        'Karnataka', 'Karnataka', 'Tamil Nadu', 'Tamil Nadu', 'Tamil Nadu',
        'Karnataka', 'Tamil Nadu', 'Kerala', 'Tamil Nadu', 'Kerala',
        'Kerala', 'Kerala', 'Kerala', 'Kerala', 'Kerala',
        'Kerala', 'Kerala', 'Kerala', 'Kerala', 'Kerala',
        'Goa', 'Goa', 'Goa', 'Goa', 'Goa',
        'Goa', 'Goa', 'Goa', 'Goa', 'Goa',
        'Himachal Pradesh', 'Himachal Pradesh', 'Himachal Pradesh', 'Himachal Pradesh', 'Himachal Pradesh',
        'Himachal Pradesh', 'Himachal Pradesh', 'Himachal Pradesh', 'Himachal Pradesh', 'Himachal Pradesh',
        'Uttarakhand', 'Uttarakhand', 'Uttarakhand', 'Uttarakhand', 'Uttarakhand',
        'Uttarakhand', 'Uttarakhand', 'Uttarakhand', 'Uttarakhand', 'Uttarakhand',
        'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra',
        'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra',
        'Assam', 'Meghalaya', 'Meghalaya', 'Arunachal Pradesh', 'Arunachal Pradesh',
        'Assam', 'Nagaland', 'Manipur', 'Assam', 'Assam',
        'Andaman & Nicobar', 'Andaman & Nicobar', 'Andaman & Nicobar', 'Andaman & Nicobar', 'Andaman & Nicobar',
        'Lakshadweep', 'Puducherry', 'Tamil Nadu', 'Tamil Nadu', 'Tamil Nadu'
    ],
    'popularity': np.random.randint(60, 100, 100).tolist(),
    'best_time_to_visit': [
        'Oct-Mar', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar',
        'Oct-Mar', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar',
        'Oct-Mar', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar',
        'Oct-Mar', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar',
        'Oct-Feb', 'Oct-Feb', 'Nov-Mar', 'Nov-Mar', 'Nov-Mar',
        'Sep-May', 'Apr-Jun', 'Sep-May', 'Apr-Jun', 'Sep-May',
        'Sep-Mar', 'Sep-Mar', 'Sep-Mar', 'Sep-Mar', 'Sep-Mar',
        'Oct-Feb', 'Jun-Sep', 'Sep-Mar', 'Oct-Feb', 'Sep-Mar',
        'Nov-Feb', 'Nov-Feb', 'Nov-Feb', 'Nov-Feb', 'Nov-Feb',
        'Jun-Sep', 'Nov-Feb', 'Nov-Feb', 'Nov-Feb', 'Nov-Feb',
        'Mar-Jun', 'Mar-Jun', 'Mar-Jun', 'May-Jun', 'Dec-Feb',
        'Mar-May', 'Apr-Jun', 'Mar-Jun', 'Mar-Jun', 'Mar-Jun',
        'Sep-Jun', 'Sep-Jun', 'Mar-Jun', 'Mar-Jun', 'Nov-Jun',
        'Dec-Mar', 'Jul-Aug', 'May-Jun', 'May-Jun', 'Mar-Jun',
        'Oct-Mar', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar',
        'Jun-Sep', 'Apr-Jun', 'Oct-Mar', 'Oct-Mar', 'Oct-Mar',
        'Nov-Apr', 'Oct-Jun', 'Oct-Jun', 'Apr-Oct', 'Oct-May',
        'Oct-Apr', 'Oct-May', 'Oct-May', 'Oct-Mar', 'Oct-May',
        'Dec-May', 'Dec-May', 'Dec-May', 'Dec-May', 'Dec-May',
        'Oct-May', 'Nov-Mar', 'Nov-Mar', 'Nov-Mar', 'Oct-Mar'
    ],
    'entry_fee': [
        50, 30, 0, 35, 0, 0, 30, 70, 35, 0,
        200, 100, 50, 100, 50, 250, 100, 200, 0, 40,
        40, 200, 50, 50, 40, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 30, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1000, 0, 0, 0, 0, 0,
        0, 0, 15, 40, 40, 0, 0, 0, 0, 0,
        50, 25, 400, 40, 0, 0, 0, 50, 0, 0,
        500, 100, 30, 30, 200, 0, 0, 40, 0, 0
    ],
    'average_rating': np.round(np.random.uniform(3.5, 5.0, 100), 1).tolist(),
    'total_reviews': np.random.randint(50, 5000, 100).tolist(),
    'latitude': [
        27.17, 28.52, 28.61, 28.65, 28.55, 28.65, 28.59, 28.61, 28.65, 28.65,
        26.98, 26.92, 26.92, 26.29, 26.91, 24.57, 24.88, 25.98, 26.48, 26.92,
        15.33, 12.30, 10.78, 9.91, 12.62, 12.33, 11.41, 10.08, 10.23, 11.68,
        9.49, 8.40, 9.46, 8.73, 9.60, 9.96, 10.27, 11.56, 12.38, 9.59,
        15.52, 15.55, 15.58, 15.01, 15.50, 15.31, 15.62, 15.27, 15.52, 15.59,
        31.10, 32.24, 32.21, 32.37, 32.32, 31.99, 32.08, 31.95, 32.53, 32.04,
        30.08, 29.94, 30.45, 29.37, 29.53, 30.52, 30.73, 30.73, 30.73, 30.64,
        18.92, 18.94, 18.47, 20.55, 19.92, 18.75, 18.11, 19.88, 19.62, 18.52,
        26.63, 25.28, 25.56, 27.10, 27.47, 26.74, 25.95, 24.78, 26.18, 25.06,
        11.62, 11.70, 11.67, 11.72, 11.84, 10.56, 11.93, 12.62, 9.28, 8.08
    ],
    'longitude': [
        78.04, 77.18, 77.23, 77.24, 77.26, 77.23, 77.25, 77.28, 77.24, 77.23,
        75.85, 75.82, 75.82, 73.01, 70.91, 73.68, 74.64, 76.50, 74.55, 75.82,
        76.46, 76.65, 79.13, 78.12, 80.19, 75.73, 76.69, 77.06, 77.47, 76.08,
        76.33, 76.98, 77.31, 76.86, 77.16, 76.24, 76.57, 76.55, 75.49, 76.55,
        73.75, 73.83, 73.74, 74.02, 73.92, 74.31, 73.86, 73.92, 73.87, 73.86,
        77.17, 77.18, 76.31, 77.15, 77.13, 77.11, 77.11, 77.10, 75.97, 77.12,
        78.26, 78.16, 78.07, 78.98, 79.21, 79.57, 79.60, 79.46, 79.45, 79.21,
        72.83, 72.82, 72.93, 75.75, 75.34, 73.40, 73.67, 75.32, 74.97, 73.85,
        93.20, 91.68, 91.88, 91.86, 93.83, 94.24, 94.56, 93.77, 91.72, 93.02,
        92.72, 92.99, 92.68, 92.75, 93.05, 72.64, 79.83, 80.19, 79.31, 77.55
    ],
    'budget_level': [
        'Low', 'Low', 'Free', 'Low', 'Free', 'Free', 'Low', 'Medium', 'Low', 'Free',
        'Medium', 'Medium', 'Low', 'Medium', 'Low', 'High', 'Medium', 'Medium', 'Low', 'Low',
        'Low', 'Medium', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low',
        'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low',
        'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low',
        'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Low', 'Low', 'Medium', 'Medium', 'Low',
        'Low', 'Low', 'Low', 'Low', 'High', 'Medium', 'Low', 'Low', 'Low', 'Low',
        'Low', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'Medium', 'Low', 'Low', 'Low',
        'Medium', 'Low', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Low',
        'High', 'High', 'Low', 'Low', 'Medium', 'High', 'Low', 'Low', 'Low', 'Low'
    ],
    'description': [
        'One of the Seven Wonders of the World, a symbol of eternal love built by Mughal Emperor Shah Jahan for his wife Mumtaz Mahal.',
        'A UNESCO World Heritage Site, this 73-metre tall minaret is the tallest brick minaret in the world.',
        'An iconic war memorial built in the memory of Indian soldiers who died in World War I.',
        'A UNESCO World Heritage Site, this massive red sandstone fort served as the main residence of the Mughal emperors.',
        'A magnificent lotus-shaped temple that welcomes people of all religions for prayer and meditation.',
        'One of the largest mosques in India, capable of holding 25,000 worshippers at a time.',
        "A magnificent mid-16th century Mughal mausoleum in Delhi, built for the Mughal Emperor Humayun.",
        'A stunning Hindu temple complex and cultural campus on the banks of the Yamuna River.',
        'Also known as Red Fort, a historic fort that served as the main residence of Mughal emperors for nearly 200 years.',
        'A historic street in Old Delhi famous for its food, street markets, and vibrant culture.',
        'A stunning hilltop fort overlooking Maota Lake, a blend of Hindu and Mughal architecture.',
        'A magnificent palace complex in Jaipur, home to the royal family of Jaipur.',
        "The Palace of Winds, a stunning five-story palace with 953 small windows (jharokhas) decorated with intricate latticework.",
        'One of the most magnificent and well-preserved forts in India, located in the blue city of Jodhpur.',
        'A desert fort that rises from the sands of the Thar Desert, also known as the Golden Fort.',
        'A stunning palace hotel built in the middle of Lake Pichola in Udaipur, the City of Lakes.',
        'A magnificent 7th-century fort complex in Rajasthan, considered the pride of Rajputs.',
        'One of the finest wildlife reserves in India, known for Bengal tigers.',
        'A sacred city built around a lake, famous for the Brahma Temple and annual Pushkar Camel Fair.',
        'A UNESCO World Heritage Site, an astronomical instrument collection built in the 18th century.',
        "An ancient city that was once the capital of the Vijayanagara Empire, a UNESCO World Heritage Site.",
        'A magnificent palace in Mysore, one of the largest palaces in India, illuminated by 100,000 bulbs.',
        'A UNESCO World Heritage Site and one of the greatest temples of India, dedicated to Lord Shiva.',
        'A magnificent temple dedicated to Goddess Meenakshi, known for its stunning colorful gopurams.',
        'A group of ancient temples built by the Pallava kings, overlooking the Bay of Bengal.',
        'A hill station in Karnataka known for its coffee plantations, misty valleys, and waterfalls.',
        'A picturesque hill station in Tamil Nadu, known as the Queen of Hill Stations.',
        'A breathtakingly beautiful hill station in Kerala, famous for its lush tea plantations.',
        'A charming hill station in Tamil Nadu, the Princess of Hill Stations, surrounded by forests.',
        'A pristine wildlife sanctuary in Kerala with dense forests, wildlife, and tribal communities.',
        'Known as the Venice of the East, famous for its serene backwater network and houseboat experiences.',
        'A beautiful beach town in Kerala with stunning shores and a vibrant fishing community.',
        'A wildlife sanctuary centered around Periyar Lake, home to elephants and tigers.',
        'A cliff-side beach town famous for its laterite cliffs, natural springs, and peaceful atmosphere.',
        'A beautiful wildlife sanctuary in Kerala, famous for elephants, spice plantations, and boating.',
        'A historic port city with colonial architecture, Chinese fishing nets, and vibrant culture.',
        "One of India's most spectacular waterfalls, surrounded by dense forests and wildlife.",
        'A tranquil hill town in Wayanad, surrounded by misty mountains and coffee plantations.',
        'A beautiful beach fort in Kerala, built by Portuguese traders in the 16th century.',
        'A pristine beach in Kerala, known for its untouched beauty and coconut palms.',
        "India's most popular tourist beach, known for its vibrant atmosphere and water sports.",
        'A lively beach in Goa, famous for its nightlife, restaurants, and water sports.',
        'A bohemian beach in North Goa, famous for its flea markets and hippie culture.',
        'One of the most beautiful beaches in Goa, a crescent-shaped bay with pristine sands.',
        'A UNESCO World Heritage Site, the oldest church in India with the mortal remains of St. Francis Xavier.',
        "One of Goa's most spectacular waterfalls, surrounded by spice plantations.",
        'A scenic clifftop fort in North Goa with stunning views of the Arabian Sea.',
        'A peaceful beach in South Goa with a relaxed atmosphere and scenic views.',
        'A clean, less-crowded beach in North Goa with beautiful white sands.',
        'A dramatic beach with towering red and black volcanic cliffs in North Goa.',
        "The summer capital of British India, a charming hill station in Himachal Pradesh.",
        'A popular hill station in the Kullu Valley, gateway to the high Himalayas.',
        'Home of the Dalai Lama and the Tibetan government in exile, a spiritual and cultural hub.',
        'A high mountain pass in the Himalayas, offering stunning views of snowcapped peaks.',
        'A beautiful valley with adventure activities like skiing, zorbing, and paragliding.',
        'A picturesque village on the banks of the Parvati River, a paradise for trekkers.',
        'A remote and starkly beautiful valley in the Himalayas, one of the coldest places in India.',
        'A scenic valley town in Himachal Pradesh, adventure hub of the western Himalayas.',
        'A beautiful hill station in Himachal Pradesh, known as the Scotland of India.',
        'A small tourist destination often called the Mini Switzerland of India.',
        'The Yoga Capital of the World, famous for yoga, meditation, and Ganga Aarti.',
        'One of the holiest cities in India, on the banks of the Ganges, famous for Ganga Aarti.',
        'The Queen of Hills, a scenic hill station in Uttarakhand with toy train and colonial architecture.',
        'A beautiful lake city in the Kumaon Himalayas, surrounded by forests.',
        'The oldest national park in Asia, famous for Bengal tigers and elephant safaris.',
        'A ski resort in Uttarakhand with breathtaking views of the Garhwal Himalayas.',
        'A UNESCO World Heritage Site, a valley of hundreds of wild flower species in the Himalayas.',
        'A sacred Shiva temple in the Garhwal Himalayas, one of the Char Dham pilgrimage sites.',
        'One of the holiest temples in Hinduism, dedicated to Lord Vishnu, in the Garhwal Himalayas.',
        'A beautiful meadow in Uttarakhand, a perfect trek for beginners.',
        'A massive arch built in 1924 by the British to commemorate the landing of King George V.',
        'A 3.6 km long boulevard along the coast of Mumbai, the most beautiful promenade in India.',
        'A group of cave temples on Elephanta Island near Mumbai, dedicated to Lord Shiva.',
        'A UNESCO World Heritage Site with 30 rock-cut Buddhist caves from the 2nd century BCE.',
        'A UNESCO World Heritage Site with 34 cave temples dedicated to Hindu, Buddhist, and Jain traditions.',
        'A popular hill station near Mumbai, famous for its scenic landscapes and waterfalls.',
        'A hill station in the Sahyadri range, known as the Strawberry Capital of India.',
        'A city known as the gateway to the Ajanta and Ellora caves.',
        'A holy town in Maharashtra, home to the Sai Baba temple.',
        'The Oxford of the East, a vibrant city with numerous educational institutions and IT companies.',
        'A UNESCO World Heritage Site, home to the Indian one-horned rhinoceros.',
        "The wettest place on Earth, famous for its living root bridges and meghalayan caves.",
        'The capital of Meghalaya, a beautiful hill city with colonial architecture.',
        'A Buddhist monastery near the China border, the largest monastery in the country.',
        'A remote valley in Arunachal Pradesh, home to the Apatani tribe.',
        'The largest river island in the world, known for its Vaishnavite monasteries and culture.',
        'A high-altitude valley in Nagaland with stunning landscapes and rich biodiversity.',
        'A freshwater lake in Manipur, home to floating islands and a unique ecosystem.',
        'A famous Hindu temple dedicated to Goddess Kamakhya, a major pilgrimage site in Assam.',
        'A scenic hill station in Assam, surrounded by dense forests and picturesque lakes.',
        'Rated as one of the best beaches in Asia, with crystal clear turquoise waters.',
        'The most popular island in the Andaman archipelago, with stunning beaches and marine life.',
        'A historical prison built during British rule, a National Memorial Monument.',
        'The former capital of the Andaman Islands, with ruins of British buildings.',
        'A serene island known for its beautiful beaches and coral reefs.',
        'A group of coral islands in the Arabian Sea, known for crystal clear waters and marine biodiversity.',
        "A French colonial city on the southeastern coast of India, known as the 'French Riviera of the East'.",
        'An ancient coastal town with seven 7th-century rock-cut temples on the beach.',
        'A sacred pilgrimage site where Adam Bridge connecting India to Sri Lanka begins.',
        'The southernmost tip of India where three seas meet - Arabian Sea, Bay of Bengal, and Indian Ocean.'
    ]
}


def generate_users(n=200):
    """
    Generate user profile data.

    WHY WE NEED USERS:
    A recommendation system needs USER data to understand preferences.
    Just like Netflix asks "What genres do you like?" when you sign up,
    we create user profiles with travel preferences.
    """

    genders = ['Male', 'Female', 'Non-binary']
    preference_types = [
        'Historical', 'Beach', 'Wildlife', 'Adventure', 'Spiritual',
        'Nature', 'Hill Station', 'Cultural', 'Heritage'
    ]
    budget_preferences = ['Budget', 'Mid-range', 'Luxury']
    travel_styles = ['Solo', 'Family', 'Couple', 'Group']
    age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']

    users = {
        'user_id': list(range(1, n + 1)),
        'age_group': [random.choice(age_groups) for _ in range(n)],
        'gender': [random.choice(genders) for _ in range(n)],
        'preferred_type': [random.choice(preference_types) for _ in range(n)],
        'budget_preference': [random.choice(budget_preferences) for _ in range(n)],
        'travel_style': [random.choice(travel_styles) for _ in range(n)],
        'num_adults': np.random.randint(1, 5, n).tolist(),
        'num_children': np.random.randint(0, 4, n).tolist(),
        'preferred_states': [
            random.sample(['Delhi', 'Rajasthan', 'Kerala', 'Goa', 'Himachal Pradesh',
                           'Uttarakhand', 'Karnataka', 'Tamil Nadu', 'Maharashtra'], 2)
            for _ in range(n)
        ],
    }
    return pd.DataFrame(users)


def generate_reviews(n=1000):
    """
    Generate review data linking users to destinations.

    WHY REVIEWS MATTER IN ML:
    Reviews are the backbone of COLLABORATIVE FILTERING.
    The idea: "If you and I both loved the Taj Mahal and Manali,
    and I also loved Coorg, then you'll probably love Coorg too!"
    This is exactly how Netflix recommends movies based on
    what similar users watched.
    """

    review_texts = [
        "Absolutely stunning! A must-visit for everyone.",
        "Breathtaking views and wonderful experience.",
        "One of the best places I've ever visited.",
        "Great place but can get very crowded during peak season.",
        "Beautiful architecture and peaceful atmosphere.",
        "Amazing wildlife experience, saw tigers up close!",
        "Perfect beach destination, crystal clear waters.",
        "A spiritual experience like no other.",
        "The food and culture here are amazing.",
        "Overrated but still worth a visit.",
        "Perfect for family vacations, kids loved it!",
        "A bit expensive but totally worth it.",
        "Best place for adventure seekers!",
        "The natural beauty here is unparalleled.",
        "Historical significance makes this a very meaningful visit.",
        "Wonderful hospitality and clean surroundings.",
        "A hidden gem that not many people know about.",
        "The sunset view here is absolutely magical.",
        "Great for photography enthusiasts.",
        "Plan at least 2-3 days to fully explore this place."
    ]

    reviews = {
        'review_id': list(range(1, n + 1)),
        'user_id': np.random.randint(1, 201, n).tolist(),
        'destination_id': np.random.randint(1, 101, n).tolist(),
        'rating': np.round(np.random.uniform(2.5, 5.0, n), 1).tolist(),
        'review_text': [random.choice(review_texts) for _ in range(n)],
        'visit_month': np.random.randint(1, 13, n).tolist(),
        'visit_year': np.random.randint(2020, 2025, n).tolist(),
        'helpful_votes': np.random.randint(0, 150, n).tolist(),
        'travel_type': [random.choice(['Solo', 'Family', 'Couple', 'Group']) for _ in range(n)]
    }
    return pd.DataFrame(reviews)


def generate_user_history(n=500):
    """
    Generate historical travel data for users.

    WHY HISTORY DATA IS IMPORTANT:
    Past behavior predicts future preferences!
    If a user has visited 5 beach destinations in the past,
    they likely enjoy beaches — so we recommend more beach places.
    This is called BEHAVIORAL FILTERING.
    """

    history = {
        'history_id': list(range(1, n + 1)),
        'user_id': np.random.randint(1, 201, n).tolist(),
        'destination_id': np.random.randint(1, 101, n).tolist(),
        'visit_year': np.random.randint(2018, 2025, n).tolist(),
        'visit_month': np.random.randint(1, 13, n).tolist(),
        'duration_days': np.random.randint(1, 15, n).tolist(),
        'total_spent': np.random.randint(1000, 50000, n).tolist(),
        'satisfaction_score': np.round(np.random.uniform(3.0, 5.0, n), 1).tolist(),
        'would_revisit': [random.choice(['Yes', 'No', 'Maybe']) for _ in range(n)]
    }
    return pd.DataFrame(history)


def create_dataset(output_dir='data/raw'):
    """
    Main function to create and save all 4 dataset CSV files.
    This replicates the exact structure of the Kaggle dataset.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Creating India Travel Recommendation Dataset")
    print("=" * 60)

    # 1. Create and save Destinations data
    print("\n[1/4] Generating destinations.csv...")
    destinations_df = pd.DataFrame(DESTINATIONS_DATA)
    destinations_path = os.path.join(output_dir, 'destinations.csv')
    destinations_df.to_csv(destinations_path, index=False)
    print(f"      ✓ Saved {len(destinations_df)} destinations to {destinations_path}")

    # 2. Create and save Users data
    print("\n[2/4] Generating users.csv...")
    users_df = generate_users(200)
    users_path = os.path.join(output_dir, 'users.csv')
    users_df.to_csv(users_path, index=False)
    print(f"      ✓ Saved {len(users_df)} user profiles to {users_path}")

    # 3. Create and save Reviews data
    print("\n[3/4] Generating reviews.csv...")
    reviews_df = generate_reviews(1000)
    reviews_path = os.path.join(output_dir, 'reviews.csv')
    reviews_df.to_csv(reviews_path, index=False)
    print(f"      ✓ Saved {len(reviews_df)} reviews to {reviews_path}")

    # 4. Create and save User History data
    print("\n[4/4] Generating user_history.csv...")
    history_df = generate_user_history(500)
    history_path = os.path.join(output_dir, 'user_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"      ✓ Saved {len(history_df)} history records to {history_path}")

    print("\n" + "=" * 60)
    print("  ✅ Dataset created successfully!")
    print(f"  📁 Location: {os.path.abspath(output_dir)}")
    print("=" * 60)

    return destinations_df, users_df, reviews_df, history_df


# Run dataset creation when this script is executed directly
if __name__ == '__main__':
    create_dataset()
