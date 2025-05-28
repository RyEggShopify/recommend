# quick_example.py
from recommendation_swiper import RecommendationSwiper, create_sample_data

# Initialize the system
swiper = RecommendationSwiper()

# Add some sample data
sample_items = create_sample_data()
swiper.add_items(sample_items)

# Get recommendations
recommendations = swiper.get_recommendations(n_results=3)
print("Initial recommendations:")
for item in recommendations:
    print(f"- {item['title']}: {item['description']}")

# Simulate some swipes
swiper.swipe("2", "like")  # Like Inception
swiper.swipe("8", "like")  # Like Blade Runner 2049
swiper.swipe("1", "dislike")  # Dislike The Great Gatsby

# Get new recommendations (should be more sci-fi focused)
new_recommendations = swiper.get_recommendations(n_results=3)
print("\nRecommendations after learning your preferences:")
for item in new_recommendations:
    print(f"- {item['title']}: {item['description']}")

# Check stats
stats = swiper.get_stats()
print(f"\nStats: {stats}")