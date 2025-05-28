

# main.py
from recommendation_swiper import RecommendationSwiper
from bigquery_client import ShopifyBigQueryClient

class SimpleSwiper:
    def __init__(self):
        self.swiper = RecommendationSwiper()
        self.bigquery_client = ShopifyBigQueryClient()
        self.current_recommendations = []
        self.setup_data()
    
    def setup_data(self):
        """Initialize with data"""
        print("🚀 Loading data...")
        
        # Check if we have items
        print("📊 Loading from BigQuery...")
        sample_items = self.bigquery_client.query_products()
        print(f"Loaded {len(sample_items)} items from BigQuery")
        sample_items = sample_items.to_dict(orient='records')
        self.swiper.add_items(sample_items)
        
        self.swiper.load_session()
        self.update_display()
    
    def update_display(self):
        """Show current top 3 recommendations"""
        self.current_recommendations = self.swiper.get_recommendations(n_results=3)
        
        # Clear screen (works on most terminals)
        print("\n" * 50)
        
        print("="*60)
        print("🔥 TOP 3 RECOMMENDATIONS:")
        print("="*60)
        
        for i, item in enumerate(self.current_recommendations, 1):
            marker = "👉 " if i == 1 else "   "
            print(f"{marker}{i}. {item['title']}")
            if item.get('description'):
                # Clean up HTML tags from description
                desc = item['description'].replace('<', '&lt;').replace('>', '&gt;')
                desc = desc[:80] + "..." if len(desc) > 80 else desc
                print(f"      📝 {desc}")
            print()
        
        if self.current_recommendations:
            print(f"🎯 CURRENT ITEM: {self.current_recommendations[0]['title']}")
        
        stats = self.swiper.get_stats()
        print(f"📊 STATS: {stats['likes']} likes, {stats['dislikes']} dislikes, {stats['total_swipes']} total")
        print("="*60)
    
    def handle_action(self, action):
        """Handle like/dislike action"""
        if not self.current_recommendations:
            print("No more recommendations!")
            return
            
        item = self.current_recommendations[0]
        self.swiper.swipe(item['id'], action)
        
        if action == 'like':
            print(f"✅ LIKED: {item['title']}")
        else:
            print(f"❌ DISLIKED: {item['title']}")
        
        # Update display immediately
        self.update_display()
    
    def run(self):
        """Run the swiper"""
        print("🎉 Simple Recommendation Swiper")
        print("Commands: 'a' = dislike, 'd' = like, 'q' = quit, 's' = save")
        
        try:
            while True:
                if not self.current_recommendations:
                    print("🎊 No more recommendations available!")
                    break
                
                command = input("\nEnter command (a/d/q/s): ").lower().strip()
                
                if command == 'q':
                    break
                elif command == 'a':
                    self.handle_action('dislike')
                elif command == 'd':
                    self.handle_action('like')
                elif command == 's':
                    self.swiper.save_session()
                    print("💾 Session saved!")
                elif command == '':
                    continue
                else:
                    print("Invalid command! Use 'a' (dislike), 'd' (like), 'q' (quit), or 's' (save)")
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.swiper.save_session()
            print("\n👋 Session saved! Goodbye!")

def main():
    app = SimpleSwiper()
    app.run()

if __name__ == "__main__":
    main()