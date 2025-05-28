# recommendation_swiper.py
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
import os

@dataclass
class SwipeResult:
    item_id: str
    action: str  # 'like' or 'dislike'
    timestamp: float

class RecommendationSwiper:
    def __init__(self, db_path: str = "./vector_db"):
        """Initialize the recommendation system"""
        self.console = Console()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Collections
        self.items_collection = self._get_or_create_collection("items")
        self.user_preferences_collection = self._get_or_create_collection("user_preferences")
        
        # User interaction history
        self.swipe_history: List[SwipeResult] = []
        self.user_profile_vector = None
        
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(name)
    
    def add_items(self, items: List[Dict[str, str]]):
        """Add items to the database with their embeddings"""
        for item in items:
            item_id = item.get('id', str(uuid.uuid4()))
            text = f"{item.get('title', '')} {item.get('description', '')} {item.get('tags', '')}"
            
            # Generate embedding
            embedding = self.model.encode(text).tolist()
            
            # Store in ChromaDB
            self.items_collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    'id': item_id,
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'tags': item.get('tags', ''),
                    'category': item.get('category', '')
                }],
                ids=[item_id]
            )
        
        self.console.print(f"✅ Added {len(items)} items to the database")
    
    def get_recommendations(self, n_results: int = 10, exclude_seen: bool = True) -> List[Dict]:
        """Get recommendations based on current user preferences"""
        if self.user_profile_vector is not None:
            # Use user profile for similarity search
            results = self.items_collection.query(
                query_embeddings=[self.user_profile_vector.tolist()],
                n_results=n_results * 2  # Get more to filter out seen items
            )
        else:
            # Cold start: get random items
            all_items = self.items_collection.get()
            if not all_items['ids']:
                return []
            
            # Simulate random selection for cold start
            results = self.items_collection.query(
                query_embeddings=[self.model.encode("popular trending recommendation").tolist()],
                n_results=n_results * 2
            )
        
        recommendations = []
        seen_ids = {swipe.item_id for swipe in self.swipe_history}
        
        for i, item_id in enumerate(results['ids'][0]):
            if exclude_seen and item_id in seen_ids:
                continue
                
            if len(recommendations) >= n_results:
                break
                
            metadata = results['metadatas'][0][i]
            recommendations.append({
                'id': item_id,
                'title': metadata.get('title', ''),
                'description': metadata.get('description', ''),
                'tags': metadata.get('tags', ''),
                'category': metadata.get('category', ''),
                'similarity_score': results['distances'][0][i] if results['distances'] else 0
            })
        
        return recommendations
    
    def swipe(self, item_id: str, action: str):
        """Record a swipe action and update user preferences"""
        import time
        
        swipe_result = SwipeResult(
            item_id=item_id,
            action=action,
            timestamp=time.time()
        )
        self.swipe_history.append(swipe_result)
        
        # Update user profile based on the swipe
        self._update_user_profile()
        
        self.console.print(f"📱 Swiped {action} on item {item_id}")
    
    def _update_user_profile(self):
        """Update user profile vector based on swipe history"""
        if not self.swipe_history:
            return
        
        liked_embeddings = []
        disliked_embeddings = []
        
        # Get embeddings for liked and disliked items
        for swipe in self.swipe_history[-20:]:  # Consider recent 20 swipes
            try:
                # Get item embedding from database
                result = self.items_collection.get(ids=[swipe.item_id], include=['embeddings'])
                if result['embeddings']:
                    embedding = np.array(result['embeddings'][0])
                    
                    if swipe.action == 'like':
                        liked_embeddings.append(embedding)
                    else:
                        disliked_embeddings.append(embedding)
            except:
                continue
        
        # Calculate user profile vector
        if liked_embeddings:
            liked_centroid = np.mean(liked_embeddings, axis=0)
            
            if disliked_embeddings:
                disliked_centroid = np.mean(disliked_embeddings, axis=0)
                # Move towards liked items and away from disliked items
                self.user_profile_vector = liked_centroid - 0.3 * disliked_centroid
            else:
                self.user_profile_vector = liked_centroid
            
            # Normalize the vector
            self.user_profile_vector = self.user_profile_vector / np.linalg.norm(self.user_profile_vector)
    
    def get_stats(self) -> Dict:
        """Get user interaction statistics"""
        if not self.swipe_history:
            return {"total_swipes": 0, "likes": 0, "dislikes": 0, "like_rate": 0}
        
        likes = sum(1 for swipe in self.swipe_history if swipe.action == 'like')
        dislikes = len(self.swipe_history) - likes
        
        return {
            "total_swipes": len(self.swipe_history),
            "likes": likes,
            "dislikes": dislikes,
            "like_rate": likes / len(self.swipe_history) if self.swipe_history else 0
        }
    
    def save_session(self, filename: str = "swipe_session.json"):
        """Save the current session"""
        session_data = {
            "swipe_history": [
                {
                    "item_id": swipe.item_id,
                    "action": swipe.action,
                    "timestamp": swipe.timestamp
                }
                for swipe in self.swipe_history
            ],
            "user_profile_vector": self.user_profile_vector.tolist() if self.user_profile_vector is not None else None
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.console.print(f"💾 Session saved to {filename}")
    
    def load_session(self, filename: str = "swipe_session.json"):
        """Load a previous session"""
        if not os.path.exists(filename):
            self.console.print(f"❌ Session file {filename} not found")
            return
        
        with open(filename, 'r') as f:
            session_data = json.load(f)
        
        # Restore swipe history
        self.swipe_history = [
            SwipeResult(
                item_id=swipe["item_id"],
                action=swipe["action"],
                timestamp=swipe["timestamp"]
            )
            for swipe in session_data["swipe_history"]
        ]
        
        # Restore user profile vector
        if session_data["user_profile_vector"]:
            self.user_profile_vector = np.array(session_data["user_profile_vector"])
        
        self.console.print(f"📂 Session loaded from {filename}")

def create_sample_data():
    """Create sample data for testing"""
    sample_items = [
        {
            "id": "1",
            "title": "The Great Gatsby",
            "description": "A classic American novel about the Jazz Age",
            "tags": "classic literature fiction drama",
            "category": "books"
        },
        {
            "id": "2",
            "title": "Inception",
            "description": "A mind-bending sci-fi thriller about dreams",
            "tags": "sci-fi thriller action christopher nolan",
            "category": "movies"
        },
        {
            "id": "3",
            "title": "The Beatles - Abbey Road",
            "description": "Iconic album by the legendary British band",
            "tags": "rock classic 60s beatles music",
            "category": "music"
        },
        {
            "id": "4",
            "title": "Machine Learning Course",
            "description": "Learn the fundamentals of machine learning and AI",
            "tags": "education technology programming AI",
            "category": "courses"
        },
        {
            "id": "5",
            "title": "Mediterranean Pasta Recipe",
            "description": "Delicious pasta with olives, tomatoes, and herbs",
            "tags": "cooking italian food recipe healthy",
            "category": "recipes"
        },
        {
            "id": "6",
            "title": "Hiking in Yosemite",
            "description": "Explore the beautiful trails of Yosemite National Park",
            "tags": "outdoors nature hiking adventure california",
            "category": "travel"
        },
        {
            "id": "7",
            "title": "Python Programming Bootcamp",
            "description": "Comprehensive course for learning Python from scratch",
            "tags": "programming python coding education technology",
            "category": "courses"
        },
        {
            "id": "8",
            "title": "Blade Runner 2049",
            "description": "Stunning sequel to the sci-fi classic",
            "tags": "sci-fi cyberpunk future dystopian",
            "category": "movies"
        }
    ]
    return sample_items

def main():
    """Main interactive loop"""
    console = Console()
    swiper = RecommendationSwiper()
    
    # Load existing session if available
    swiper.load_session()
    
    # Add sample data if database is empty
    if len(swiper.items_collection.get()['ids']) == 0:
        console.print("🚀 Setting up sample data...")
        sample_items = create_sample_data()
        swiper.add_items(sample_items)
    
    console.print("\n🎉 Welcome to the Recommendation Swiper!")
    console.print("Commands: 'like', 'dislike', 'skip', 'stats', 'save', 'quit'")
    
    while True:
        # Get recommendations
        recommendations = swiper.get_recommendations(n_results=1)
        
        if not recommendations:
            console.print("🎊 No more recommendations available!")
            break
        
        current_item = recommendations[0]
        
        # Display current item
        panel_content = f"""
🏷️  **{current_item['title']}**

📝 {current_item['description']}

🏷️  Tags: {current_item['tags']}
📂 Category: {current_item['category']}
        """
        
        console.print(Panel(panel_content, title="Current Recommendation", expand=False))
        
        # Get user input
        action = Prompt.ask(
            "What's your choice?",
            choices=["like", "dislike", "skip", "stats", "save", "quit"],
            default="skip"
        )
        
        if action == "quit":
            break
        elif action == "save":
            swiper.save_session()
            continue
        elif action == "stats":
            stats = swiper.get_stats()
            console.print(Panel(
                f"Total Swipes: {stats['total_swipes']}\n"
                f"Likes: {stats['likes']}\n"
                f"Dislikes: {stats['dislikes']}\n"
                f"Like Rate: {stats['like_rate']:.1%}",
                title="📊 Your Stats"
            ))
            continue
        elif action == "skip":
            continue
        else:
            swiper.swipe(current_item['id'], action)
    
    # Auto-save on exit
    swiper.save_session()
    console.print("👋 Thanks for using the Recommendation Swiper!")

if __name__ == "__main__":
    main()