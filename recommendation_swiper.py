import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import List, Dict
from dataclasses import dataclass
import os
import random
from collections import defaultdict

@dataclass
class SwipeResult:
    item_id: str
    action: str  # 'like' or 'dislike'
    timestamp: float

class RecommendationSwiper:
    def __init__(self, db_path: str = "./vector_db"):
        """Initialize the recommendation system"""
        
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
        
        # Simple recommendation parameters
        self.exploration_ratio = 0.4  # 40% exploration, 60% exploitation
        self.liked_items_embeddings = []  # Store embeddings of liked items
        self.disliked_items_embeddings = []  # Store embeddings of disliked items
        
        # Direction switching parameters
        self.consecutive_dislike_threshold = 3  # Switch direction after 3 consecutive dislikes
        self.direction_switch_active = False  # Whether we're in direction switch mode
        self.switch_exploration_ratio = 0.8  # Much higher exploration when switching direction
    
        
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(name)
    
    def add_items(self, items_data):
        """Add items to the vector database"""
        import pandas as pd
        
        # Convert to DataFrame if it's not already
        if isinstance(items_data, list):
            items_df = pd.DataFrame(items_data)
        elif isinstance(items_data, pd.DataFrame):
            items_df = items_data.copy()
        else:
            raise ValueError("items_data must be either a list or pandas DataFrame")
        
        # Get existing items from the database
        existing_items = self.items_collection.get()
        existing_ids = set(existing_items['ids']) if existing_items['ids'] else set()
        
        # Filter out items that already exist
        new_items_df = items_df[~items_df['product_id'].astype(str).isin(existing_ids)]
        
        if len(new_items_df) == 0:
            print("All items already exist in the database.")
            return
            
        print(f"Adding {len(new_items_df)} new items to the database...")
        
        # Clean the data before adding to ChromaDB
        # Convert None values to appropriate defaults
        for col in new_items_df.columns:
            if new_items_df[col].dtype == 'object':  # String columns
                new_items_df[col] = new_items_df[col].fillna('')
            elif new_items_df[col].dtype in ['int64', 'float64']:  # Numeric columns
                new_items_df[col] = new_items_df[col].fillna(0)
            elif new_items_df[col].dtype == 'bool':  # Boolean columns
                new_items_df[col] = new_items_df[col].fillna(False)
        
        # Convert datetime columns to strings
        for col in new_items_df.columns:
            if new_items_df[col].dtype == 'datetime64[ns, UTC]' or 'datetime' in str(new_items_df[col].dtype):
                new_items_df[col] = new_items_df[col].astype(str)
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for _, row in new_items_df.iterrows():
            # Create document text (you might want to customize this)
            doc_text = f"Product: {row.get('title', '')} Description: {row.get('body_html', '')}"
            documents.append(doc_text)
            
            # Create metadata dictionary, filtering out problematic values
            metadata = {}
            for key, value in row.items():
                if key == 'product_id':  # Use product_id as the ID
                    continue
                
                # Ensure value is not None and is a supported type
                if value is not None:
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)
                else:
                    # Set appropriate default for None values
                    metadata[key] = ""
            
            metadatas.append(metadata)
            ids.append(str(row['product_id']))
        
        # Add to ChromaDB
        self.items_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully added {len(new_items_df)} new items to the database.")
    
    def _get_item_embedding(self, item_id: str) -> np.ndarray:
        """Get embedding for a specific item"""
        try:
            result = self.items_collection.get(ids=[item_id], include=['embeddings'])
            embeddings = result.get('embeddings')
            if embeddings is not None and len(embeddings) > 0:
                embedding = embeddings[0]
                # Check if embedding exists and is not empty
                if embedding is not None and hasattr(embedding, '__len__') and len(embedding) > 0:
                    return np.array(embedding)
        except Exception as e:
            print(f"Error getting embedding for {item_id}: {e}")
        return None
    
    def _calculate_preference_score(self, item_embedding: np.ndarray) -> float:
        """Calculate how much user would like this item based on history"""
        if item_embedding is None:
            return 0.5  # Neutral score for items without embeddings
        
        like_score = 0.0
        dislike_score = 0.0
        
        # Calculate similarity to liked items
        if self.liked_items_embeddings:
            like_similarities = []
            for liked_emb in self.liked_items_embeddings:
                similarity = np.dot(item_embedding, liked_emb) / (
                    np.linalg.norm(item_embedding) * np.linalg.norm(liked_emb))
                like_similarities.append(max(0, similarity))
            like_score = np.mean(like_similarities) if like_similarities else 0
        
        # Calculate similarity to disliked items (penalty)
        if self.disliked_items_embeddings:
            dislike_similarities = []
            for disliked_emb in self.disliked_items_embeddings:
                similarity = np.dot(item_embedding, disliked_emb) / (
                    np.linalg.norm(item_embedding) * np.linalg.norm(disliked_emb))
                dislike_similarities.append(max(0, similarity))
            dislike_score = np.mean(dislike_similarities) if dislike_similarities else 0
        
        # Combine scores: boost likes, penalize similarity to dislikes
        final_score = like_score - (dislike_score * 0.5)
        return max(0.0, min(1.0, final_score))  # Clamp between 0 and 1
    
    def _check_consecutive_dislikes(self) -> int:
        """Count consecutive dislikes from the end of swipe history"""
        if not self.swipe_history:
            return 0
        
        consecutive_dislikes = 0
        for swipe in reversed(self.swipe_history):
            if swipe.action == 'dislike':
                consecutive_dislikes += 1
            else:
                break
        
        return consecutive_dislikes
    
    def _should_switch_direction(self) -> bool:
        """Check if we should switch direction due to consecutive dislikes"""
        consecutive_dislikes = self._check_consecutive_dislikes()
        
        # Check if the last action was a like (to deactivate direction switch)
        if self.swipe_history and self.swipe_history[-1].action == 'like':
            if self.direction_switch_active:
                print("✅ Direction switch deactivated - found something you like!")
                self.direction_switch_active = False
            return False
        
        # Activate direction switch if we hit the threshold
        if consecutive_dislikes >= self.consecutive_dislike_threshold:
            if not self.direction_switch_active:
                print(f"🔄 DIRECTION SWITCH ACTIVATED! {consecutive_dislikes} consecutive dislikes detected")
                self.direction_switch_active = True
            return True
        
        # Deactivate if consecutive dislikes drop below threshold
        if self.direction_switch_active and consecutive_dislikes < self.consecutive_dislike_threshold:
            print(f"✅ Direction switch deactivated - consecutive dislikes dropped to {consecutive_dislikes}")
            self.direction_switch_active = False
        
        return self.direction_switch_active
    
    def _get_anti_preference_score(self, item_embedding: np.ndarray) -> float:
        """Calculate score that's OPPOSITE to user preferences - for direction switching"""
        if item_embedding is None:
            return 0.5
        
        # Start with neutral score
        anti_score = 0.5
        
        # AVOID similarity to liked items (opposite of normal behavior)
        if self.liked_items_embeddings:
            like_similarities = []
            for liked_emb in self.liked_items_embeddings:
                similarity = np.dot(item_embedding, liked_emb) / (
                    np.linalg.norm(item_embedding) * np.linalg.norm(liked_emb))
                like_similarities.append(max(0, similarity))
            avg_like_similarity = np.mean(like_similarities)
            # PENALIZE similarity to likes (opposite of normal)
            anti_score -= avg_like_similarity * 0.5
        
        # PREFER similarity to disliked items (find different areas)
        if self.disliked_items_embeddings:
            dislike_similarities = []
            for disliked_emb in self.disliked_items_embeddings:
                similarity = np.dot(item_embedding, disliked_emb) / (
                    np.linalg.norm(item_embedding) * np.linalg.norm(disliked_emb))
                dislike_similarities.append(max(0, similarity))
            avg_dislike_similarity = np.mean(dislike_similarities)
            # AVOID areas similar to dislikes too (explore completely new areas)
            anti_score -= avg_dislike_similarity * 0.3
        
        return max(0.0, min(1.0, anti_score))
    
    def get_recommendations(self, n_results: int = 10, exclude_seen: bool = True) -> List[Dict]:
        """Simple, effective recommendations with direction switching for consecutive dislikes"""
        
        # Get all available items
        all_items = self.items_collection.get(include=['metadatas'])
        if not all_items['ids']:
            return []
        
        seen_ids = {swipe.item_id for swipe in self.swipe_history}
        candidates = []
        
        # Check if we should switch direction
        direction_switch = self._should_switch_direction()
        consecutive_dislikes = self._check_consecutive_dislikes()
        
        print(f"Recommending from {len(all_items['ids'])} total items")
        print(f"User has liked {len(self.liked_items_embeddings)} items, disliked {len(self.disliked_items_embeddings)} items")
        if direction_switch:
            print(f"🔄 DIRECTION SWITCH MODE: Exploring completely new areas (consecutive dislikes: {consecutive_dislikes})")
        
        # Score all unseen items
        for i, item_id in enumerate(all_items['ids']):
            if exclude_seen and item_id in seen_ids:
                continue
            
            # Get item embedding for preference calculation
            item_embedding = self._get_item_embedding(item_id)
            
            # Calculate preference score based on current mode
            if direction_switch:
                # Direction switch mode: explore opposite directions
                preference_score = self._get_anti_preference_score(item_embedding)
                exploration_ratio = self.switch_exploration_ratio  # Much higher exploration
                recommendation_type = "direction_switch"
            else:
                # Normal mode: use regular preferences
                preference_score = self._calculate_preference_score(item_embedding)
                exploration_ratio = self.exploration_ratio
                recommendation_type = "learned" if (self.liked_items_embeddings or self.disliked_items_embeddings) else "cold_start"
            
            # Add some randomness for exploration
            random_score = random.random()
            
            # Combine scores based on mode
            if recommendation_type != "cold_start":
                final_score = (1 - exploration_ratio) * preference_score + exploration_ratio * random_score
            else:
                # Cold start: pure random
                final_score = random_score
            
            metadata = all_items['metadatas'][i]
            candidates.append({
                'id': item_id,
                'title': metadata.get('title', ''),
                'description': metadata.get('description', ''),
                'tags': metadata.get('tags', ''),
                'category': metadata.get('category', ''),
                'preference_score': preference_score,
                'random_score': random_score,
                'final_score': final_score,
                'recommendation_type': recommendation_type,
                'consecutive_dislikes': consecutive_dislikes,
                'exploration_ratio': exploration_ratio
            })
        
        # Sort by final score and return top results
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        top_candidates = candidates[:n_results]
        
        print(f"Returning {len(top_candidates)} recommendations ({top_candidates[0]['recommendation_type'] if top_candidates else 'none'})")
        return top_candidates
    
    def swipe(self, item_id: str, action: str):
        """Record a swipe action and update user preferences"""
        import time
        
        swipe_result = SwipeResult(
            item_id=item_id,
            action=action,
            timestamp=time.time()
        )
        self.swipe_history.append(swipe_result)
        
        # Get item embedding and add to appropriate collection
        item_embedding = self._get_item_embedding(item_id)
        if item_embedding is not None:
            if action == 'like':
                self.liked_items_embeddings.append(item_embedding)
                print(f"Added item {item_id} to liked embeddings (total: {len(self.liked_items_embeddings)})")
            else:  # dislike
                self.disliked_items_embeddings.append(item_embedding)
                print(f"Added item {item_id} to disliked embeddings (total: {len(self.disliked_items_embeddings)})")
        else:
            print(f"Warning: Could not get embedding for item {item_id}")
        
        print(f"Recorded swipe: {action} for item {item_id}")
        
    
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
            "liked_items_embeddings": [emb.tolist() for emb in self.liked_items_embeddings],
            "disliked_items_embeddings": [emb.tolist() for emb in self.disliked_items_embeddings],
            "exploration_ratio": self.exploration_ratio,
            "consecutive_dislike_threshold": self.consecutive_dislike_threshold,
            "direction_switch_active": self.direction_switch_active,
            "switch_exploration_ratio": self.switch_exploration_ratio
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
    
    def load_session(self, filename: str = "swipe_session.json"):
        """Load a previous session"""
        if not os.path.exists(filename):
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
        
        # Restore embedding collections
        if "liked_items_embeddings" in session_data:
            self.liked_items_embeddings = [np.array(emb) for emb in session_data["liked_items_embeddings"]]
        
        if "disliked_items_embeddings" in session_data:
            self.disliked_items_embeddings = [np.array(emb) for emb in session_data["disliked_items_embeddings"]]
        
        # Restore exploration ratio
        if "exploration_ratio" in session_data:
            self.exploration_ratio = session_data["exploration_ratio"]
        
        # Restore direction switch parameters
        if "consecutive_dislike_threshold" in session_data:
            self.consecutive_dislike_threshold = session_data["consecutive_dislike_threshold"]
        
        if "direction_switch_active" in session_data:
            self.direction_switch_active = session_data["direction_switch_active"]
        
        if "switch_exploration_ratio" in session_data:
            self.switch_exploration_ratio = session_data["switch_exploration_ratio"]
        
        print(f"Loaded session: {len(self.swipe_history)} swipes, {len(self.liked_items_embeddings)} likes, {len(self.disliked_items_embeddings)} dislikes")
        
        # Check if direction switch should be active
        consecutive_dislikes = self._check_consecutive_dislikes()
        if consecutive_dislikes >= self.consecutive_dislike_threshold:
            print(f"📍 Session loaded in direction switch mode ({consecutive_dislikes} consecutive dislikes)")
    
    def set_exploration_ratio(self, ratio: float):
        """Set the exploration ratio (0.0 = pure exploitation, 1.0 = pure exploration)"""
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Exploration ratio must be between 0.0 and 1.0")
        self.exploration_ratio = ratio
        print(f"Exploration ratio set to {ratio}")
    
    def set_consecutive_dislike_threshold(self, threshold: int):
        """Set the threshold for consecutive dislikes that triggers direction switch"""
        if threshold < 2:
            raise ValueError("Consecutive dislike threshold must be at least 2")
        self.consecutive_dislike_threshold = threshold
        print(f"Consecutive dislike threshold set to {threshold}")
    
    def get_recommendation_stats(self) -> Dict:
        """Get statistics about the recommendation system"""
        consecutive_dislikes = self._check_consecutive_dislikes()
        return {
            "total_swipes": len(self.swipe_history),
            "liked_items": len(self.liked_items_embeddings),
            "disliked_items": len(self.disliked_items_embeddings),
            "exploration_ratio": self.exploration_ratio,
            "consecutive_dislikes": consecutive_dislikes,
            "consecutive_dislike_threshold": self.consecutive_dislike_threshold,
            "direction_switch_active": self.direction_switch_active,
            "switch_exploration_ratio": self.switch_exploration_ratio,
            "learning_status": "direction_switch" if self.direction_switch_active else ("learned" if (self.liked_items_embeddings or self.disliked_items_embeddings) else "cold_start")
        }