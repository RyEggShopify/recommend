import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import List, Dict
from dataclasses import dataclass
import os
import random

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
    
    def get_recommendations(self, n_results: int = 10, exclude_seen: bool = True) -> List[Dict]:
        """Get recommendations based on current user preferences"""
        if self.user_profile_vector is not None:
            print("Using user profile for recommendations")
            # Use user profile for similarity search
            results = self.items_collection.query(
                query_embeddings=[self.user_profile_vector.tolist()],
                n_results=n_results * 2  # Get more to filter out seen items
            )
        else:
            print("Cold start: using random sampling")
            # Cold start: get random items
            all_items = self.items_collection.get()
            if not all_items['ids']:
                return []
            
            # Get random sample of items for cold start
            random_ids = random.sample(all_items['ids'], min(n_results * 2, len(all_items['ids'])))
            results = self.items_collection.get(
                ids=random_ids,
                include=['metadatas']
            )
            
            # Restructure results to match query format
            results = {
                'ids': [results['ids']],
                'metadatas': [results['metadatas']],
                'distances': [[0] * len(results['ids'])]  # No meaningful distance for random sampling
            }
        
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
        
        print(f"Recorded swipe: {action} for item {item_id}")
        
        # Update user profile based on the swipe
        self._update_user_profile()
        
    
    def _update_user_profile(self):
        """Update user profile vector based on swipe history"""
        if not self.swipe_history:
            return
        
        print(f"Updating user profile based on {len(self.swipe_history)} swipes")
        
        liked_embeddings = []
        disliked_embeddings = []
        
        # Get embeddings for liked and disliked items
        for swipe in self.swipe_history[-20:]:  # Consider recent 20 swipes
            try:
                # Get item embedding from database
                result = self.items_collection.get(ids=[swipe.item_id], include=['embeddings'])
                
                # Check if embeddings exist and are not empty
                if (result.get('embeddings') is not None and 
                    len(result['embeddings']) > 0 and 
                    result['embeddings'][0] is not None):
                    
                    embedding = np.array(result['embeddings'][0])
                    
                    if swipe.action == 'like':
                        liked_embeddings.append(embedding)
                        print(f"Added liked embedding for item {swipe.item_id}")
                    else:
                        disliked_embeddings.append(embedding)
                        print(f"Added disliked embedding for item {swipe.item_id}")
                else:
                    print(f"No embeddings found for item {swipe.item_id}")
                    
            except Exception as e:
                print(f"Error getting embedding for item {swipe.item_id}: {e}")
                continue
        
        # Calculate user profile vector
        if liked_embeddings:
            liked_centroid = np.mean(liked_embeddings, axis=0)
            print(f"Calculated liked centroid from {len(liked_embeddings)} items")
            
            if disliked_embeddings:
                disliked_centroid = np.mean(disliked_embeddings, axis=0)
                # Move towards liked items and away from disliked items
                self.user_profile_vector = liked_centroid - 0.3 * disliked_centroid
                print(f"Updated profile vector considering {len(disliked_embeddings)} disliked items")
            else:
                self.user_profile_vector = liked_centroid
                print("Updated profile vector with only liked items")
            
            # Normalize the vector
            self.user_profile_vector = self.user_profile_vector / np.linalg.norm(self.user_profile_vector)
            print("User profile vector updated and normalized")
        else:
            print("No liked embeddings found - user profile not updated")
    
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
        
        # Restore user profile vector
        if session_data["user_profile_vector"]:
            self.user_profile_vector = np.array(session_data["user_profile_vector"])