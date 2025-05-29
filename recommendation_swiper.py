import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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
        
        # Advanced recommendation parameters
        self.exploration_ratio = 0.4  # 40% exploration, 60% exploitation
        self.liked_items_embeddings = []  # Store embeddings of liked items
        self.disliked_items_embeddings = []  # Store embeddings of disliked items
        
        # Thematic similarity parameters
        self.similarity_threshold_min = 0.3  # Minimum similarity for relevance
        self.similarity_threshold_max = 0.85  # Maximum similarity to avoid near-duplicates
        self.diversity_weight = 0.3  # Weight for diversity in recommendations
        
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
        
        # Filter out items that already exist and create a proper copy
        new_items_df = items_df[~items_df['product_id'].astype(str).isin(existing_ids)].copy()
        
        if len(new_items_df) == 0:
            print("All items already exist in the database.")
            return
            
        print(f"Adding {len(new_items_df)} new items to the database...")
        
        # Clean the data before adding to ChromaDB
        # Convert None values to appropriate defaults using .loc
        for col in new_items_df.columns:
            if new_items_df[col].dtype == 'object':  # String columns
                new_items_df.loc[:, col] = new_items_df[col].fillna('')
            elif new_items_df[col].dtype in ['int64', 'float64']:  # Numeric columns
                new_items_df.loc[:, col] = new_items_df[col].fillna(0)
            elif new_items_df[col].dtype == 'bool':  # Boolean columns
                new_items_df.loc[:, col] = new_items_df[col].fillna(False)
        
        # Convert datetime columns to strings using .loc
        for col in new_items_df.columns:
            if new_items_df[col].dtype == 'datetime64[ns, UTC]' or 'datetime' in str(new_items_df[col].dtype):
                new_items_df.loc[:, col] = new_items_df[col].astype(str)
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for _, row in new_items_df.iterrows():
            # Create document text (you might want to customize this)
            doc_text = f"Product: {row.get('title', '')} Description: {row.get('description', '')}"
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
    
    def _calculate_thematic_similarity(self, item_embedding: np.ndarray, reference_embeddings: List[np.ndarray]) -> Tuple[float, float]:
        """Calculate thematic similarity that avoids near-duplicates"""
        if not reference_embeddings:
            return 0.0, 1.0
        
        similarities = []
        for ref_emb in reference_embeddings:
            similarity = np.dot(item_embedding, ref_emb) / (
                np.linalg.norm(item_embedding) * np.linalg.norm(ref_emb))
            similarities.append(max(0, similarity))
        
        max_similarity = max(similarities)
        avg_similarity = np.mean(similarities)
        
        # Apply similarity bounds for thematic relevance without exact matches
        if max_similarity > self.similarity_threshold_max:
            # Too similar - likely near-duplicate
            thematic_score = 0.1
            diversity_bonus = 0.0
        elif avg_similarity < self.similarity_threshold_min:
            # Not similar enough - not thematically relevant
            thematic_score = avg_similarity * 0.5
            diversity_bonus = 0.8
        else:
            # Sweet spot - thematically similar but not duplicate
            normalized_sim = (avg_similarity - self.similarity_threshold_min) / \
                           (self.similarity_threshold_max - self.similarity_threshold_min)
            thematic_score = 0.3 + (normalized_sim * 0.6)  # Scale to 0.3-0.9
            diversity_bonus = 1.0 - (max_similarity - self.similarity_threshold_min) / \
                            (self.similarity_threshold_max - self.similarity_threshold_min)
        
        return thematic_score, diversity_bonus
    
    def _calculate_preference_score(self, item_embedding: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive preference score with thematic similarity and diversity"""
        if item_embedding is None:
            return {
                'thematic_score': 0.5,
                'diversity_score': 0.5,
                'dislike_penalty': 0.0,
                'final_score': 0.5
            }
        
        # Calculate thematic similarity to liked items
        like_thematic_score = 0.0
        like_diversity_bonus = 1.0
        if self.liked_items_embeddings:
            like_thematic_score, like_diversity_bonus = self._calculate_thematic_similarity(
                item_embedding, self.liked_items_embeddings)
        
        # Calculate penalty from disliked items
        dislike_penalty = 0.0
        if self.disliked_items_embeddings:
            dislike_similarities = []
            for disliked_emb in self.disliked_items_embeddings:
                similarity = np.dot(item_embedding, disliked_emb) / (
                    np.linalg.norm(item_embedding) * np.linalg.norm(disliked_emb))
                dislike_similarities.append(max(0, similarity))
            # Strong penalty for high similarity to disliked items
            max_dislike_sim = max(dislike_similarities) if dislike_similarities else 0
            dislike_penalty = max_dislike_sim * 0.6
        
        # Combine scores with diversity weighting
        diversity_score = like_diversity_bonus * self.diversity_weight
        final_score = like_thematic_score + diversity_score - dislike_penalty
        final_score = max(0.0, min(1.0, final_score))
        
        return {
            'thematic_score': like_thematic_score,
            'diversity_score': diversity_score,
            'dislike_penalty': dislike_penalty,
            'final_score': final_score
        }
    
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
                self.direction_switch_active = False
            return False
        
        # Activate direction switch if we hit the threshold
        if consecutive_dislikes >= self.consecutive_dislike_threshold:
            if not self.direction_switch_active:
                self.direction_switch_active = True
            return True
        
        # Deactivate if consecutive dislikes drop below threshold
        if self.direction_switch_active and consecutive_dislikes < self.consecutive_dislike_threshold:
            self.direction_switch_active = False
        
        return self.direction_switch_active
    
    def _get_anti_preference_score(self, item_embedding: np.ndarray) -> Dict[str, float]:
        """Calculate score that explores different thematic areas - for direction switching"""
        if item_embedding is None:
            return {
                'exploration_score': 0.5,
                'novelty_bonus': 0.5,
                'final_score': 0.5
            }
        
        # Start with neutral exploration score
        exploration_score = 0.5
        novelty_bonus = 0.0
        
        # AVOID high similarity to liked items (explore different themes)
        if self.liked_items_embeddings:
            like_similarities = []
            for liked_emb in self.liked_items_embeddings:
                similarity = np.dot(item_embedding, liked_emb) / (
                    np.linalg.norm(item_embedding) * np.linalg.norm(liked_emb))
                like_similarities.append(max(0, similarity))
            
            max_like_similarity = max(like_similarities) if like_similarities else 0
            # Reward items that are moderately different from likes
            if max_like_similarity < 0.4:
                novelty_bonus = 0.4  # High novelty bonus for very different items
            elif max_like_similarity < 0.7:
                novelty_bonus = 0.2  # Moderate bonus for somewhat different items
            else:
                exploration_score -= 0.3  # Penalty for too similar to likes
        
        # ALSO avoid high similarity to dislikes (find truly new areas)
        if self.disliked_items_embeddings:
            dislike_similarities = []
            for disliked_emb in self.disliked_items_embeddings:
                similarity = np.dot(item_embedding, disliked_emb) / (
                    np.linalg.norm(item_embedding) * np.linalg.norm(disliked_emb))
                dislike_similarities.append(max(0, similarity))
            
            max_dislike_similarity = max(dislike_similarities) if dislike_similarities else 0
            if max_dislike_similarity > 0.6:
                exploration_score -= 0.4  # Strong penalty for similarity to dislikes
        
        final_score = exploration_score + novelty_bonus
        final_score = max(0.0, min(1.0, final_score))
        
        return {
            'exploration_score': exploration_score,
            'novelty_bonus': novelty_bonus,
            'final_score': final_score
        }
    
    def _apply_diversity_filter(self, candidates: List[Dict], n_results: int) -> List[Dict]:
        """Apply diversity filtering to avoid recommending too many similar items"""
        if len(candidates) <= n_results:
            candidates.sort(key=lambda x: x['final_score'], reverse=True)
            return candidates
        
        # Sort by score first
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Take top candidates for diversity filtering
        top_pool = candidates[:min(n_results * 3, len(candidates))]
        selected = []
        
        for candidate in top_pool:
            if len(selected) >= n_results:
                break
                
            # First candidate is always selected
            if not selected:
                selected.append(candidate)
                continue
            
            # Check diversity against already selected items
            candidate_embedding = self._get_item_embedding(candidate['id'])
            if candidate_embedding is None:
                selected.append(candidate)
                continue
            
            # Calculate minimum similarity to selected items
            min_similarity = float('inf')
            for selected_item in selected:
                selected_embedding = self._get_item_embedding(selected_item['id'])
                if selected_embedding is not None:
                    similarity = np.dot(candidate_embedding, selected_embedding) / (
                        np.linalg.norm(candidate_embedding) * np.linalg.norm(selected_embedding))
                    min_similarity = min(min_similarity, similarity)
            
            # Only add if sufficiently different from selected items
            diversity_threshold = 0.9  # Allow some similarity but not near-duplicates
            if min_similarity == float('inf') or min_similarity < diversity_threshold:
                selected.append(candidate)
        
        # Fill remaining slots with highest scoring items if needed
        while len(selected) < n_results and len(selected) < len(candidates):
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                    break
        
        return selected
    
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
        
        if direction_switch:
            print(f"🔄 Direction switch: exploring new themes ({consecutive_dislikes} consecutive dislikes)")
        
        # Score all unseen items
        for i, item_id in enumerate(all_items['ids']):
            if exclude_seen and item_id in seen_ids:
                continue
            
            # Get item embedding for preference calculation
            item_embedding = self._get_item_embedding(item_id)
            
            # Calculate preference score based on current mode
            if direction_switch:
                # Direction switch mode: explore different thematic areas
                score_data = self._get_anti_preference_score(item_embedding)
                preference_score = score_data['final_score']
                exploration_ratio = self.switch_exploration_ratio
                recommendation_type = "direction_switch"
                score_breakdown = score_data
            else:
                # Normal mode: use thematic similarity with diversity
                score_data = self._calculate_preference_score(item_embedding)
                preference_score = score_data['final_score']
                exploration_ratio = self.exploration_ratio
                recommendation_type = "learned" if (self.liked_items_embeddings or self.disliked_items_embeddings) else "cold_start"
                score_breakdown = score_data
            
            # Add randomness for exploration
            random_score = random.random()
            
            # Combine scores based on mode
            if recommendation_type != "cold_start":
                final_score = (1 - exploration_ratio) * preference_score + exploration_ratio * random_score
            else:
                # Cold start: pure random
                final_score = random_score
                score_breakdown = {'final_score': final_score}
            
            metadata = all_items['metadatas'][i]
            candidate = {
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
            }
            # Add score breakdown for debugging
            candidate.update(score_breakdown)
            candidates.append(candidate)
        
        # Apply diversity filter to avoid too many similar items
        top_candidates = self._apply_diversity_filter(candidates, n_results)
        
        if top_candidates:
            print(f"Recommended {len(top_candidates)} items ({top_candidates[0]['recommendation_type']})")
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
            else:
                self.disliked_items_embeddings.append(item_embedding)
        
    
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
        
        # Check if direction switch should be active
        consecutive_dislikes = self._check_consecutive_dislikes()
        if consecutive_dislikes >= self.consecutive_dislike_threshold:
            self.direction_switch_active = True
    
    def set_exploration_ratio(self, ratio: float):
        """Set the exploration ratio (0.0 = pure exploitation, 1.0 = pure exploration)"""
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Exploration ratio must be between 0.0 and 1.0")
        self.exploration_ratio = ratio
        pass
    
    def set_consecutive_dislike_threshold(self, threshold: int):
        """Set the threshold for consecutive dislikes that triggers direction switch"""
        if threshold < 2:
            raise ValueError("Consecutive dislike threshold must be at least 2")
        self.consecutive_dislike_threshold = threshold
        pass
    
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