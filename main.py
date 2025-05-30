from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import uvicorn
from recommendation_swiper import RecommendationSwiper
import os
import ast
import json

# Pydantic models
class SwipeRequest(BaseModel):
    item_id: str
    action: str  # 'like' or 'dislike'

class StatsResponse(BaseModel):
    likes: int
    dislikes: int
    total_swipes: int

class RecommendationResponse(BaseModel):
    id: str
    # title: str
    description: str | None = None
    recommendation_type: str | None = None
    final_score: float | None = None

class SwipeResponse(BaseModel):
    success: bool
    message: str
    # item_title: str
    # item_title: str
    action: str

# Global application state
class AppState:
    def __init__(self):
        self.swiper: RecommendationSwiper | None = None
        self.bigquery_client = None
        self.initialized = False
    def parse_json(self, json_path="data.json", target_dim=768):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not found")

        items = []
        skipped_items = 0

        with open(json_path, "r") as file:
            for idx, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                obj = json.loads(line)
                vector = obj.get("product_search_document_embedding")

                # embedding stored as string
                if isinstance(vector, str):
                    vector = ast.literal_eval(vector)

                # Check embedding dimensions explicitly
                if len(vector) != target_dim:
                    skipped_items += 1
                    print(f"⚠️ Skipped line {idx}: Embedding dimension is {len(vector)}, expected {target_dim}")
                    continue

                items.append({
                    "id": str(obj["product_id"]),
                    "embedding": vector,
                    "gmv_usd_60d": float(obj["gmv_usd_60d"])
                })

        if skipped_items:
            print(f"⚠️ Skipped {skipped_items} items due to mismatch dimensions.")

        if not self.swiper:
            raise Exception("Swiper is not initialized")

        return items


    def initialize(self):
        if not self.initialized:
            print("🚀 Initializing recommendation swiper...")
            self.swiper = RecommendationSwiper()

            data = self.parse_json()
            self.swiper.add_items(data)

            # Load existing session
            self.swiper.load_session()
            self.initialized = True
            print("✅ Initialization complete!")


state = AppState()

# Modern lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting up...")
    state.initialize()
    yield
    # Shutdown
    print("🛑 Shutting down...")
    if state.swiper:
        state.swiper.save_session()
        print("💾 Session saved on shutdown")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Recommendation Swiper API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Recommendation Swiper API is running!", "status": "healthy"}

@app.get("/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(n_results: int = 3):
    """Get current top N recommendations"""
    if not state.initialized or not state.swiper:
        raise HTTPException(status_code=500, detail="Application not initialized")

    try:
        recommendations = state.swiper.get_recommendations(n_results=n_results)

        # Convert to response format
        formatted_recommendations = []
        print("\n📋 Sending Recommendations:")
        for idx, item in enumerate(recommendations, 1):
        print("\n📋 Sending Recommendations:")
        for idx, item in enumerate(recommendations, 1):
            # Clean up HTML tags from description
            description = item.get('description', '')
            if description:
                description = description.replace('<', '&lt;').replace('>', '&gt;')
                if len(description) > 200:
                    description = description[:200] + "..."

            print("item", item)
            formatted_item = {
            print("item", item)
            formatted_item = {
                'id': str(item['id']),
                # 'title': item['title'],
                'description': description,
                'recommendation_type': item.get('recommendation_type', 'unknown'),
                'final_score': item.get('final_score', 0.0)
            }
            formatted_recommendations.append(formatted_item)

            # Log each recommendation
            print(f"\n  {idx}. Recommendation:")
            print(f"     ID: {formatted_item['id']}")
            # print(f"     Title: {formatted_item['title']}")
            print(f"     Description: {formatted_item['description']}")
            print(f"     Type: {formatted_item['recommendation_type']}")
            print(f"     Score: {formatted_item['final_score']:.2f}")
                # 'title': item['title'],
                'description': description,
                'recommendation_type': item.get('recommendation_type', 'unknown'),
                'final_score': item.get('final_score', 0.0)
            }
            formatted_recommendations.append(formatted_item)

            # Log each recommendation
            print(f"\n  {idx}. Recommendation:")
            print(f"     ID: {formatted_item['id']}")
            # print(f"     Title: {formatted_item['title']}")
            print(f"     Description: {formatted_item['description']}")
            print(f"     Type: {formatted_item['recommendation_type']}")
            print(f"     Score: {formatted_item['final_score']:.2f}")

        return formatted_recommendations

    except Exception as e:
        print(f"\n❌ Error getting recommendations: {str(e)}")
        print(f"\n❌ Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/swipe", response_model=SwipeResponse)
async def swipe_item(swipe_request: SwipeRequest):
    """Handle like/dislike action on an item"""
    if not state.initialized or not state.swiper:
        raise HTTPException(status_code=500, detail="Application not initialized")

    if swipe_request.action not in ['like', 'dislike']:
        raise HTTPException(status_code=400, detail="Action must be 'like' or 'dislike'")

    try:
        # Log the incoming swipe request
        print("\n📱 Received Swipe Request:")
        print(f"  Item ID: {swipe_request.item_id}")
        print(f"  Action: {swipe_request.action}")
        
        # Check against stored recommendations
        current_recommendations = state.swiper.current_recommendations
        if not current_recommendations:
            # If no stored recommendations, fetch new ones
            current_recommendations = state.swiper.get_recommendations(n_results=10)
        # Log the incoming swipe request
        print("\n📱 Received Swipe Request:")
        print(f"  Item ID: {swipe_request.item_id}")
        print(f"  Action: {swipe_request.action}")
        
        # Check against stored recommendations
        current_recommendations = state.swiper.current_recommendations
        if not current_recommendations:
            # If no stored recommendations, fetch new ones
            current_recommendations = state.swiper.get_recommendations(n_results=10)

        # Log current recommendations
        print("\n📋 Current Recommendations:")
        for idx, item in enumerate(current_recommendations, 1):
            print(f"  {idx}. ID: {item['id']}")
        # Log current recommendations
        print("\n📋 Current Recommendations:")
        for idx, item in enumerate(current_recommendations, 1):
            print(f"  {idx}. ID: {item['id']}")

        # Perform the swipe
        state.swiper.swipe(swipe_request.item_id, swipe_request.action)

        # Remove the swiped item from current recommendations
        state.swiper.current_recommendations = [
            item for item in current_recommendations 
            if str(item['id']) != swipe_request.item_id
        ]

        # Remove the swiped item from current recommendations
        state.swiper.current_recommendations = [
            item for item in current_recommendations 
            if str(item['id']) != swipe_request.item_id
        ]

        action_emoji = "✅" if swipe_request.action == 'like' else "❌"
        message = f"{action_emoji} {swipe_request.action.upper()}: {swipe_request.item_id}"
        message = f"{action_emoji} {swipe_request.action.upper()}: {swipe_request.item_id}"

        return SwipeResponse(
            success=True,
            message=message,
            # item_title=item_title,
            # item_title=item_title,
            action=swipe_request.action
        )

    except Exception as e:
        print(f"\n❌ Error processing swipe: {str(e)}")
        print(f"\n❌ Error processing swipe: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing swipe: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get current swipe statistics"""
    if not state.initialized or not state.swiper:
        raise HTTPException(status_code=500, detail="Application not initialized")

    try:
        stats = state.swiper.get_stats()
        return StatsResponse(
            likes=stats['likes'],
            dislikes=stats['dislikes'],
            total_swipes=stats['total_swipes']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/save")
async def save_session():
    """Save the current session"""
    if not state.initialized or not state.swiper:
        raise HTTPException(status_code=500, detail="Application not initialized")

    try:
        state.swiper.save_session()
        return {"message": "💾 Session saved successfully!", "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving session: {str(e)}")

@app.get("/current")
async def get_current_item():
    """Get the current top recommendation"""
    if not state.initialized or not state.swiper:
        raise HTTPException(status_code=500, detail="Application not initialized")

    try:
        recommendations = state.swiper.get_recommendations(n_results=1)
        if not recommendations:
            return {"message": "No more recommendations available", "item": None}

        item = recommendations[0]
        description = item.get('description', '')
        if description:
            description = description.replace('<', '&lt;').replace('>', '&gt;')
            if len(description) > 200:
                description = description[:200] + "..."

        return {
            "item": {
                'id': str(item['id']),
                'title': item['title'],
                'description': description
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting current item: {str(e)}")

def main():
    """Main function to run the server"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
