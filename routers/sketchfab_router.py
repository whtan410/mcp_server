# routers/sketchfab_router.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from services.sketchfab_service import SketchfabService

router = APIRouter(prefix="/api/sketchfab", tags=["sketchfab"])

# Pydantic models for request/response
class ModelInfoResponse(BaseModel):
    definition: str
    model_name: Optional[str] = None
    preview_link: Optional[str] = None
    is_downloadable: Optional[bool] = None
    error: Optional[str] = None

class ModelSearchResponse(BaseModel):
    uid: str
    name: str
    preview_link: str
    thumbnail_url: str
    author: str
    is_downloadable: bool

class ModelDetailsResponse(BaseModel):
    uid: str
    name: str
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    views: Optional[int] = None
    likes: Optional[int] = None
    author: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

# Dependency
def get_sketchfab_service():
    return SketchfabService()

@router.post("/model-info", response_model=ModelInfoResponse)
async def get_model_info(
    query: str,
    service: SketchfabService = Depends(get_sketchfab_service)
):
    """Get model info with AI-generated description"""
    try:
        result = await service.get_model_info(query)
        return result
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return ModelInfoResponse(
            definition=f"An error occurred while processing your query",
            model_name=None,
            preview_link=None,
            is_downloadable=None,
            error=f"Error: {str(e)}\n\nTraceback: {tb}"
        )

# @router.get("/search", response_model=List[ModelSearchResponse])
# async def search_models(
#     query: str,
#     limit: Optional[int] = 5,
#     service: SketchfabService = Depends(get_sketchfab_service)
# ):
#     """Search for 3D models"""
#     results = await service.search_models(query, limit)
#     if results and "error" in results[0]:
#         raise HTTPException(status_code=500, detail=results[0]["error"])
#     return results

# @router.get("/details/{uid}", response_model=ModelDetailsResponse)
# async def get_model_details(
#     uid: str,
#     service: SketchfabService = Depends(get_sketchfab_service)
# ):
#     """Get detailed info for a specific model"""
#     result = await service.get_model_details(uid)
#     if "error" in result:
#         raise HTTPException(status_code=500, detail=result["error"])
#     return result