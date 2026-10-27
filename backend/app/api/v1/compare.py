"""1v1 Model Battle Comparison Endpoint."""

from fastapi import APIRouter, HTTPException
from backend.app.schemas.compare import CompareRequest, CompareResponse
from backend.app.services.model_service import model_service

router = APIRouter(prefix="/compare", tags=["Compare"])


@router.post("", response_model=CompareResponse)
def compare_models(req: CompareRequest):
    """Execute side-by-side Top-N comparison between two models for the same user."""
    try:
        res_a = model_service.get_recommendations(user_id=req.user_id, model_id=req.model_a_id, top_n=req.top_n)
        res_b = model_service.get_recommendations(user_id=req.user_id, model_id=req.model_b_id, top_n=req.top_n)

        ids_a = {it.movie_id for it in res_a.recommendations}
        ids_b = {it.movie_id for it in res_b.recommendations}

        shared_ids = sorted(list(ids_a.intersection(ids_b)))
        union_ids = ids_a.union(ids_b)
        jaccard = round(len(shared_ids) / len(union_ids), 4) if union_ids else 0.0

        return CompareResponse(
            user_id=req.user_id,
            model_a_id=req.model_a_id,
            model_a_name=res_a.model_name,
            model_b_id=req.model_b_id,
            model_b_name=res_b.model_name,
            top_n=req.top_n,
            jaccard_similarity=jaccard,
            overlap_count=len(shared_ids),
            recommendations_a=res_a.recommendations,
            recommendations_b=res_b.recommendations,
            shared_movie_ids=shared_ids,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(exc)}")
