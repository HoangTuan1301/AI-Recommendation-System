"""Pydantic Schemas for MusicAI REST API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class MusicModelInfo(BaseModel):
    """Metadata and runtime availability for a MusicAI recommendation model."""
    id: str = Field(..., description="Unique model identifier slug, e.g. 'item_cf'")
    name: str = Field(..., description="Human-readable model name")
    family: str = Field(..., description="Algorithm family/paradigm")
    description: str = Field(..., description="Summary of algorithm and characteristics")
    is_personalized: bool = Field(..., description="Whether model provides user-specific recommendations")
    is_default: bool = Field(..., description="Whether this model is the default recommendation engine")
    is_available: bool = Field(..., description="Whether the model checkpoint exists and is ready")
    is_loaded: bool = Field(..., description="Whether the model is currently cached in process RAM")
    checkpoint_file: str = Field(..., description="Serialized checkpoint filename")
    checkpoint_size_kb: Optional[float] = Field(None, description="Checkpoint size in kilobytes")
    latency_profile: Optional[str] = Field(None, description="Expected inference latency tier")


class MusicModelsListResponse(BaseModel):
    """Response containing all available MusicAI models."""
    total: int = Field(..., description="Total registered models count")
    default_model: str = Field(..., description="ID of the default model")
    models: List[MusicModelInfo] = Field(..., description="List of registered models")


class RecommendedSong(BaseModel):
    """Ranked song recommendation item with metadata."""
    rank: int = Field(..., ge=1, description="Rank position in Top-K list (1-indexed)")
    song_id: str = Field(..., description="Million Song Dataset song identifier")
    title: str = Field(..., description="Song track title")
    artist_name: str = Field(..., description="Artist or performer name")
    track_id: Optional[str] = Field(None, description="Echo Nest / MSD track identifier")
    score: float = Field(..., description="Model predicted affinity score or probability")


class MusicRecommendationResponse(BaseModel):
    """Standardized response contract for song recommendations."""
    user_id: str = Field(..., description="Target user identifier")
    model_requested: str = Field(..., description="Model ID requested by client")
    model_used: str = Field(..., description="Actual model ID used for inference")
    fallback_used: bool = Field(..., description="Whether a fallback model was invoked (e.g. cold start)")
    fallback_reason: Optional[str] = Field(None, description="Reason for fallback if applicable")
    top_k: int = Field(..., description="Requested Top-K limit")
    count: int = Field(..., description="Number of items returned")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    recommendations: List[RecommendedSong] = Field(default_factory=list, description="Ranked songs list")


class SongDetailResponse(BaseModel):
    """Catalog metadata and listening statistics for an individual song."""
    song_id: str = Field(..., description="Unique song identifier")
    track_id: Optional[str] = Field(None, description="MSD track identifier")
    title: str = Field(..., description="Song title")
    artist_name: str = Field(..., description="Artist name")
    total_play_count: Optional[int] = Field(None, description="Total listening count in training split")
    unique_listeners: Optional[int] = Field(None, description="Unique listeners count in training split")


class UserHistorySong(BaseModel):
    """A song previously listened to by user in training data."""
    song_id: str = Field(..., description="Song identifier")
    title: str = Field(..., description="Song title")
    artist_name: str = Field(..., description="Artist name")
    track_id: Optional[str] = Field(None, description="Track identifier")
    play_count: int = Field(..., ge=1, description="Listening frequency in training data")


class UserHistoryResponse(BaseModel):
    """Listening history of a user from training split."""
    user_id: str = Field(..., description="User identifier")
    total_plays: int = Field(..., description="Sum of play counts in training data")
    total_songs: int = Field(..., description="Number of distinct songs listened to")
    history: List[UserHistorySong] = Field(default_factory=list, description="List of listened songs")


class MusicHealthResponse(BaseModel):
    """Diagnostics and operational health of the MusicAI service layer."""
    status: str = Field(..., description="Operational status: healthy or degraded")
    service: str = Field(..., description="Service identifier string")
    domain: str = Field(default="music", description="Recommendation domain")
    catalog_songs: int = Field(..., description="Total songs in verified catalog")
    total_users: int = Field(..., description="Total unique users with training history")
    total_train_interactions: int = Field(..., description="Total user-song pairs in train data")
    available_models: int = Field(..., description="Count of models with checkpoints present")
    loaded_models: List[str] = Field(default_factory=list, description="Models currently loaded in memory")
    default_model: str = Field(..., description="Default model ID for inference")
