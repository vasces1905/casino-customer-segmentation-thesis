# src/api/main.py

"""
FastAPI Application for Casino Analytics
========================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import logging

from ..config.academic_config import ACADEMIC_METADATA
from ..data.db_connector import AcademicDBConnector
from ..models.segmentation import CustomerSegmentation
from ..models.promotion_rf import PromotionResponseModel
from ..models.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Casino Customer Segmentation API",
    description="Academic research API for casino customer analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # For frontend development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class CustomerFeatures(BaseModel):
    customer_id: str
    total_wagered: float
    avg_bet_per_session: float
    loss_rate: float
    total_sessions: int
    days_since_last_visit: int
    loss_chasing_score: Optional[float] = 0.0

class SegmentationResponse(BaseModel):
    customer_id: str
    segment: int
    segment_label: str
    confidence: float

class PromotionResponse(BaseModel):
    customer_id: str
    should_receive_promo: bool
    promo_probability: float
    recommended_promo_type: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    academic_metadata: Dict

# Initialize models (in production, load from model registry)
segmentation_model = None
promotion_model = None
model_registry = ModelRegistry()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global segmentation_model, promotion_model
    
    logger.info("Starting Casino Analytics API")
    logger.info(f"Ethics Approval: {ACADEMIC_METADATA['ethics_ref']}")
    
    # In production, load from model registry
    # For now, initialize empty models
    segmentation_model = CustomerSegmentation()
    promotion_model = PromotionResponseModel()

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        academic_metadata=ACADEMIC_METADATA
    )

@app.post("/api/v1/segment", response_model=SegmentationResponse)
async def predict_segment(features: CustomerFeatures):
    """
    Predict customer segment based on features.
    
    Academic note: Segments are interpretable business categories.
    """
    try:
        # Convert to format expected by model
        feature_dict = features.dict()
        
        # Mock prediction for now (replace with actual model)
        segment_id = 2  # Regular_Visitor
        segment_label = "Regular_Visitor"
        confidence = 0.85
        
        logger.info(f"Segmentation request for customer: {features.customer_id}")
        
        return SegmentationResponse(
            customer_id=features.customer_id,
            segment=segment_id,
            segment_label=segment_label,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/promotion", response_model=PromotionResponse)
async def predict_promotion(features: CustomerFeatures):
    """
    Predict promotion response likelihood.
    
    Returns recommendation for promotional targeting.
    """
    try:
        # Mock prediction for now
        promo_probability = 0.72
        should_receive = promo_probability > 0.5
        
        # Determine promo type based on segment
        if features.total_wagered > 5000:
            promo_type = "VIP_CASHBACK"
        elif features.days_since_last_visit > 30:
            promo_type = "REACTIVATION_BONUS"
        else:
            promo_type = "STANDARD_FREEPLAY"
        
        logger.info(f"Promotion prediction for customer: {features.customer_id}")
        
        return PromotionResponse(
            customer_id=features.customer_id,
            should_receive_promo=should_receive,
            promo_probability=promo_probability,
            recommended_promo_type=promo_type
        )
        
    except Exception as e:
        logger.error(f"Promotion prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models")
async def list_models():
    """List all registered models"""
    try:
        models = model_registry.list_models()
        return {"models": models.to_dict('records') if not models.empty else []}
    except Exception as e:
        logger.error(f"Model listing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/segments/summary")
async def get_segment_summary():
    """Get summary statistics for all segments"""
    # Mock data for demonstration
    return {
        "segments": [
            {
                "segment_id": 0,
                "label": "Casual_Player",
                "count": 1250,
                "avg_wagered": 500.50,
                "avg_loss_rate": 8.2
            },
            {
                "segment_id": 1,
                "label": "High_Roller",
                "count": 150,
                "avg_wagered": 15000.00,
                "avg_loss_rate": 12.5
            },
            {
                "segment_id": 2,
                "label": "Regular_Visitor",
                "count": 800,
                "avg_wagered": 2500.00,
                "avg_loss_rate": 10.1
            },
            {
                "segment_id": 3,
                "label": "At_Risk_Player",
                "count": 200,
                "avg_wagered": 3500.00,
                "avg_loss_rate": 18.5
            }
        ],
        "total_customers": 2400,
        "last_updated": datetime.now()
    }

# Academic compliance endpoint
@app.get("/api/v1/compliance")
async def get_compliance_info():
    """Return academic compliance information"""
    return {
        "ethics_approval": ACADEMIC_METADATA["ethics_ref"],
        "institution": ACADEMIC_METADATA["institution"],
        "gdpr_compliance": ACADEMIC_METADATA["gdpr_compliance"],
        "data_classification": ACADEMIC_METADATA["data_classification"],
        "academic_use_only": True
    }