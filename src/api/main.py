"""FastAPI inference service for churn prediction."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchCustomerResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
    ChurnDriver,
    CustomerInput,
    ErrorResponse,
    ExplanationResponse,
    HealthResponse,
    PredictionResponse,
    RecommendedAction,
    RiskLevel,
)
from src.utils.logging import logger


# ============================================
# APPLICATION SETUP
# ============================================

app = FastAPI(
    title="Churn Predictor API",
    description="""
    Production-grade API for customer churn prediction.
    
    ## Features
    - Real-time single customer predictions
    - Batch predictions for multiple customers
    - SHAP-based explanations
    - Business-friendly risk assessments
    
    ## Model Information
    - Model: XGBoost Classifier
    - Target: Customer Churn (binary)
    - Metrics: ROC-AUC ~0.84
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# MODEL LOADING
# ============================================

class ModelService:
    """Service for managing model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names: List[str] = []
        self.threshold: float = 0.54
        self.model_version: Optional[str] = None
        self._loaded = False
        
    def load(
        self,
        model_path: Path = Path("models/final_model.joblib"),
        preprocessor_path: Path = Path("models/feature_preprocessor.joblib"),
        feature_names_path: Path = Path("models/feature_names.json"),
        threshold_path: Path = Path("models/threshold.json"),
    ) -> bool:
        """Load model and artifacts."""
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            
            with open(feature_names_path, "r") as f:
                data = json.load(f)
                self.feature_names = data.get("feature_names", [])
                
            if threshold_path.exists():
                with open(threshold_path, "r") as f:
                    threshold_data = json.load(f)
                    self.threshold = threshold_data.get("f1_threshold", 0.54)
                    
            self._loaded = True
            self.model_version = "1.0.0"
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
            
    @property
    def is_loaded(self) -> bool:
        return self._loaded
        
    def preprocess(self, customers: List[Dict]) -> np.ndarray:
        """Preprocess customer data for inference."""
        df = pd.DataFrame(customers)
        return self.preprocessor.transform(df)
        
    def predict(self, X: np.ndarray) -> tuple:
        """Get predictions and probabilities."""
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions, probabilities
        
    def get_risk_level(self, probability: float) -> RiskLevel:
        """Classify probability into risk level."""
        if probability >= 0.7:
            return RiskLevel.HIGH
        elif probability >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


# Global model service
model_service = ModelService()


# ============================================
# STARTUP EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting Churn Predictor API...")
    if not model_service.load():
        logger.warning("Model not loaded - some endpoints will be unavailable")


# ============================================
# HEALTH ENDPOINTS
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
        model_version=model_service.model_version,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Churn Predictor API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


# ============================================
# PREDICTION ENDPOINTS
# ============================================

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(customer: CustomerInput):
    """
    Predict churn for a single customer.
    
    Returns probability, risk level, key drivers, and recommended actions.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Convert to dict for preprocessing
        customer_dict = customer.dict()
        customer_id = customer_dict.pop("customerID", None)
        
        # Preprocess
        X = model_service.preprocess([customer_dict])
        
        # Predict
        predictions, probabilities = model_service.predict(X)
        probability = float(probabilities[0])
        will_churn = bool(predictions[0])
        risk_level = model_service.get_risk_level(probability)
        
        # Generate drivers and recommendations
        drivers, recommendations = _generate_insights(customer_dict, probability, risk_level)
        
        # Create summary
        summary = _generate_summary(probability, risk_level, drivers)
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=probability,
            churn_probability_percent=f"{probability * 100:.1f}%",
            risk_level=risk_level,
            will_churn=will_churn,
            threshold_used=model_service.threshold,
            key_drivers=drivers,
            recommended_actions=recommendations,
            summary=summary,
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers.
    
    Supports up to 1000 customers per request.
    Returns aggregated summary and per-customer results.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        results = []
        customers_data = []
        
        for customer in request.customers:
            customer_dict = customer.dict()
            customer_id = customer_dict.pop("customerID", None)
            customers_data.append((customer_id, customer_dict))
            
        # Batch preprocess
        X = model_service.preprocess([c[1] for c in customers_data])
        
        # Batch predict
        predictions, probabilities = model_service.predict(X)
        
        # Process results
        high_risk = medium_risk = low_risk = churners = 0
        
        for i, (customer_id, _) in enumerate(customers_data):
            probability = float(probabilities[i])
            will_churn = bool(predictions[i])
            risk_level = model_service.get_risk_level(probability)
            
            if risk_level == RiskLevel.HIGH:
                high_risk += 1
            elif risk_level == RiskLevel.MEDIUM:
                medium_risk += 1
            else:
                low_risk += 1
                
            if will_churn:
                churners += 1
                
            results.append(BatchCustomerResult(
                customer_id=customer_id,
                churn_probability=probability,
                risk_level=risk_level,
                will_churn=will_churn,
            ))
            
        summary = BatchSummary(
            total_customers=len(results),
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            low_risk_count=low_risk,
            avg_churn_probability=float(probabilities.mean()),
            predicted_churners=churners,
        )
        
        return BatchPredictionResponse(
            summary=summary,
            predictions=results,
            failed_count=0,
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# EXPLANATION ENDPOINTS
# ============================================

@app.post("/explain", response_model=ExplanationResponse, tags=["Explanations"])
async def explain_prediction(customer: CustomerInput):
    """
    Get SHAP-based explanation for a customer.
    
    Returns detailed breakdown of risk factors and protective factors
    with recommended interventions.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        from src.explainability.shap_explainer import ShapExplainer
        from src.explainability.business_insights import BusinessInsightGenerator
        
        # Get prediction first
        customer_dict = customer.dict()
        customer_id = customer_dict.pop("customerID", None)
        
        X = model_service.preprocess([customer_dict])
        _, probabilities = model_service.predict(X)
        probability = float(probabilities[0])
        risk_level = model_service.get_risk_level(probability)
        
        # Get SHAP explanation
        explainer = ShapExplainer(
            model=model_service.model,
            feature_names=model_service.feature_names,
        )
        
        shap_explanation = explainer.explain_individual(X, index=0)
        
        # Business insights
        insight_gen = BusinessInsightGenerator(model_service.feature_names)
        business_exp = insight_gen.generate_customer_explanation(shap_explanation)
        
        # Format response
        risk_factors = [
            ChurnDriver(
                factor=f["factor"],
                impact="risk",
                contribution=f["contribution"]
            )
            for f in business_exp["risk_factors"]
        ]
        
        protective_factors = [
            ChurnDriver(
                factor=f["factor"],
                impact="protection",
                contribution=f["contribution"]
            )
            for f in business_exp["protective_factors"]
        ]
        
        recommendations = [
            RecommendedAction(
                priority=a["priority"],
                action=a["action"],
                rationale=a["recommendation"]
            )
            for a in business_exp["recommended_actions"]
        ]
        
        return ExplanationResponse(
            customer_id=customer_id,
            churn_probability=probability,
            base_probability=shap_explanation.get("base_probability", 0.5),
            risk_level=risk_level,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            narrative=business_exp["narrative_summary"],
            recommended_actions=recommendations,
            shap_values=shap_explanation.get("all_shap_values"),
        )
        
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="SHAP explanation not available. Install shap package."
        )
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# UTILITY FUNCTIONS
# ============================================

def _generate_insights(
    customer: Dict,
    probability: float,
    risk_level: RiskLevel,
) -> tuple:
    """Generate drivers and recommendations from customer data."""
    drivers = []
    recommendations = []
    
    # Contract type analysis
    contract = customer.get("Contract", "")
    if contract == "Month-to-month":
        drivers.append(ChurnDriver(
            factor="Month-to-Month Contract",
            impact="risk",
            contribution=0.25
        ))
        recommendations.append(RecommendedAction(
            priority="HIGH",
            action="Offer annual contract with discount",
            rationale="Contract type is the #1 churn predictor"
        ))
    elif contract in ["One year", "Two year"]:
        drivers.append(ChurnDriver(
            factor=f"{contract} Contract",
            impact="protection",
            contribution=0.20
        ))
        
    # Tenure analysis
    tenure = customer.get("tenure", 0)
    if tenure < 12:
        drivers.append(ChurnDriver(
            factor="Low Tenure (New Customer)",
            impact="risk",
            contribution=0.15
        ))
        recommendations.append(RecommendedAction(
            priority="MEDIUM",
            action="Engage with onboarding program",
            rationale="New customers need early engagement"
        ))
        
    # Internet service
    internet = customer.get("InternetService", "")
    if internet == "Fiber optic":
        drivers.append(ChurnDriver(
            factor="Fiber Optic Internet",
            impact="risk",
            contribution=0.12
        ))
        recommendations.append(RecommendedAction(
            priority="MEDIUM",
            action="Review fiber pricing competitiveness",
            rationale="Fiber customers have higher churn rates"
        ))
        
    # Payment method
    payment = customer.get("PaymentMethod", "")
    if payment == "Electronic check":
        drivers.append(ChurnDriver(
            factor="Electronic Check Payment",
            impact="risk",
            contribution=0.08
        ))
        recommendations.append(RecommendedAction(
            priority="LOW",
            action="Incentivize auto-pay enrollment",
            rationale="Electronic check has higher churn correlation"
        ))
        
    return drivers[:5], recommendations[:3]


def _generate_summary(
    probability: float,
    risk_level: RiskLevel,
    drivers: List[ChurnDriver],
) -> str:
    """Generate human-readable summary."""
    risk_text = {
        RiskLevel.LOW: "LOW risk of churning",
        RiskLevel.MEDIUM: "MEDIUM risk of churning",
        RiskLevel.HIGH: "HIGH risk of churning",
        RiskLevel.CRITICAL: "CRITICAL risk of churning",
    }
    
    summary = f"This customer has a {risk_text[risk_level]} ({probability*100:.1f}% probability)."
    
    risk_drivers = [d.factor for d in drivers if d.impact == "risk"]
    if risk_drivers:
        summary += f" Key risk factors: {', '.join(risk_drivers[:3])}."
        
    return summary


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Request failed", "detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# ============================================
# CLI RUNNER
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
