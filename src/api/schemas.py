"""Pydantic schemas for API input validation and response formatting."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ContractType(str, Enum):
    """Contract type enumeration."""
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class PaymentMethodType(str, Enum):
    """Payment method enumeration."""
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"


class InternetServiceType(str, Enum):
    """Internet service type."""
    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"


class YesNo(str, Enum):
    """Yes/No enumeration for service fields."""
    YES = "Yes"
    NO = "No"


class YesNoNoPhone(str, Enum):
    """Yes/No/No phone service enumeration."""
    YES = "Yes"
    NO = "No"
    NO_PHONE = "No phone service"


class YesNoNoInternet(str, Enum):
    """Yes/No/No internet service enumeration."""
    YES = "Yes"
    NO = "No"
    NO_INTERNET = "No internet service"


class Gender(str, Enum):
    """Gender enumeration."""
    MALE = "Male"
    FEMALE = "Female"


class RiskLevel(str, Enum):
    """Customer risk level."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# ============================================
# INPUT SCHEMAS
# ============================================

class CustomerInput(BaseModel):
    """Input schema for customer data.
    
    All fields match the Telco churn dataset schema.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }
    )
    
    customerID: Optional[str] = Field(None, description="Unique customer identifier")
    gender: Gender = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen flag (0/1)")
    Partner: YesNo = Field(..., description="Has partner")
    Dependents: YesNo = Field(..., description="Has dependents")
    tenure: int = Field(..., ge=0, le=100, description="Months with company")
    PhoneService: YesNo = Field(..., description="Has phone service")
    MultipleLines: YesNoNoPhone = Field(..., description="Has multiple lines")
    InternetService: InternetServiceType = Field(..., description="Internet service type")
    OnlineSecurity: YesNoNoInternet = Field(..., description="Has online security")
    OnlineBackup: YesNoNoInternet = Field(..., description="Has online backup")
    DeviceProtection: YesNoNoInternet = Field(..., description="Has device protection")
    TechSupport: YesNoNoInternet = Field(..., description="Has tech support")
    StreamingTV: YesNoNoInternet = Field(..., description="Has streaming TV")
    StreamingMovies: YesNoNoInternet = Field(..., description="Has streaming movies")
    Contract: ContractType = Field(..., description="Contract type")
    PaperlessBilling: YesNo = Field(..., description="Uses paperless billing")
    PaymentMethod: PaymentMethodType = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, le=200, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")
    
    @field_validator("TotalCharges", mode="before")
    @classmethod
    def parse_total_charges(cls, v):
        """Handle empty string TotalCharges."""
        if v == "" or v == " ":
            return 0.0
        return float(v)


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    customers: List[CustomerInput] = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="List of customer records"
    )
    include_explanations: bool = Field(
        False,
        description="Include SHAP explanations in response"
    )


# ============================================
# OUTPUT SCHEMAS
# ============================================

class ChurnDriver(BaseModel):
    """Individual churn driver."""
    factor: str = Field(..., description="Factor name in business language")
    impact: str = Field(..., description="'risk' or 'protection'")
    contribution: float = Field(..., description="Relative contribution")


class RecommendedAction(BaseModel):
    """Recommended intervention action."""
    priority: str = Field(..., description="HIGH/MEDIUM/LOW")
    action: str = Field(..., description="Recommended action")
    rationale: str = Field(..., description="Why this action is recommended")


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "7590-VHVEG",
                "churn_probability": 0.72,
                "churn_probability_percent": "72.0%",
                "risk_level": "High",
                "will_churn": True,
                "threshold_used": 0.54,
                "key_drivers": [
                    {"factor": "Month-to-Month Contract", "impact": "risk", "contribution": 0.25},
                    {"factor": "Low Tenure", "impact": "risk", "contribution": 0.18}
                ],
                "recommended_actions": [
                    {"priority": "HIGH", "action": "Offer annual contract discount", 
                     "rationale": "Contract type is primary churn driver"}
                ],
                "summary": "This customer has a HIGH risk of churning (72.0% probability). Key risk factors: Month-to-Month Contract, Low Tenure."
            }
        }
    )
    
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    churn_probability: float = Field(..., ge=0, le=1, description="Churn probability")
    churn_probability_percent: str = Field(..., description="Probability as percentage")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    will_churn: bool = Field(..., description="Prediction above threshold")
    threshold_used: float = Field(..., description="Classification threshold")
    
    # Business insights
    key_drivers: List[ChurnDriver] = Field(default=[], description="Top churn drivers")
    recommended_actions: List[RecommendedAction] = Field(
        default=[], 
        description="Recommended interventions"
    )
    summary: str = Field(..., description="Human-readable summary")


class BatchCustomerResult(BaseModel):
    """Result for a single customer in batch."""
    customer_id: Optional[str] = None
    churn_probability: float
    risk_level: RiskLevel
    will_churn: bool
    error: Optional[str] = None


class BatchSummary(BaseModel):
    """Summary statistics for batch prediction."""
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_churn_probability: float
    predicted_churners: int


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary": {
                    "total_customers": 100,
                    "high_risk_count": 25,
                    "medium_risk_count": 35,
                    "low_risk_count": 40,
                    "avg_churn_probability": 0.42,
                    "predicted_churners": 45
                },
                "predictions": [
                    {"customer_id": "001", "churn_probability": 0.72, 
                     "risk_level": "High", "will_churn": True}
                ],
                "failed_count": 0
            }
        }
    )
    
    summary: BatchSummary
    predictions: List[BatchCustomerResult]
    failed_count: int = Field(0, description="Number of failed predictions")


class ExplanationResponse(BaseModel):
    """Response schema for SHAP explanations."""
    customer_id: Optional[str] = None
    churn_probability: float
    base_probability: float
    risk_level: RiskLevel
    
    # Detailed SHAP breakdown
    risk_factors: List[ChurnDriver]
    protective_factors: List[ChurnDriver]
    
    # Business narrative
    narrative: str
    recommended_actions: List[RecommendedAction]
    
    # Raw SHAP values (optional)
    shap_values: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    timestamp: str


class ValidationErrorDetail(BaseModel):
    """Validation error details."""
    field: str
    message: str
    value: Any = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: str
    validation_errors: Optional[List[ValidationErrorDetail]] = None
