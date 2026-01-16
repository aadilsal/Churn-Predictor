"""Data validation schemas using Pydantic."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Gender(str, Enum):
    """Gender categories."""

    MALE = "Male"
    FEMALE = "Female"


class YesNo(str, Enum):
    """Yes/No categorical values."""

    YES = "Yes"
    NO = "No"


class InternetService(str, Enum):
    """Internet service types."""

    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"


class Contract(str, Enum):
    """Contract types."""

    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class PaymentMethod(str, Enum):
    """Payment methods."""

    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"


class TelcoCustomerRecord(BaseModel):
    """Validation schema for a single customer record."""

    customerID: str = Field(..., description="Unique customer identifier")

    # Demographics
    gender: Gender
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: YesNo
    Dependents: YesNo

    # Account information
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: YesNo
    MultipleLines: str  # Yes, No, or "No phone service"

    # Internet services
    InternetService: InternetService
    OnlineSecurity: str  # Yes, No, or "No internet service"
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    # Billing
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges in dollars")
    TotalCharges: str  # Can be empty string or numeric

    # Target variable
    Churn: YesNo

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v: str) -> str:
        """Validate TotalCharges field.

        Args:
            v: TotalCharges value

        Returns:
            Validated value

        Raises:
            ValueError: If value is invalid
        """
        if v.strip() == "":
            return v  # Allow empty strings (will be handled in preprocessing)

        try:
            float(v)
            return v
        except ValueError:
            raise ValueError(f"TotalCharges must be numeric or empty, got: {v}")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class DataQualityReport(BaseModel):
    """Data quality report schema."""

    total_records: int
    valid_records: int
    invalid_records: int
    missing_values: dict
    duplicate_records: int
    validation_errors: list
    data_types_correct: bool
    schema_version: str = "1.0"
