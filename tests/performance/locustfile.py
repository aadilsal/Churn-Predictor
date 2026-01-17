"""Locust load testing configuration for churn prediction API.

Run with:
    locust -f locustfile.py --host=http://localhost:8000

Web UI available at http://localhost:8089
"""

import random
from locust import HttpUser, task, between


# Sample customer data templates
HIGH_RISK_CUSTOMER = {
    "customerID": "LOAD-TEST-HIGH",
    "gender": "Female",
    "SeniorCitizen": "Yes",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.00,
    "TotalCharges": 95.00,
}

LOW_RISK_CUSTOMER = {
    "customerID": "LOAD-TEST-LOW",
    "gender": "Male",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 65.00,
    "TotalCharges": 3900.00,
}


def generate_random_customer():
    """Generate a random customer for load testing."""
    return {
        "customerID": f"LOAD-{random.randint(1, 100000):06d}",
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.choice(["Yes", "No"]),
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["Yes", "No"]),
        "tenure": random.randint(0, 72),
        "PhoneService": random.choice(["Yes", "No"]),
        "MultipleLines": random.choice(["Yes", "No", "No phone service"]),
        "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": random.choice(["Yes", "No", "No internet service"]),
        "OnlineBackup": random.choice(["Yes", "No", "No internet service"]),
        "DeviceProtection": random.choice(["Yes", "No", "No internet service"]),
        "TechSupport": random.choice(["Yes", "No", "No internet service"]),
        "StreamingTV": random.choice(["Yes", "No", "No internet service"]),
        "StreamingMovies": random.choice(["Yes", "No", "No internet service"]),
        "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": random.choice(["Yes", "No"]),
        "PaymentMethod": random.choice([
            "Electronic check",
            "Mailed check", 
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]),
        "MonthlyCharges": round(random.uniform(18.0, 118.0), 2),
        "TotalCharges": round(random.uniform(0, 8000.0), 2),
    }


class ChurnPredictionUser(HttpUser):
    """Simulated user making API requests."""
    
    wait_time = between(0.5, 2)  # Wait 0.5-2 seconds between requests
    
    @task(5)
    def health_check(self):
        """Check API health (most frequent task)."""
        self.client.get("/health")
    
    @task(10)
    def single_prediction(self):
        """Make single prediction (common task)."""
        customer = generate_random_customer()
        self.client.post("/predict", json=customer)
    
    @task(3)
    def high_risk_prediction(self):
        """Predict high-risk customer."""
        customer = HIGH_RISK_CUSTOMER.copy()
        customer["customerID"] = f"LOAD-HIGH-{random.randint(1, 10000)}"
        self.client.post("/predict", json=customer)
    
    @task(3)
    def low_risk_prediction(self):
        """Predict low-risk customer."""
        customer = LOW_RISK_CUSTOMER.copy()
        customer["customerID"] = f"LOAD-LOW-{random.randint(1, 10000)}"
        self.client.post("/predict", json=customer)
    
    @task(2)
    def explanation_request(self):
        """Get explanation for a customer."""
        customer = generate_random_customer()
        self.client.post("/explain", json=customer)
    
    @task(1)
    def small_batch_prediction(self):
        """Make small batch prediction (5 customers)."""
        customers = [generate_random_customer() for _ in range(5)]
        self.client.post("/predict/batch", json={"customers": customers})
    
    @task(1)
    def medium_batch_prediction(self):
        """Make medium batch prediction (20 customers).""" 
        customers = [generate_random_customer() for _ in range(20)]
        self.client.post("/predict/batch", json={"customers": customers})


class HeavyUser(HttpUser):
    """Simulated heavy user making frequent batch requests."""
    
    wait_time = between(1, 3)
    weight = 1  # Less common than regular users
    
    @task(1)
    def large_batch_prediction(self):
        """Make large batch prediction (100 customers)."""
        customers = [generate_random_customer() for _ in range(100)]
        self.client.post("/predict/batch", json={"customers": customers})
    
    @task(2)
    def rapid_single_predictions(self):
        """Make multiple rapid single predictions."""
        for _ in range(10):
            customer = generate_random_customer()
            self.client.post("/predict", json=customer)


# Load test scenarios for different conditions
# Use these with: locust -f locustfile.py --users 50 --spawn-rate 5

"""
Recommended test scenarios:

1. Baseline:
   locust -f locustfile.py --users 10 --spawn-rate 2 --run-time 1m

2. Normal load:
   locust -f locustfile.py --users 50 --spawn-rate 10 --run-time 5m

3. Peak load:
   locust -f locustfile.py --users 100 --spawn-rate 20 --run-time 5m

4. Stress test:
   locust -f locustfile.py --users 200 --spawn-rate 50 --run-time 10m

Performance targets:
- p95 response time < 500ms
- Error rate < 1%
- Throughput > 100 requests/second at normal load
"""
