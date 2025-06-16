# scripts/test_api.py

"""Test API endpoints"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("Health Check:", response.json())

def test_segmentation():
    """Test segmentation endpoint"""
    customer_data = {
        "customer_id": "CUST_001234",
        "total_wagered": 2500.0,
        "avg_bet_per_session": 50.0,
        "loss_rate": 12.5,
        "total_sessions": 50,
        "days_since_last_visit": 7,
        "loss_chasing_score": 0.15
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/segment",
        json=customer_data
    )
    print("Segmentation Result:", response.json())

def test_promotion():
    """Test promotion endpoint"""
    customer_data = {
        "customer_id": "CUST_001234",
        "total_wagered": 2500.0,
        "avg_bet_per_session": 50.0,
        "loss_rate": 12.5,
        "total_sessions": 50,
        "days_since_last_visit": 7
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/promotion",
        json=customer_data
    )
    print("Promotion Result:", response.json())

if __name__ == "__main__":
    print("Testing Casino Analytics API...")
    test_health()
    test_segmentation()
    test_promotion()