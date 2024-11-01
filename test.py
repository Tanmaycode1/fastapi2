import requests
import json
import logging
import sys
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# API endpoint
BASE_URL = "http://localhost:8000"
PARSE_ENDPOINT = f"{BASE_URL}/api/v1/parse"


def make_api_request(text: str) -> Dict[str, Any]:
    """Make API request with detailed error handling and logging"""
    try:
        logger.debug("Preparing to send request to API")
        logger.debug(f"Request length: {len(text)} characters")

        # First check if API is available
        health_check = requests.get(f"{BASE_URL}/health")
        logger.debug(f"Health check status: {health_check.status_code}")

        # Prepare the request
        payload = {
            "text": text,
            "include_cost_estimate": True
        }

        logger.debug("Sending POST request to API")
        response = requests.post(
            PARSE_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # 30 seconds timeout
        )

        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")

        try:
            response_data = response.json()
            logger.debug("Successfully parsed response JSON")
            return response_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response JSON: {str(e)}")
            logger.error(f"Response text: {response.text}")
            return {"error": "Invalid JSON response", "details": str(e)}

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return {"error": "Request failed", "details": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": "Unexpected error", "details": str(e)}


def test_simple_policy():
    """Test with a simple policy section first"""
    logger.info("Testing simple policy parsing")

    simple_text = """
    Policy Information:
    Drug: Taltz (ixekizumab)
    Policy Number: 00513

    Basic Criteria:
    1. Age: 6 years or older
    2. Diagnosis: Moderate to severe plaque psoriasis
    3. Required: Negative TB test
    """

    result = make_api_request(simple_text)
    logger.info("Simple policy test result:")
    logger.info(json.dumps(result, indent=2))


def test_full_policy():
    """Test with full policy text"""
    logger.info("Testing full policy parsing")

    policy_text = """
    
    """

    result = make_api_request(policy_text)
    logger.info("Full policy test result:")
    logger.info(json.dumps(result, indent=2))


def check_api_status():
    """Check if the API is running and responsive"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        logger.info(f"API Health Check Status: {response.status_code}")
        if response.status_code == 200:
            logger.info("API is healthy")
            return True
        else:
            logger.error(f"API health check failed: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to API: {str(e)}")
        return False


if __name__ == "__main__":
    logger.info("Starting API tests...")

    if not check_api_status():
        logger.error("API is not available. Please check if the server is running.")
        sys.exit(1)

    # Run tests
    logger.info("Running simple policy test...")
    test_simple_policy()

    logger.info("Running full policy test...")
    test_full_policy()

    logger.info("Testing completed.")