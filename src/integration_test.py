import os
import logging
import shutil
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integration_test")

class IntegrationTester:
    """Class to test the integration between frontend and backend."""
    
    def __init__(self, app_path=None, test_data_path=None):
        """Initialize the integration tester."""
        self.app_path = app_path or "/home/ubuntu/optimized_project/Research_Paper_Classification/src/app.py"
        self.test_data_path = test_data_path or "/home/ubuntu/optimized_project/Research_Paper_Classification/test_data"
        
        # Create test data directory if it doesn't exist
        os.makedirs(self.test_data_path, exist_ok=True)
        
        # Create test results directory
        self.test_results_path = os.path.join(self.test_data_path, "results")
        os.makedirs(self.test_results_path, exist_ok=True)
    
    def prepare_test_environment(self):
        """Prepare the test environment."""
        logger.info("Preparing test environment...")
        
        # Create necessary directories
        os.makedirs(os.path.join(self.test_data_path, "papers"), exist_ok=True)
        os.makedirs(os.path.join(self.test_data_path, "reference"), exist_ok=True)
        
        # Copy sample papers if available
        sample_papers_dir = "/home/ubuntu/project/Research-Papers-Classification-main/Research_Paper_Classification/data/raw/papers"
        if os.path.exists(sample_papers_dir):
            for file in os.listdir(sample_papers_dir):
                if file.endswith(".pdf"):
                    source = os.path.join(sample_papers_dir, file)
                    destination = os.path.join(self.test_data_path, "papers", file)
                    shutil.copy2(source, destination)
                    logger.info(f"Copied test paper: {file}")
        
        # Copy sample reference papers if available
        sample_ref_dir = "/home/ubuntu/project/Research-Papers-Classification-main/Research_Paper_Classification/data/raw/reference"
        if os.path.exists(sample_ref_dir):
            for subdir, _, files in os.walk(sample_ref_dir):
                for file in files:
                    if file.endswith(".pdf"):
                        rel_dir = os.path.relpath(subdir, sample_ref_dir)
                        dest_dir = os.path.join(self.test_data_path, "reference", rel_dir)
                        os.makedirs(dest_dir, exist_ok=True)
                        source = os.path.join(subdir, file)
                        destination = os.path.join(dest_dir, file)
                        shutil.copy2(source, destination)
                        logger.info(f"Copied reference paper: {rel_dir}/{file}")
        
        logger.info("Test environment prepared successfully")
        return True
    
    def test_api_endpoints(self):
        """Test API endpoints."""
        logger.info("Testing API endpoints...")
        
        # Import Flask app
        import sys
        sys.path.append(os.path.dirname(self.app_path))
        from app import app
        
        # Create test client
        client = app.test_client()
        
        # Test home page
        logger.info("Testing home page...")
        response = client.get('/')
        if response.status_code == 200:
            logger.info("Home page test passed")
        else:
            logger.error(f"Home page test failed: {response.status_code}")
            return False
        
        # Test about page
        logger.info("Testing about page...")
        response = client.get('/about')
        if response.status_code == 200:
            logger.info("About page test passed")
        else:
            logger.error(f"About page test failed: {response.status_code}")
            return False
        
        # Test documentation page
        logger.info("Testing documentation page...")
        response = client.get('/documentation')
        if response.status_code == 200:
            logger.info("Documentation page test passed")
        else:
            logger.error(f"Documentation page test failed: {response.status_code}")
            return False
        
        # Test file upload endpoint
        logger.info("Testing file upload endpoint...")
        test_file_path = os.path.join(self.test_data_path, "papers")
        test_files = [f for f in os.listdir(test_file_path) if f.endswith('.pdf')]
        
        if not test_files:
            logger.warning("No test files found for upload test")
            return False
        
        test_file = os.path.join(test_file_path, test_files[0])
        with open(test_file, 'rb') as f:
            response = client.post(
                '/upload',
                data={'file': (f, test_files[0], 'application/pdf')}
            )
        
        if response.status_code == 200 and 'submission_id' in response.json:
            logger.info("File upload test passed")
            submission_id = response.json['submission_id']
            
            # Test status endpoint
            logger.info("Testing status endpoint...")
            response = client.get(f'/status/{submission_id}')
            if response.status_code == 200 and 'status' in response.json:
                logger.info("Status endpoint test passed")
            else:
                logger.error(f"Status endpoint test failed: {response.status_code}")
                return False
            
            # Wait for processing to complete (max 60 seconds)
            logger.info("Waiting for processing to complete...")
            for _ in range(20):
                response = client.get(f'/status/{submission_id}')
                if response.json.get('status') in ['completed', 'error']:
                    break
                time.sleep(3)
            
            # Test result endpoint
            logger.info("Testing result endpoint...")
            response = client.get(f'/result/{submission_id}')
            if response.status_code == 200:
                logger.info("Result endpoint test passed")
                
                # Save test results
                with open(os.path.join(self.test_results_path, f"{submission_id}_result.json"), 'w') as f:
                    json.dump(response.json, f, indent=2)
            else:
                logger.error(f"Result endpoint test failed: {response.status_code}")
                return False
            
            # Test visualizations endpoint
            logger.info("Testing visualizations endpoint...")
            response = client.get(f'/visualizations/{submission_id}')
            if response.status_code == 200:
                logger.info("Visualizations endpoint test passed")
                
                # Save test results
                with open(os.path.join(self.test_results_path, f"{submission_id}_visualizations.json"), 'w') as f:
                    json.dump(response.json, f, indent=2)
            else:
                logger.error(f"Visualizations endpoint test failed: {response.status_code}")
                return False
        else:
            logger.error(f"File upload test failed: {response.status_code}")
            return False
        
        logger.info("All API endpoint tests passed successfully")
        return True
    
    def test_frontend_backend_integration(self):
        """Test frontend-backend integration."""
        logger.info("Testing frontend-backend integration...")
        
        # Test API endpoints
        if not self.test_api_endpoints():
            logger.error("API endpoint tests failed")
            return False
        
        logger.info("Frontend-backend integration tests passed successfully")
        return True
    
    def run_all_tests(self):
        """Run all integration tests."""
        logger.info("Starting integration tests...")
        
        # Prepare test environment
        if not self.prepare_test_environment():
            logger.error("Failed to prepare test environment")
            return False
        
        # Test frontend-backend integration
        if not self.test_frontend_backend_integration():
            logger.error("Frontend-backend integration tests failed")
            return False
        
        logger.info("All integration tests passed successfully")
        return True

if __name__ == "__main__":
    # Create and run integration tester
    tester = IntegrationTester()
    tester.run_all_tests()
