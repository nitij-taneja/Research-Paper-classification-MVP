import os
import logging
import shutil
import json
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("system_test")

class SystemTester:
    """Class to perform comprehensive system testing."""
    
    def __init__(self, project_path=None, test_data_path=None):
        """Initialize the system tester."""
        self.project_path = project_path or "/home/ubuntu/optimized_project/Research_Paper_Classification"
        self.test_data_path = test_data_path or os.path.join(self.project_path, "test_data")
        
        # Create test data directory if it doesn't exist
        os.makedirs(self.test_data_path, exist_ok=True)
        
        # Create test results directory
        self.test_results_path = os.path.join(self.test_data_path, "results")
        os.makedirs(self.test_results_path, exist_ok=True)
        
        # Test report path
        self.test_report_path = os.path.join(self.project_path, "test_report.md")
    
    def test_pdf_processing(self):
        """Test PDF processing functionality."""
        logger.info("Testing PDF processing functionality...")
        
        try:
            # Import PDF processing module
            import sys
            sys.path.append(os.path.join(self.project_path, "src"))
            from pdf_processing import process_pdfs, process_pdfs_parallel
            
            # Test directory
            test_dir = os.path.join(self.test_data_path, "papers")
            output_file = os.path.join(self.test_results_path, "pdf_processing_test.csv")
            
            # Test sequential processing
            logger.info("Testing sequential PDF processing...")
            start_time = time.time()
            result_df = process_pdfs(test_dir, output_file)
            sequential_time = time.time() - start_time
            
            if result_df is not None and os.path.exists(output_file):
                logger.info(f"Sequential PDF processing test passed in {sequential_time:.2f} seconds")
            else:
                logger.error("Sequential PDF processing test failed")
                return False
            
            # Test parallel processing
            logger.info("Testing parallel PDF processing...")
            output_file_parallel = os.path.join(self.test_results_path, "pdf_processing_parallel_test.csv")
            start_time = time.time()
            result_df_parallel = process_pdfs_parallel(test_dir, output_file_parallel, max_workers=4)
            parallel_time = time.time() - start_time
            
            if result_df_parallel is not None and os.path.exists(output_file_parallel):
                logger.info(f"Parallel PDF processing test passed in {parallel_time:.2f} seconds")
            else:
                logger.error("Parallel PDF processing test failed")
                return False
            
            # Compare results
            if len(result_df) == len(result_df_parallel):
                logger.info("PDF processing results match")
            else:
                logger.warning("PDF processing results do not match in length")
            
            # Check speedup
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            logger.info(f"Parallel processing speedup: {speedup:.2f}x")
            
            return True
        except Exception as e:
            logger.error(f"Error in PDF processing test: {str(e)}")
            return False
    
    def test_preprocessing(self):
        """Test text preprocessing functionality."""
        logger.info("Testing text preprocessing functionality...")
        
        try:
            # Import preprocessing module
            import sys
            sys.path.append(os.path.join(self.project_path, "src"))
            from preprocessing import preprocess_text
            
            # Test input and output files
            input_file = os.path.join(self.test_results_path, "pdf_processing_test.csv")
            output_file = os.path.join(self.test_results_path, "preprocessing_test.csv")
            
            # Test preprocessing
            result_df = preprocess_text(input_file, output_file)
            
            if result_df is not None and os.path.exists(output_file):
                logger.info("Text preprocessing test passed")
                
                # Check if preprocessing reduced text length
                if 'abstract' in result_df.columns:
                    original_lengths = result_df['abstract'].str.len().mean()
                    logger.info(f"Average preprocessed abstract length: {original_lengths:.2f} characters")
                
                return True
            else:
                logger.error("Text preprocessing test failed")
                return False
        except Exception as e:
            logger.error(f"Error in preprocessing test: {str(e)}")
            return False
    
    def test_metadata_generation(self):
        """Test metadata generation functionality."""
        logger.info("Testing metadata generation functionality...")
        
        try:
            # Import metadata generation module
            import sys
            sys.path.append(os.path.join(self.project_path, "src"))
            from metadata_generation import generate_metadata
            
            # Test input and output files
            input_file = os.path.join(self.test_results_path, "preprocessing_test.csv")
            output_file = os.path.join(self.test_results_path, "metadata_test.csv")
            
            # Test metadata generation
            result_df = generate_metadata(input_file, output_file)
            
            if result_df is not None and os.path.exists(output_file):
                logger.info("Metadata generation test passed")
                
                # Check generated metadata columns
                metadata_columns = [col for col in result_df.columns if any(x in col for x in ['word_count', 'readability', 'sentiment', 'topic_diversity'])]
                logger.info(f"Generated metadata columns: {len(metadata_columns)}")
                
                return True
            else:
                logger.error("Metadata generation test failed")
                return False
        except Exception as e:
            logger.error(f"Error in metadata generation test: {str(e)}")
            return False
    
    def test_classification(self):
        """Test classification functionality."""
        logger.info("Testing classification functionality...")
        
        try:
            # Import classification module
            import sys
            sys.path.append(os.path.join(self.project_path, "src"))
            from classification import PublishabilityClassifier
            
            # Test files
            reference_file = os.path.join(self.test_results_path, "metadata_test.csv")
            papers_file = os.path.join(self.test_results_path, "metadata_test.csv")  # Using same file for testing
            model_save_path = os.path.join(self.test_results_path, "test_model.pkl")
            
            # Create classifier
            classifier = PublishabilityClassifier(reference_file, papers_file, model_save_path)
            
            # Test classification pipeline
            success = classifier.run_pipeline()
            
            if success and os.path.exists(model_save_path):
                logger.info("Classification test passed")
                return True
            else:
                logger.error("Classification test failed")
                return False
        except Exception as e:
            logger.error(f"Error in classification test: {str(e)}")
            return False
    
    def test_enhanced_model(self):
        """Test enhanced model functionality."""
        logger.info("Testing enhanced model functionality...")
        
        try:
            # Import enhanced model module
            import sys
            sys.path.append(os.path.join(self.project_path, "src"))
            from model_improvement import EnhancedClassifier
            
            # Test files
            reference_file = os.path.join(self.test_results_path, "metadata_test.csv")
            papers_file = os.path.join(self.test_results_path, "metadata_test.csv")  # Using same file for testing
            model_save_path = os.path.join(self.test_results_path, "enhanced_test_model.pkl")
            
            # Create enhanced classifier
            classifier = EnhancedClassifier(reference_file, papers_file, model_save_path)
            
            # Test enhanced classification pipeline
            success = classifier.run_pipeline()
            
            if success and os.path.exists(model_save_path):
                logger.info("Enhanced model test passed")
                return True
            else:
                logger.error("Enhanced model test failed")
                return False
        except Exception as e:
            logger.error(f"Error in enhanced model test: {str(e)}")
            return False
    
    def test_journal_recommendation(self):
        """Test journal recommendation functionality."""
        logger.info("Testing journal recommendation functionality...")
        
        try:
            # Import journal recommendation module
            import sys
            sys.path.append(os.path.join(self.project_path, "src"))
            from journal_recommendation import JournalRecommender
            
            # Test files
            reference_file = os.path.join(self.test_results_path, "metadata_test.csv")
            papers_file = os.path.join(self.test_results_path, "metadata_test.csv")
            publishability_file = os.path.join(self.project_path, "data/task1_prediction/predicted_publishability.csv")
            output_file = os.path.join(self.test_results_path, "journal_recommendation_test.csv")
            
            # Create recommender
            recommender = JournalRecommender(reference_file, papers_file, publishability_file, output_file)
            
            # Test recommendation pipeline
            success = recommender.run_pipeline()
            
            if success and os.path.exists(output_file):
                logger.info("Journal recommendation test passed")
                return True
            else:
                logger.error("Journal recommendation test failed")
                return False
        except Exception as e:
            logger.error(f"Error in journal recommendation test: {str(e)}")
            return False
    
    def test_pipeline_integration(self):
        """Test pipeline integration."""
        logger.info("Testing pipeline integration...")
        
        try:
            # Import pipeline module
            import sys
            sys.path.append(os.path.join(self.project_path, "src"))
            from pipeline import ResearchPaperPipeline
            
            # Configure pipeline for testing
            config = {
                "data_paths": {
                    "raw_papers": os.path.join(self.test_data_path, "papers"),
                    "raw_reference": os.path.join(self.test_data_path, "reference"),
                    "processed_dir": os.path.join(self.test_results_path, "processed"),
                    "results_dir": os.path.join(self.test_results_path, "results")
                },
                "processing_options": {
                    "use_parallel": True,
                    "max_workers": 2,
                    "generate_visualizations": True
                }
            }
            
            # Create pipeline
            pipeline = ResearchPaperPipeline(config)
            
            # Test pipeline
            success = pipeline.run_pipeline()
            
            if success:
                logger.info("Pipeline integration test passed")
                return True
            else:
                logger.error("Pipeline integration test failed")
                return False
        except Exception as e:
            logger.error(f"Error in pipeline integration test: {str(e)}")
            return False
    
    def test_enhanced_pipeline(self):
        """Test enhanced pipeline integration."""
        logger.info("Testing enhanced pipeline integration...")
        
        try:
            # Import enhanced pipeline module
            import sys
            sys.path.append(os.path.join(self.project_path, "src"))
            from enhanced_pipeline import EnhancedPipeline
            
            # Configure pipeline for testing
            config = {
                "data_paths": {
                    "raw_papers": os.path.join(self.test_data_path, "papers"),
                    "raw_reference": os.path.join(self.test_data_path, "reference"),
                    "processed_dir": os.path.join(self.test_results_path, "enhanced_processed"),
                    "results_dir": os.path.join(self.test_results_path, "enhanced_results")
                },
                "processing_options": {
                    "use_parallel": True,
                    "max_workers": 2,
                    "generate_visualizations": True,
                    "augment_data": False  # Skip data augmentation for testing
                },
                "model_options": {
                    "use_enhanced_model": True,
                    "compare_models": True
                }
            }
            
            # Create enhanced pipeline
            pipeline = EnhancedPipeline(config)
            
            # Test enhanced pipeline
            success = pipeline.run_pipeline()
            
            if success:
                logger.info("Enhanced pipeline integration test passed")
                return True
            else:
                logger.error("Enhanced pipeline integration test failed")
                return False
        except Exception as e:
            logger.error(f"Error in enhanced pipeline integration test: {str(e)}")
            return False
    
    def test_flask_app(self):
        """Test Flask app functionality."""
        logger.info("Testing Flask app functionality...")
        
        try:
            # Import Flask app
            import sys
            sys.path.append(os.path.join(self.project_path, "src"))
            from app import app
            
            # Create test client
            client = app.test_client()
            
            # Test home page
            response = client.get('/')
            if response.status_code == 200:
                logger.info("Flask app home page test passed")
            else:
                logger.error(f"Flask app home page test failed: {response.status_code}")
                return False
            
            # Test about page
            response = client.get('/about')
            if response.status_code == 200:
                logger.info("Flask app about page test passed")
            else:
                logger.error(f"Flask app about page test failed: {response.status_code}")
                return False
            
            # Test documentation page
            response = client.get('/documentation')
            if response.status_code == 200:
                logger.info("Flask app documentation page test passed")
            else:
                logger.error(f"Flask app documentation page test failed: {response.status_code}")
                return False
            
            logger.info("Flask app tests passed")
            return True
        except Exception as e:
            logger.error(f"Error in Flask app test: {str(e)}")
            return False
    
    def generate_test_report(self, test_results):
        """Generate a comprehensive test report."""
        logger.info("Generating test report...")
        
        report = """# Research Papers Classification - Test Report

## Overview

This report summarizes the results of comprehensive testing performed on the Research Papers Classification system. The tests cover individual components, integration between components, and end-to-end functionality.

## Test Results Summary

"""
        
        # Add test results summary
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report += f"- **Tests Passed:** {passed_tests}/{total_tests} ({pass_rate:.1f}%)\n"
        report += f"- **Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add detailed test results
        report += "## Detailed Test Results\n\n"
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            report += f"### {test_name}: {status}\n\n"
        
        # Add system recommendations
        report += "## System Recommendations\n\n"
        
        if all(test_results.values()):
            report += "All tests have passed successfully. The system is ready for deployment.\n\n"
        else:
            report += "Some tests have failed. Please address the issues before deployment.\n\n"
            
            # Add specific recommendations for failed tests
            failed_tests = [test_name for test_name, result in test_results.items() if not result]
            report += "### Failed Tests Recommendations\n\n"
            
            for test_name in failed_tests:
                if test_name == "PDF Processing":
                    report += "- **PDF Processing:** Check PDF extraction methods and error handling.\n"
                elif test_name == "Text Preprocessing":
                    report += "- **Text Preprocessing:** Verify text cleaning and tokenization functionality.\n"
                elif test_name == "Metadata Generation":
                    report += "- **Metadata Generation:** Ensure all metadata features are being calculated correctly.\n"
                elif test_name == "Classification":
                    report += "- **Classification:** Check model training and prediction pipeline.\n"
                elif test_name == "Enhanced Model":
                    report += "- **Enhanced Model:** Verify advanced feature engineering and model selection.\n"
                elif test_name == "Journal Recommendation":
                    report += "- **Journal Recommendation:** Check similarity calculation and recommendation logic.\n"
                elif test_name == "Pipeline Integration":
                    report += "- **Pipeline Integration:** Ensure all pipeline components are properly connected.\n"
                elif test_name == "Enhanced Pipeline":
                    report += "- **Enhanced Pipeline:** Verify enhanced pipeline configuration and execution.\n"
                elif test_name == "Flask App":
                    report += "- **Flask App:** Check Flask routes and template rendering.\n"
        
        # Write report to file
        with open(self.test_report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Test report generated: {self.test_report_path}")
        return self.test_report_path
    
    def run_all_tests(self):
        """Run all system tests."""
        logger.info("Starting system tests...")
        
        # Initialize test results
        test_results = {}
        
        # Test PDF processing
        test_results["PDF Processing"] = self.test_pdf_processing()
        
        # Test preprocessing
        test_results["Text Preprocessing"] = self.test_preprocessing()
        
        # Test metadata generation
        test_results["Metadata Generation"] = self.test_metadata_generation()
        
        # Test classification
        test_results["Classification"] = self.test_classification()
        
        # Test enhanced model
        test_results["Enhanced Model"] = self.test_enhanced_model()
        
        # Test journal recommendation
        test_results["Journal Recommendation"] = self.test_journal_recommendation()
        
        # Test pipeline integration
        test_results["Pipeline Integration"] = self.test_pipeline_integration()
        
        # Test enhanced pipeline
        test_results["Enhanced Pipeline"] = self.test_enhanced_pipeline()
        
        # Test Flask app
        test_results["Flask App"] = self.test_flask_app()
        
        # Generate test report
        report_path = self.generate_test_report(test_results)
        
        # Log summary
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        logger.info(f"System tests completed: {passed_tests}/{total_tests} tests passed")
        
        return test_results, report_path

if __name__ == "__main__":
    # Create and run system tester
    tester = SystemTester()
    tester.run_all_tests()
