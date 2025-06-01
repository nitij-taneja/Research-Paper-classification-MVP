import os
import logging
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_pipeline")

class EnhancedPipeline:
    """Main enhanced pipeline class to orchestrate the entire workflow with improved models."""

    def __init__(self, config=None):
        """Initialize the pipeline with configuration."""
        # Default configuration
        self.config = {
            "data_paths": {
                "raw_papers": "data/raw/papers/",
                "raw_reference": "data/raw/reference/",
                "processed_dir": "data/processed/",
                "results_dir": "results/"
            },
            "processing_options": {
                "use_parallel": True,
                "max_workers": 4,
                "generate_visualizations": True,
                "augment_data": True
            },
            "model_options": {
                "use_enhanced_model": True,
                "compare_models": True
            }
        }

        # Update with custom config if provided
        if config:
            self.config.update(config)

        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for the pipeline."""
        for _, path in self.config["data_paths"].items():
            os.makedirs(path, exist_ok=True)

        # Create subdirectories
        os.makedirs(os.path.join(self.config["data_paths"]["processed_dir"], "markdown_files"), exist_ok=True)
        os.makedirs(os.path.join(self.config["data_paths"]["raw_reference"], "publishable"), exist_ok=True)
        os.makedirs(os.path.join(self.config["data_paths"]["raw_reference"], "non_publishable"), exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("data/task1_prediction", exist_ok=True)

    def augment_data(self, progress_callback=None):
        """Augment the dataset with additional papers."""
        if not self.config["processing_options"]["augment_data"]:
            logger.info("Data augmentation skipped as per configuration")
            return True

        try:
            from data_augmentation import DataAugmenter

            if progress_callback:
                progress_callback("Starting data augmentation...")

            logger.info("Starting data augmentation...")

            # Create and run data augmenter
            augmenter = DataAugmenter()
            success_count = augmenter.download_additional_papers(progress_callback)

            if progress_callback:
                progress_callback(f"Data augmentation completed: {success_count} papers added")

            logger.info(f"Data augmentation completed: {success_count} papers added")
            return True
        except Exception as e:
            logger.error(f"Error in data augmentation: {str(e)}")
            if progress_callback:
                progress_callback(f"Error in data augmentation: {str(e)}")
            return False

    def process_pdfs(self, progress_callback=None):
        """Process PDF files to extract text."""
        from pdf_processing import process_pdfs_parallel, process_pdfs

        if progress_callback:
            progress_callback("Starting PDF processing...")

        logger.info("Starting PDF processing...")

        # Process papers
        papers_path = self.config["data_paths"]["raw_papers"]
        papers_output = os.path.join(self.config["data_paths"]["processed_dir"], "papers.csv")

        # Process reference papers
        reference_path = self.config["data_paths"]["raw_reference"]
        reference_output = os.path.join(self.config["data_paths"]["processed_dir"], "reference.csv")

        if self.config["processing_options"]["use_parallel"]:
            # Parallel processing
            papers_df = process_pdfs_parallel(
                papers_path,
                papers_output,
                progress_callback=progress_callback,
                max_workers=self.config["processing_options"]["max_workers"]
            )

            reference_df = process_pdfs_parallel(
                reference_path,
                reference_output,
                is_reference=True,
                progress_callback=progress_callback,
                max_workers=self.config["processing_options"]["max_workers"]
            )
        else:
            # Sequential processing
            papers_df = process_pdfs(papers_path, papers_output, progress_callback=progress_callback)
            reference_df = process_pdfs(reference_path, reference_output, is_reference=True, progress_callback=progress_callback)

        if progress_callback:
            progress_callback("PDF processing completed")

        logger.info("PDF processing completed")
        return papers_df, reference_df

    def preprocess_text(self, progress_callback=None):
        """Clean and preprocess extracted text."""
        from preprocessing import preprocess_text

        if progress_callback:
            progress_callback("Starting text preprocessing...")

        logger.info("Starting text preprocessing...")

        # Preprocess papers
        papers_input = os.path.join(self.config["data_paths"]["processed_dir"], "papers.csv")
        papers_output = os.path.join(self.config["data_paths"]["processed_dir"], "preprocessed_papers.csv")

        # Preprocess reference papers
        reference_input = os.path.join(self.config["data_paths"]["processed_dir"], "reference.csv")
        reference_output = os.path.join(self.config["data_paths"]["processed_dir"], "preprocessed_reference.csv")

        # Run preprocessing
        papers_df = preprocess_text(papers_input, papers_output, progress_callback=progress_callback)
        reference_df = preprocess_text(reference_input, reference_output, progress_callback=progress_callback)

        if progress_callback:
            progress_callback("Text preprocessing completed")

        logger.info("Text preprocessing completed")
        return papers_df, reference_df

    def generate_metadata(self, progress_callback=None):
        """Generate metadata for preprocessed papers."""
        from metadata_generation import generate_metadata

        if progress_callback:
            progress_callback("Starting metadata generation...")

        logger.info("Starting metadata generation...")

        # Define keywords for density calculation
        research_keywords = [
            'method', 'result', 'study', 'analysis', 'experiment',
            'data', 'model', 'algorithm', 'performance', 'evaluation'
        ]

        # Generate metadata for papers
        papers_input = os.path.join(self.config["data_paths"]["processed_dir"], "preprocessed_papers.csv")
        papers_output = os.path.join(self.config["data_paths"]["processed_dir"], "metadata_papers.csv")

        # Generate metadata for reference papers
        reference_input = os.path.join(self.config["data_paths"]["processed_dir"], "preprocessed_reference.csv")
        reference_output = os.path.join(self.config["data_paths"]["processed_dir"], "metadata_reference.csv")

        # Run metadata generation
        papers_df = generate_metadata(papers_input, papers_output, keywords=research_keywords, progress_callback=progress_callback)
        reference_df = generate_metadata(reference_input, reference_output, keywords=research_keywords, progress_callback=progress_callback)

        if progress_callback:
            progress_callback("Metadata generation completed")

        logger.info("Metadata generation completed")
        return papers_df, reference_df

    def train_and_evaluate_models(self, progress_callback=None):
        """Train and evaluate both standard and enhanced models."""
        if progress_callback:
            progress_callback("Starting model training and evaluation...")

        logger.info("Starting model training and evaluation...")

        # Train standard model
        from classification import PublishabilityClassifier

        if progress_callback:
            progress_callback("Training standard model...")

        logger.info("Training standard model...")

        # File paths
        reference_file = os.path.join(self.config["data_paths"]["processed_dir"], "metadata_reference.csv")
        papers_file = os.path.join(self.config["data_paths"]["processed_dir"], "metadata_papers.csv")
        model_save_path = "models/publishability_model.pkl"

        # Create and run classifier
        classifier = PublishabilityClassifier(reference_file, papers_file, model_save_path)
        standard_success = classifier.run_pipeline(progress_callback=progress_callback)

        # Train enhanced model if configured
        enhanced_success = False
        if self.config["model_options"]["use_enhanced_model"]:
            from model_improvement import EnhancedClassifier

            if progress_callback:
                progress_callback("Training enhanced model...")

            logger.info("Training enhanced model...")

            # Create and run enhanced classifier
            enhanced_model_path = "models/enhanced_model.pkl"
            enhanced_classifier = EnhancedClassifier(reference_file, papers_file, enhanced_model_path)
            enhanced_success = enhanced_classifier.run_pipeline(progress_callback=progress_callback)

        # Compare models if both were successful
        if standard_success and enhanced_success and self.config["model_options"]["compare_models"]:
            self.compare_model_results(progress_callback)

        if progress_callback:
            progress_callback("Model training and evaluation completed")

        logger.info("Model training and evaluation completed")
        return standard_success or enhanced_success

    def compare_model_results(self, progress_callback=None):
        """Compare results from standard and enhanced models."""
        try:
            if progress_callback:
                progress_callback("Comparing model results...")

            logger.info("Comparing model results...")

            # Load prediction results
            standard_predictions = pd.read_csv("data/task1_prediction/predicted_publishability.csv")
            enhanced_predictions = pd.read_csv("data/task1_prediction/enhanced_predicted_publishability.csv")

            # Rename columns for consistency
            if 'Paper ID' in standard_predictions.columns:
                standard_predictions.rename(columns={"Paper ID": "Paper_ID"}, inplace=True)
            if 'publishable' in standard_predictions.columns:
                standard_predictions.rename(columns={"publishable": "Publishable"}, inplace=True)

            if 'Paper ID' in enhanced_predictions.columns:
                enhanced_predictions.rename(columns={"Paper ID": "Paper_ID"}, inplace=True)
            if 'publishable' in enhanced_predictions.columns:
                enhanced_predictions.rename(columns={"publishable": "Publishable"}, inplace=True)

            # Merge predictions
            merged = pd.merge(
                standard_predictions,
                enhanced_predictions,
                on="Paper_ID",
                suffixes=('_standard', '_enhanced')
            )

            # Calculate agreement rate
            agreement = (merged['Publishable_standard'] == merged['Publishable_enhanced']).mean() * 100

            # Create comparison visualization
            plt.figure(figsize=(10, 6))

            # Count publishable papers by model
            standard_count = merged['Publishable_standard'].sum()
            enhanced_count = merged['Publishable_enhanced'].sum()

            # Create bar chart
            plt.bar(['Standard Model', 'Enhanced Model'], [standard_count, enhanced_count])
            plt.title('Publishable Papers by Model')
            plt.ylabel('Number of Publishable Papers')
            plt.savefig("results/model_comparison.png")
            plt.close()

            # Create confidence distribution plot
            plt.figure(figsize=(10, 6))
            sns.histplot(merged['confidence_standard'], color='blue', alpha=0.5, label='Standard Model')
            sns.histplot(merged['confidence_enhanced'], color='red', alpha=0.5, label='Enhanced Model')
            plt.title('Confidence Score Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig("../results/confidence_distribution.png")
            plt.close()

            # Log comparison results
            logger.info(f"Model agreement rate: {agreement:.2f}%")
            logger.info(f"Standard model publishable count: {standard_count}")
            logger.info(f"Enhanced model publishable count: {enhanced_count}")

            if progress_callback:
                progress_callback(f"Model comparison completed. Agreement rate: {agreement:.2f}%")

            return True
        except Exception as e:
            logger.error(f"Error comparing model results: {str(e)}")
            if progress_callback:
                progress_callback(f"Error comparing model results: {str(e)}")
            return False

    def recommend_journals(self, progress_callback=None):
        """Recommend journals for publishable papers."""
        from journal_recommendation import JournalRecommender

        if progress_callback:
            progress_callback("Starting journal recommendation...")

        logger.info("Starting journal recommendation...")

        # File paths
        reference_file = os.path.join(self.config["data_paths"]["processed_dir"], "metadata_reference.csv")
        papers_file = os.path.join(self.config["data_paths"]["processed_dir"], "metadata_papers.csv")

        # Use enhanced model predictions if available
        if self.config["model_options"]["use_enhanced_model"] and os.path.exists("data/task1_prediction/enhanced_predicted_publishability.csv"):
            publishability_file = "data/task1_prediction/enhanced_predicted_publishability.csv"
            logger.info("Using enhanced model predictions for journal recommendation")
        else:
            publishability_file = "data/task1_prediction/predicted_publishability.csv"
            logger.info("Using standard model predictions for journal recommendation")

        output_file = "results/results.csv"

        # Create and run recommender
        recommender = JournalRecommender(reference_file, papers_file, publishability_file, output_file)
        success = recommender.run_pipeline(progress_callback=progress_callback)

        if success:
            if progress_callback:
                progress_callback("Journal recommendation completed")

            logger.info("Journal recommendation completed")
            return True
        else:
            if progress_callback:
                progress_callback("Journal recommendation failed")

            logger.error("Journal recommendation failed")
            return False

    def run_pipeline(self, progress_callback=None):
        """Run the complete enhanced pipeline."""
        try:
            # Step 1: Augment data if configured
            if self.config["processing_options"]["augment_data"]:
                self.augment_data(progress_callback)

            # Step 2: Process PDFs
            self.process_pdfs(progress_callback)

            # Step 3: Preprocess text
            self.preprocess_text(progress_callback)

            # Step 4: Generate metadata
            self.generate_metadata(progress_callback)

            # Step 5: Train and evaluate models
            self.train_and_evaluate_models(progress_callback)

            # Step 6: Recommend journals
            self.recommend_journals(progress_callback)

            if progress_callback:
                progress_callback("Enhanced pipeline completed successfully")

            logger.info("Enhanced pipeline completed successfully")
            return True
        except Exception as e:
            error_msg = f"Enhanced pipeline failed: {str(e)}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            return False

if __name__ == "__main__":
    # Create and run enhanced pipeline
    pipeline = EnhancedPipeline()
    pipeline.run_pipeline()
