import os
import logging
from tqdm import tqdm
import pandas as pd
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pipeline")

class ResearchPaperPipeline:
    """Main pipeline class to orchestrate the entire workflow."""

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
                "generate_visualizations": True
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

    def classify_publishability(self, progress_callback=None):
        """Classify papers as publishable or not."""
        from classification import PublishabilityClassifier

        if progress_callback:
            progress_callback("Starting publishability classification...")

        logger.info("Starting publishability classification...")

        # File paths
        reference_file = os.path.join(self.config["data_paths"]["processed_dir"], "metadata_reference.csv")
        papers_file = os.path.join(self.config["data_paths"]["processed_dir"], "metadata_papers.csv")
        model_save_path = "models/publishability_model.pkl"

        # Create and run classifier
        classifier = PublishabilityClassifier(reference_file, papers_file, model_save_path)
        success = classifier.run_pipeline(progress_callback=progress_callback)

        if success:
            if progress_callback:
                progress_callback("Publishability classification completed")

            logger.info("Publishability classification completed")
            return True
        else:
            if progress_callback:
                progress_callback("Publishability classification failed")

            logger.error("Publishability classification failed")
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
        publishability_file = "data/task1_prediction/predicted_publishability.csv"
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
        """Run the complete pipeline."""
        try:
            # Step 1: Process PDFs
            self.process_pdfs(progress_callback)

            # Step 2: Preprocess text
            self.preprocess_text(progress_callback)

            # Step 3: Generate metadata
            self.generate_metadata(progress_callback)

            # Step 4: Classify publishability
            self.classify_publishability(progress_callback)

            # Step 5: Recommend journals
            self.recommend_journals(progress_callback)

            if progress_callback:
                progress_callback("Pipeline completed successfully")

            logger.info("Pipeline completed successfully")
            return True
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            return False

if __name__ == "__main__":
    # Create and run pipeline
    pipeline = ResearchPaperPipeline()
    pipeline.run_pipeline()
