import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("journal_recommendation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("journal_recommender")

class JournalRecommender:
    def __init__(self, reference_file=None, papers_file=None, publishability_file=None, output_file=None):
        """Initialize the journal recommender with file paths."""
        self.reference_file = reference_file or "data/processed/metadata_reference.csv"
        self.papers_file = papers_file or "data/processed/metadata_papers.csv"
        self.publishability_file = publishability_file or "data/task1_prediction/predicted_publishability.csv"
        self.output_file = output_file or "results/results.csv"
        self.embedding_model = None

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def load_data(self, progress_callback=None):
        """Load datasets with error handling."""
        try:
            if progress_callback:
                progress_callback("Loading reference data...")

            self.reference_data = pd.read_csv(self.reference_file)
            logger.info(f"Loaded reference data: {len(self.reference_data)} rows")

            # Rename columns for consistency if needed
            if 'Paper ID' in self.reference_data.columns:
                self.reference_data.rename(columns={"Paper ID": "Paper_ID"}, inplace=True)
            if 'abstract' in self.reference_data.columns:
                self.reference_data.rename(columns={"abstract": "Abstract"}, inplace=True)

            if progress_callback:
                progress_callback("Loading papers data...")

            self.papers_data = pd.read_csv(self.papers_file)
            logger.info(f"Loaded papers data: {len(self.papers_data)} rows")

            # Rename columns for consistency if needed
            if 'Paper ID' in self.papers_data.columns:
                self.papers_data.rename(columns={"Paper ID": "Paper_ID"}, inplace=True)
            if 'abstract' in self.papers_data.columns:
                self.papers_data.rename(columns={"abstract": "Abstract"}, inplace=True)

            if progress_callback:
                progress_callback("Loading publishability data...")

            self.publishability_data = pd.read_csv(self.publishability_file)
            logger.info(f"Loaded publishability data: {len(self.publishability_data)} rows")

            # Rename columns for consistency if needed
            if 'Paper ID' in self.publishability_data.columns:
                self.publishability_data.rename(columns={"Paper ID": "Paper_ID"}, inplace=True)
            if 'publishable' in self.publishability_data.columns:
                self.publishability_data.rename(columns={"publishable": "Publishable"}, inplace=True)

            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            if progress_callback:
                progress_callback(f"Error loading data: {str(e)}")
            return False

    def initialize_model(self, progress_callback=None):
        """Initialize the sentence transformer model."""
        try:
            if progress_callback:
                progress_callback("Initializing sentence transformer model...")

            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")

            # Initialize model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            logger.info("Sentence transformer model initialized")

            return True
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            if progress_callback:
                progress_callback(f"Error initializing model: {str(e)}")
            return False

    def compute_embeddings(self, progress_callback=None):
        """Compute embeddings for abstracts."""
        try:
            if progress_callback:
                progress_callback("Computing embeddings for reference papers...")

            # Compute embeddings for reference abstracts
            reference_abstracts = self.reference_data['Abstract'].fillna('').tolist()
            self.reference_embeddings = self.embedding_model.encode(
                reference_abstracts,
                show_progress_bar=True,
                convert_to_tensor=False
            )
            logger.info(f"Computed embeddings for {len(reference_abstracts)} reference papers")

            if progress_callback:
                progress_callback("Computing embeddings for papers...")

            # Compute embeddings for paper abstracts
            paper_abstracts = self.papers_data['Abstract'].fillna('').tolist()
            self.paper_embeddings = self.embedding_model.encode(
                paper_abstracts,
                show_progress_bar=True,
                convert_to_tensor=False
            )
            logger.info(f"Computed embeddings for {len(paper_abstracts)} papers")

            return True
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            if progress_callback:
                progress_callback(f"Error computing embeddings: {str(e)}")
            return False

    def assign_journal(self, ref_id):
        """Assign a journal based on reference ID."""
        journal_mapping = {
            "R006": "CVPR", "R007": "CVPR",
            "R008": "EMNLP", "R009": "EMNLP",
            "R010": "KDD", "R011": "KDD",
            "R012": "NEURIPS", "R013": "NEURIPS",
            "R014": "TMLR", "R015": "TMLR",
        }
        return journal_mapping.get(ref_id, "unknown")

    def generate_detailed_rationale(self, conference, top_references, similarities):
        """Generate detailed rationale for conference assignment."""
        if conference == "unknown":
            return "Not applicable"

        # Get the most similar paper from this conference
        conf_papers = []
        for i, (_, row) in enumerate(top_references.iterrows()):
            if row.get('Conference') == conference:
                conf_papers.append((i, similarities[i]))

        if not conf_papers:
            return f"The paper is recommended for {conference} based on overall thematic similarity."

        best_idx, best_sim = max(conf_papers, key=lambda x: x[1])
        best_paper = top_references.iloc[best_idx]

        # Generate detailed rationale
        rationale = f"The paper is recommended for {conference} with {best_sim:.2f} similarity score. "
        rationale += f"It shares methodological approaches with papers like '{best_paper['Paper_ID']}'. "

        # Add specific details based on conference
        conference_details = {
            "CVPR": "computer vision and pattern recognition techniques",
            "EMNLP": "empirical methods in natural language processing",
            "KDD": "knowledge discovery and data mining approaches",
            "NEURIPS": "neural information processing systems and machine learning",
            "TMLR": "theoretical machine learning research"
        }

        if conference in conference_details:
            rationale += f"The content aligns with {conference_details[conference]}."

        return rationale

    def recommend_journals(self, progress_callback=None):
        """Perform similarity matching and recommend journals."""
        try:
            if progress_callback:
                progress_callback("Performing similarity matching...")

            results = []

            # For each paper, find the best matching reference papers
            for i, (_, paper) in enumerate(tqdm(self.papers_data.iterrows(), total=len(self.papers_data))):
                paper_embedding = self.paper_embeddings[i]

                # Calculate similarity to all reference papers
                similarities = []
                for ref_embedding in self.reference_embeddings:
                    # Cosine similarity
                    similarity = np.dot(paper_embedding, ref_embedding) / (
                        np.linalg.norm(paper_embedding) * np.linalg.norm(ref_embedding)
                    )
                    similarities.append(similarity)

                # Get top 3 matches
                top_indices = np.argsort(similarities)[::-1][:3]
                top_similarities = [similarities[idx] for idx in top_indices]
                top_references = self.reference_data.iloc[top_indices]

                # Count weighted votes for each conference
                conference_scores = {}
                for j, (_, ref) in enumerate(top_references.iterrows()):
                    conf = self.assign_journal(ref['Paper_ID'])
                    if conf not in conference_scores:
                        conference_scores[conf] = 0
                    conference_scores[conf] += top_similarities[j]

                # Get the best conference
                if conference_scores:
                    best_conference = max(conference_scores.items(), key=lambda x: x[1])[0]
                else:
                    best_conference = "unknown"

                # Generate detailed rationale
                rationale = self.generate_detailed_rationale(
                    best_conference,
                    top_references,
                    top_similarities
                )

                results.append({
                    "Paper_ID": paper['Paper_ID'],
                    "Conference": best_conference,
                    "Rationale": rationale,
                    "conference_scores": conference_scores
                })

                # Update progress every 10 papers
                if progress_callback and i % 10 == 0:
                    progress_callback(f"Processed {i+1}/{len(self.papers_data)} papers")

            self.recommendation_results = results
            logger.info(f"Generated recommendations for {len(results)} papers")

            return True
        except Exception as e:
            logger.error(f"Error recommending journals: {str(e)}")
            if progress_callback:
                progress_callback(f"Error recommending journals: {str(e)}")
            return False

    def create_visualizations(self, progress_callback=None):
        """Create visualizations for journal recommendations."""
        try:
            if progress_callback:
                progress_callback("Creating visualizations...")

            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)

            # Count recommendations by conference
            conferences = [result['Conference'] for result in self.recommendation_results
                          if result['Conference'] != "unknown"]

            if not conferences:
                logger.warning("No valid conferences for visualization")
                return True

            # Create conference distribution plot
            plt.figure(figsize=(10, 6))
            sns.countplot(y=conferences)
            plt.title('Conference Recommendations Distribution')
            plt.xlabel('Count')
            plt.ylabel('Conference')
            plt.tight_layout()
            plt.savefig("results/conference_distribution.png")
            plt.close()

            logger.info("Visualizations saved to results/")

            return True
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            if progress_callback:
                progress_callback(f"Error creating visualizations: {str(e)}")
            return False

    def save_results(self, progress_callback=None):
        """Save recommendation results to CSV."""
        try:
            if progress_callback:
                progress_callback("Saving recommendation results...")

            # Merge with publishability data
            publishable_papers = self.publishability_data[self.publishability_data['Publishable'] == 1]['Paper_ID'].tolist()

            # Prepare final results
            final_results = []
            for paper_id in self.papers_data['Paper_ID']:
                # Find recommendation for this paper
                recommendation = next(
                    (r for r in self.recommendation_results if r['Paper_ID'] == paper_id),
                    {'Conference': 'na', 'Rationale': 'na'}
                )

                # Only include conference and rationale if paper is publishable
                if paper_id in publishable_papers:
                    conference = recommendation['Conference']
                    rationale = recommendation['Rationale']
                    publishable = 1
                else:
                    conference = 'na'
                    rationale = 'na'
                    publishable = 0

                final_results.append({
                    'Paper_ID': paper_id,
                    'Conference': conference,
                    'Rationale': rationale,
                    'Publishable': publishable
                })

            # Convert to DataFrame and save
            results_df = pd.DataFrame(final_results)
            results_df.to_csv(self.output_file, index=False)

            logger.info(f"Results saved to {self.output_file}")
            if progress_callback:
                progress_callback(f"Results saved to {self.output_file}")

            return True
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            if progress_callback:
                progress_callback(f"Error saving results: {str(e)}")
            return False

    def run_pipeline(self, progress_callback=None):
        """Run the complete journal recommendation pipeline."""
        # Load data
        if not self.load_data(progress_callback):
            return False

        # Initialize model
        if not self.initialize_model(progress_callback):
            return False

        # Compute embeddings
        if not self.compute_embeddings(progress_callback):
            return False

        # Recommend journals
        if not self.recommend_journals(progress_callback):
            return False

        # Create visualizations
        if not self.create_visualizations(progress_callback):
            return False

        # Save results
        if not self.save_results(progress_callback):
            return False

        logger.info("Journal recommendation pipeline completed successfully")
        if progress_callback:
            progress_callback("Journal recommendation pipeline completed successfully")

        return True

if __name__ == "__main__":
    # File paths
    reference_file = "data/processed/metadata_reference.csv"
    papers_file = "data/processed/metadata_papers.csv"
    publishability_file = "data/task1_prediction/predicted_publishability.csv"
    output_file = "results/results.csv"

    # Create and run recommender
    recommender = JournalRecommender(reference_file, papers_file, publishability_file, output_file)
    recommender.run_pipeline()
