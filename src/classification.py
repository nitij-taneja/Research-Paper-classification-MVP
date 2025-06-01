import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("classifier")

class PublishabilityClassifier:
    def __init__(self, reference_file=None, papers_file=None, model_save_path=None):
        """Initialize the classifier with file paths."""
        self.reference_file = reference_file or "data/processed/metadata_reference.csv"
        self.papers_file = papers_file or "data/processed/metadata_papers.csv"
        self.model_save_path = model_save_path or "models/publishability_model.pkl"
        self.vectorizer = None
        self.model = None
        self.feature_names = None

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def load_data(self, progress_callback=None):
        """Load datasets with error handling."""
        try:
            if progress_callback:
                progress_callback("Loading reference data...")

            self.reference_data = pd.read_csv(self.reference_file)
            logger.info(f"Loaded reference data: {len(self.reference_data)} rows")

            if progress_callback:
                progress_callback("Loading papers data...")

            self.papers_data = pd.read_csv(self.papers_file)
            logger.info(f"Loaded papers data: {len(self.papers_data)} rows")

            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            if progress_callback:
                progress_callback(f"Error loading data: {str(e)}")
            return False

    def compute_similarity(self, text_1, text_2):
        """Compute text similarity using TF-IDF vectors."""
        if not self.vectorizer:
            logger.error("Vectorizer not initialized")
            return 0

        if not isinstance(text_1, str) or not isinstance(text_2, str):
            return 0

        if not text_1.strip() or not text_2.strip():
            return 0

        try:
            vec_1 = self.vectorizer.transform([text_1]).toarray()
            vec_2 = self.vectorizer.transform([text_2]).toarray()
            return cosine_similarity(vec_1, vec_2)[0][0]
        except Exception as e:
            logger.warning(f"Error computing similarity: {str(e)}")
            return 0

    def preprocess_data(self, progress_callback=None):
        """Preprocess data and initialize vectorizer."""
        if progress_callback:
            progress_callback("Initializing text vectorizer...")

        try:
            # Initialize and fit vectorizer
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.vectorizer.fit(self.reference_data['abstract'].fillna(''))
            logger.info("Vectorizer initialized and fitted")

            # Calculate similarity features for reference data
            if progress_callback:
                progress_callback("Calculating similarity features...")

            published_abstracts = ' '.join(self.reference_data[self.reference_data['Label'] == 1]['abstract'].fillna(''))
            unpublished_abstracts = ' '.join(self.reference_data[self.reference_data['Label'] == 0]['abstract'].fillna(''))

            self.reference_data['abstract_similarity_published'] = self.reference_data['abstract'].apply(
                lambda x: self.compute_similarity(x, published_abstracts)
            )

            self.reference_data['abstract_similarity_unpublished'] = self.reference_data['abstract'].apply(
                lambda x: self.compute_similarity(x, unpublished_abstracts)
            )

            logger.info("Similarity features calculated for reference data")
            return True
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            if progress_callback:
                progress_callback(f"Error preprocessing data: {str(e)}")
            return False

    def extract_advanced_features(self, row):
        """Extract advanced features for classification."""
        try:
            # Get reference data subsets
            published_refs = self.reference_data[self.reference_data['Label'] == 1]
            unpublished_refs = self.reference_data[self.reference_data['Label'] == 0]

            # Published and unpublished abstracts for similarity
            published_abstracts = ' '.join(published_refs['abstract'].fillna(''))
            unpublished_abstracts = ' '.join(unpublished_refs['abstract'].fillna(''))

            # Similarity features
            abstract_sim_published = self.compute_similarity(row['abstract'], published_abstracts)
            abstract_sim_unpublished = self.compute_similarity(row['abstract'], unpublished_abstracts)

            # Prevent division by zero
            similarity_ratio = abstract_sim_published / (abstract_sim_unpublished + 1e-6)

            # Word count difference features
            word_count_diff_published = abs(row.get('abstract_word_count', 0) - published_refs['abstract_word_count'].mean())
            word_count_diff_unpublished = abs(row.get('abstract_word_count', 0) - unpublished_refs['abstract_word_count'].mean())

            # Additional features
            features = {
                'abstract_sim_published': abstract_sim_published,
                'abstract_sim_unpublished': abstract_sim_unpublished,
                'similarity_ratio': similarity_ratio,
                'word_count_diff_published': word_count_diff_published,
                'word_count_diff_unpublished': word_count_diff_unpublished,
                'abstract_word_count': row.get('abstract_word_count', 0),
                'abstract_readability': row.get('abstract_readability', 0),
                'abstract_sentiment': row.get('abstract_sentiment', 0),
                'abstract_topic_diversity': row.get('abstract_topic_diversity', 0),
                'section_balance': row.get('section_balance', 0),
            }

            # Add section-specific features if available
            for section in ['introduction', 'methodology', 'results', 'conclusion']:
                word_count_key = f'{section}_word_count'
                if word_count_key in row:
                    features[word_count_key] = row[word_count_key]

            return pd.Series(features)
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            # Return default features to prevent training failure
            return pd.Series({
                'abstract_sim_published': 0,
                'abstract_sim_unpublished': 0,
                'similarity_ratio': 0,
                'word_count_diff_published': 0,
                'word_count_diff_unpublished': 0,
                'abstract_word_count': 0,
                'abstract_readability': 0,
                'abstract_sentiment': 0,
                'abstract_topic_diversity': 0,
                'section_balance': 0,
            })

    def prepare_training_data(self, progress_callback=None):
        """Prepare training data with feature engineering."""
        if progress_callback:
            progress_callback("Preparing training data...")

        try:
            # Apply feature engineering on reference data
            training_features = self.reference_data.apply(
                lambda row: self.extract_advanced_features(row), axis=1
            )

            # Store feature names for later use
            self.feature_names = training_features.columns.tolist()

            # Prepare X and y
            X = training_features
            y = self.reference_data['Label']

            # Split the data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            logger.info(f"Training data prepared: {len(X_train)} training samples, {len(X_val)} validation samples")

            return X_train, X_val, y_train, y_val
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            if progress_callback:
                progress_callback(f"Error preparing training data: {str(e)}")
            return None, None, None, None

    def train_model(self, X_train, y_train, X_val, y_val, progress_callback=None):
        """Train and evaluate the model with hyperparameter tuning."""
        if progress_callback:
            progress_callback("Training model with hyperparameter tuning...")

        try:
            # Define parameter grid for hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            # Create grid search
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                verbose=1
            )

            # Train model
            grid_search.fit(X_train, y_train)

            # Get best model
            self.model = grid_search.best_estimator_

            # Log best parameters
            logger.info(f"Best parameters: {grid_search.best_params_}")
            if progress_callback:
                progress_callback(f"Best parameters: {grid_search.best_params_}")

            # Evaluate model
            y_pred = self.model.predict(X_val)

            # Print classification report
            report = classification_report(y_val, y_pred)
            logger.info(f"Validation Classification Report:\n{report}")
            if progress_callback:
                progress_callback(f"Validation Classification Report:\n{report}")

            # Create confusion matrix visualization
            self.create_evaluation_visualizations(X_val, y_val, y_pred)

            # Save the model
            joblib.dump(self.model, model_save_path_pkl)
            joblib.dump(self.model, model_save_path_joblib)

            logger.info(f"Model saved to {model_save_path_pkl} and {model_save_path_joblib}")

            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            if progress_callback:
                progress_callback(f"Error training model: {str(e)}")
            return False

    def create_evaluation_visualizations(self, X_val, y_val, y_pred):
        """Create and save model evaluation visualizations."""
        try:
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)

            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Non-Publishable', 'Publishable'],
                        yticklabels=['Non-Publishable', 'Publishable'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig("results/confusion_matrix.png")
            plt.close()

            # Feature importance
            if self.model and self.feature_names:
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1]

                plt.figure(figsize=(10, 8))
                plt.title('Feature Importance')
                plt.bar(range(X_val.shape[1]), importances[indices], align='center')
                plt.xticks(range(X_val.shape[1]), [self.feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig("results/feature_importance.png")
                plt.close()

            logger.info("Evaluation visualizations saved to results/")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

    def predict_publishability(self, progress_callback=None):
        """Predict publishability for new papers."""
        if progress_callback:
            progress_callback("Predicting publishability for new papers...")

        try:
            # Apply the same feature engineering to new papers
            papers_features = self.papers_data.apply(
                lambda row: self.extract_advanced_features(row), axis=1
            )

            # Predict publishability
            predictions = self.model.predict(papers_features)
            probabilities = self.model.predict_proba(papers_features)[:, 1]  # Probability of being publishable

            # Add predictions to papers data
            self.papers_data['publishable'] = predictions
            self.papers_data['confidence'] = probabilities

            # Save predictions
            os.makedirs("data/task1_prediction", exist_ok=True)
            prediction_file = "data/task1_prediction/predicted_publishability.csv"
            self.papers_data[['Paper ID', 'publishable', 'confidence']].to_csv(prediction_file, index=False)

            logger.info(f"Predictions saved to {prediction_file}")
            if progress_callback:
                progress_callback(f"Predictions saved to {prediction_file}")

            return True
        except Exception as e:
            logger.error(f"Error predicting publishability: {str(e)}")
            if progress_callback:
                progress_callback(f"Error predicting publishability: {str(e)}")
            return False

    def run_pipeline(self, progress_callback=None):
        """Run the complete classification pipeline."""
        # Load data
        if not self.load_data(progress_callback):
            return False

        # Preprocess data
        if not self.preprocess_data(progress_callback):
            return False

        # Prepare training data
        X_train, X_val, y_train, y_val = self.prepare_training_data(progress_callback)
        if X_train is None:
            return False

        # Train model
        if not self.train_model(X_train, y_train, X_val, y_val, progress_callback):
            return False

        # Predict publishability
        if not self.predict_publishability(progress_callback):
            return False

        logger.info("Classification pipeline completed successfully")
        if progress_callback:
            progress_callback("Classification pipeline completed successfully")

        return True

if __name__ == "__main__":
    reference_file = "data/processed/metadata_reference.csv"
    papers_file = "data/processed/metadata_papers.csv"
    model_save_path_pkl = "models/publishability_model.pkl"
    model_save_path_joblib = model_save_path_pkl.replace(".pkl", ".joblib")

    classifier = PublishabilityClassifier(reference_file, papers_file, model_save_path_pkl)
    success = classifier.run_pipeline()  # run_pipeline returns True/False

    # Save joblib copy only if training was successful
    if success:
        import joblib
        joblib.dump(classifier.model, model_save_path_joblib)
        print(f"Model saved also as joblib at {model_save_path_joblib}")
