import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from tqdm import tqdm
import shap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_improvement.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_improver")

class EnhancedClassifier:
    def __init__(self, reference_file=None, papers_file=None, model_save_path=None):
        """Initialize the enhanced classifier with file paths."""
        self.reference_file = reference_file or "data/processed/metadata_reference.csv"
        self.papers_file = papers_file or "data/processed/metadata_papers.csv"
        self.model_save_path = model_save_path or "models/enhanced_model.pkl"
        self.feature_names = None
        self.best_model = None

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs("results", exist_ok=True)

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

    def extract_enhanced_features(self, df):
        """Extract enhanced features for model training."""
        try:
            features = pd.DataFrame()

            # Basic metadata features
            metadata_features = [
                'abstract_word_count', 'introduction_word_count', 'methodology_word_count',
                'conclusion_word_count', 'abstract_readability', 'introduction_readability',
                'methodology_readability', 'conclusion_readability', 'abstract_sentiment',
                'introduction_sentiment', 'methodology_sentiment', 'conclusion_sentiment',
                'abstract_topic_diversity', 'section_balance'
            ]

            # Add available metadata features
            for feature in metadata_features:
                if feature in df.columns:
                    features[feature] = df[feature]

            # Calculate section ratios
            total_words = df['abstract_word_count'] + df['introduction_word_count'] + \
                          df['methodology_word_count'] + df['conclusion_word_count']

            # Avoid division by zero
            total_words = total_words.replace(0, 1)

            features['abstract_ratio'] = df['abstract_word_count'] / total_words
            features['introduction_ratio'] = df['introduction_word_count'] / total_words
            features['methodology_ratio'] = df['methodology_word_count'] / total_words
            features['conclusion_ratio'] = df['conclusion_word_count'] / total_words

            # Calculate readability differences
            if 'abstract_readability' in df.columns and 'methodology_readability' in df.columns:
                features['readability_diff'] = abs(df['abstract_readability'] - df['methodology_readability'])

            # Calculate sentiment variance
            sentiment_cols = [col for col in df.columns if 'sentiment' in col]
            if sentiment_cols:
                sentiment_values = df[sentiment_cols].values
                features['sentiment_variance'] = np.var(sentiment_values, axis=1)

            # Store feature names
            self.feature_names = features.columns.tolist()

            return features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def prepare_data(self, progress_callback=None):
        """Prepare data for model training."""
        try:
            if progress_callback:
                progress_callback("Preparing training data...")

            # Extract features
            X = self.extract_enhanced_features(self.reference_data)
            y = self.reference_data['Label']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            logger.info(f"Prepared training data with {X_train.shape[1]} features")
            logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            if progress_callback:
                progress_callback(f"Error preparing data: {str(e)}")
            return None, None, None, None

    def train_models(self, X_train, y_train, X_test, y_test, progress_callback=None):
        """Train and evaluate multiple models."""
        try:
            if progress_callback:
                progress_callback("Training multiple models...")

            # Define models to try
            models = {
                'RandomForest': RandomForestClassifier(random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'SVM': SVC(probability=True, random_state=42)
            }

            # Define parameter grids
            param_grids = {
                'RandomForest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': [None, 'balanced']
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            }

            # Train and evaluate each model
            results = {}
            for model_name, model in models.items():
                if progress_callback:
                    progress_callback(f"Training {model_name}...")

                # Create pipeline with scaling for SVM
                if model_name == 'SVM':
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', model)
                    ])
                    grid_search = GridSearchCV(
                        pipeline,
                        {'classifier__' + key: value for key, value in param_grids[model_name].items()},
                        cv=5,
                        scoring='f1',
                        verbose=1
                    )
                else:
                    grid_search = GridSearchCV(
                        model,
                        param_grids[model_name],
                        cv=5,
                        scoring='f1',
                        verbose=1
                    )

                # Train model
                grid_search.fit(X_train, y_train)

                # Get best model
                best_model = grid_search.best_estimator_

                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]

                # Calculate metrics
                report = classification_report(y_test, y_pred, output_dict=True)

                # Calculate ROC curve and AUC
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                # Store results
                results[model_name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'accuracy': report['accuracy'],
                    'f1': report['weighted avg']['f1-score'],
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'roc_auc': roc_auc,
                    'fpr': fpr,
                    'tpr': tpr
                }

                logger.info(f"{model_name} - Best params: {grid_search.best_params_}")
                logger.info(f"{model_name} - F1 Score: {report['weighted avg']['f1-score']:.4f}")

            # Find best model
            best_model_name = max(results, key=lambda k: results[k]['f1'])
            self.best_model = results[best_model_name]['model']

            logger.info(f"Best model: {best_model_name} with F1 score: {results[best_model_name]['f1']:.4f}")
            if progress_callback:
                progress_callback(f"Best model: {best_model_name} with F1 score: {results[best_model_name]['f1']:.4f}")

            # Create visualizations
            self.create_model_comparison_visualizations(results, X_test, y_test)

            return results
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            if progress_callback:
                progress_callback(f"Error training models: {str(e)}")
            return None

    def create_model_comparison_visualizations(self, results, X_test, y_test):
        """Create visualizations comparing model performance."""
        try:
            # Create ROC curve comparison
            plt.figure(figsize=(10, 8))
            for model_name, result in results.items():
                plt.plot(
                    result['fpr'],
                    result['tpr'],
                    label=f"{model_name} (AUC = {result['roc_auc']:.2f})"
                )

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison')
            plt.legend(loc="lower right")
            plt.savefig("results/roc_curve_comparison.png")
            plt.close()

            # Create bar chart of metrics
            metrics = ['accuracy', 'f1', 'precision', 'recall']
            model_names = list(results.keys())

            plt.figure(figsize=(12, 8))
            x = np.arange(len(model_names))
            width = 0.2

            for i, metric in enumerate(metrics):
                values = [results[model]['f1'] if metric == 'f1' else results[model][metric] for model in model_names]
                plt.bar(x + i*width, values, width, label=metric)

            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x + width*1.5, model_names)
            plt.legend(loc='lower right')
            plt.savefig("results/model_performance_comparison.png")
            plt.close()

            # Create SHAP visualization for best model
            if self.best_model is not None:
                try:
                    # Get the actual model from pipeline if needed
                    if hasattr(self.best_model, 'named_steps') and 'classifier' in self.best_model.named_steps:
                        model_for_shap = self.best_model.named_steps['classifier']
                        # Need to transform X_test with the scaler
                        X_test_transformed = self.best_model.named_steps['scaler'].transform(X_test)
                    else:
                        model_for_shap = self.best_model
                        X_test_transformed = X_test

                    # Only create SHAP plots for tree-based models
                    if isinstance(model_for_shap, (RandomForestClassifier, GradientBoostingClassifier)):
                        explainer = shap.TreeExplainer(model_for_shap)
                        shap_values = explainer.shap_values(X_test_transformed)

                        # Summary plot
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(
                            shap_values[1] if isinstance(shap_values, list) else shap_values,
                            X_test_transformed,
                            feature_names=self.feature_names,
                            show=False
                        )
                        plt.tight_layout()
                        plt.savefig("results/shap_summary.png")
                        plt.close()
                except Exception as shap_error:
                    logger.warning(f"Error creating SHAP visualization: {str(shap_error)}")

            logger.info("Model comparison visualizations created")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

    def save_model(self, progress_callback=None):
        """Save the best model."""
        try:
            if self.best_model is not None:
                joblib.dump(self.best_model, self.model_save_path)
                logger.info(f"Best model saved to {self.model_save_path}")
                if progress_callback:
                    progress_callback(f"Best model saved to {self.model_save_path}")
                return True
            else:
                logger.error("No model to save")
                if progress_callback:
                    progress_callback("No model to save")
                return False
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            if progress_callback:
                progress_callback(f"Error saving model: {str(e)}")
            return False

    def predict_with_enhanced_model(self, progress_callback=None):
        """Predict publishability using the enhanced model."""
        try:
            if progress_callback:
                progress_callback("Predicting with enhanced model...")

            # Extract features for papers
            X_papers = self.extract_enhanced_features(self.papers_data)

            # Predict publishability
            predictions = self.best_model.predict(X_papers)
            probabilities = self.best_model.predict_proba(X_papers)[:, 1]

            # Add predictions to papers data
            self.papers_data['publishable'] = predictions
            self.papers_data['confidence'] = probabilities

            # Save predictions
            os.makedirs("data/task1_prediction", exist_ok=True)
            prediction_file = "data/task1_prediction/enhanced_predicted_publishability.csv"
            self.papers_data[['Paper ID', 'publishable', 'confidence']].to_csv(prediction_file, index=False)

            logger.info(f"Enhanced predictions saved to {prediction_file}")
            if progress_callback:
                progress_callback(f"Enhanced predictions saved to {prediction_file}")

            return True
        except Exception as e:
            logger.error(f"Error predicting with enhanced model: {str(e)}")
            if progress_callback:
                progress_callback(f"Error predicting with enhanced model: {str(e)}")
            return False

    def run_pipeline(self, progress_callback=None):
        """Run the complete enhanced classification pipeline."""
        # Load data
        if not self.load_data(progress_callback):
            return False

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(progress_callback)
        if X_train is None:
            return False

        # Train models
        results = self.train_models(X_train, y_train, X_test, y_test, progress_callback)
        if results is None:
            return False

        # Save best model
        if not self.save_model(progress_callback):
            return False

        # Predict with enhanced model
        if not self.predict_with_enhanced_model(progress_callback):
            return False

        logger.info("Enhanced classification pipeline completed successfully")
        if progress_callback:
            progress_callback("Enhanced classification pipeline completed successfully")

        return True

if __name__ == "__main__":
    # File paths
    reference_file = "data/processed/metadata_reference.csv"
    papers_file = "data/processed/metadata_papers.csv"
    model_save_path = "models/enhanced_model.pkl"

    # Create and run enhanced classifier
    classifier = EnhancedClassifier(reference_file, papers_file, model_save_path)
    classifier.run_pipeline()
