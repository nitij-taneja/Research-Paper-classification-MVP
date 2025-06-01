import os
import shutil
import logging
import subprocess
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("deployment")

class DeploymentManager:
    """Class to manage deployment of the Research Papers Classification system."""
    
    def __init__(self, project_path=None, deployment_path=None):
        """Initialize the deployment manager."""
        self.project_path = project_path or "/home/ubuntu/optimized_project/Research_Paper_Classification"
        self.deployment_path = deployment_path or "/home/ubuntu/optimized_project/deployment"
        
        # Create deployment directory if it doesn't exist
        os.makedirs(self.deployment_path, exist_ok=True)
    
    def prepare_deployment_package(self):
        """Prepare the deployment package."""
        logger.info("Preparing deployment package...")
        
        # Create necessary directories
        os.makedirs(os.path.join(self.deployment_path, "src"), exist_ok=True)
        os.makedirs(os.path.join(self.deployment_path, "frontend"), exist_ok=True)
        os.makedirs(os.path.join(self.deployment_path, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.deployment_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.deployment_path, "results"), exist_ok=True)
        
        # Copy source files
        logger.info("Copying source files...")
        src_files = [
            "app.py",
            "pdf_processing.py",
            "preprocessing.py",
            "metadata_generation.py",
            "classification.py",
            "journal_recommendation.py",
            "pipeline.py",
            "model_improvement.py",
            "enhanced_pipeline.py",
            "data_augmentation.py"
        ]
        
        for file in src_files:
            source = os.path.join(self.project_path, "src", file)
            destination = os.path.join(self.deployment_path, "src", file)
            if os.path.exists(source):
                shutil.copy2(source, destination)
                logger.info(f"Copied {file}")
        
        # Copy frontend files
        logger.info("Copying frontend files...")
        frontend_dirs = ["templates", "static"]
        
        for dir_name in frontend_dirs:
            source_dir = os.path.join(self.project_path, "frontend", dir_name)
            dest_dir = os.path.join(self.deployment_path, "frontend", dir_name)
            if os.path.exists(source_dir):
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                shutil.copytree(source_dir, dest_dir)
                logger.info(f"Copied {dir_name} directory")
        
        # Copy setup script
        logger.info("Copying setup script...")
        setup_script = os.path.join(self.project_path, "setup.sh")
        if os.path.exists(setup_script):
            shutil.copy2(setup_script, os.path.join(self.deployment_path, "setup.sh"))
            logger.info("Copied setup.sh")
        
        # Copy documentation
        logger.info("Copying documentation...")
        docs = [
            "documentation.md",
            "README.md"
        ]
        
        for doc in docs:
            source = os.path.join(self.project_path, doc)
            destination = os.path.join(self.deployment_path, doc)
            if os.path.exists(source):
                shutil.copy2(source, destination)
                logger.info(f"Copied {doc}")
        
        # Create requirements.txt
        logger.info("Creating requirements.txt...")
        requirements = [
            "flask==2.1.1",
            "flask-cors==3.0.10",
            "PyPDF2==2.4.0",
            "pdfplumber==0.7.0",
            "pymupdf==1.19.6",
            "pytesseract==0.3.9",
            "nltk==3.7",
            "spacy==3.2.4",
            "scikit-learn==1.0.2",
            "pandas==1.4.2",
            "numpy==1.22.3",
            "matplotlib==3.5.1",
            "seaborn==0.11.2",
            "gensim==4.1.2",
            "textstat==0.7.3",
            "sentence-transformers==2.2.0",
            "tqdm==4.64.0",
            "shap==0.40.0",
            "werkzeug==2.1.1"
        ]
        
        with open(os.path.join(self.deployment_path, "requirements.txt"), "w") as f:
            f.write("\n".join(requirements))
        
        logger.info("Created requirements.txt")
        
        # Create README.md if it doesn't exist
        if not os.path.exists(os.path.join(self.deployment_path, "README.md")):
            logger.info("Creating README.md...")
            readme_content = """# Research Papers Classification

A machine learning system for evaluating research papers and recommending suitable journals.

## Features

- PDF processing and text extraction
- Advanced machine learning classification
- Journal recommendation
- Interactive web interface

## Setup

1. Run the setup script:
   ```
   bash setup.sh
   ```

2. Start the application:
   ```
   source venv/bin/activate
   python src/app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Documentation

See `documentation.md` for detailed documentation.
"""
            with open(os.path.join(self.deployment_path, "README.md"), "w") as f:
                f.write(readme_content)
            
            logger.info("Created README.md")
        
        # Create Docker files
        logger.info("Creating Docker files...")
        
        # Dockerfile
        dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    tesseract-ocr \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy application files
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY models/ ./models/
COPY results/ ./results/

# Create necessary directories
RUN mkdir -p data/raw/papers data/raw/reference/publishable data/raw/reference/non_publishable data/processed

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "src/app.py"]
"""
        
        with open(os.path.join(self.deployment_path, "Dockerfile"), "w") as f:
            f.write(dockerfile_content)
        
        logger.info("Created Dockerfile")
        
        # docker-compose.yml
        docker_compose_content = """version: '3'

services:
  research-papers-classification:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
    restart: unless-stopped
"""
        
        with open(os.path.join(self.deployment_path, "docker-compose.yml"), "w") as f:
            f.write(docker_compose_content)
        
        logger.info("Created docker-compose.yml")
        
        # Create deployment instructions
        logger.info("Creating deployment instructions...")
        
        deployment_instructions = """# Deployment Instructions

## Local Deployment

1. Set up the environment:
   ```
   bash setup.sh
   ```

2. Start the application:
   ```
   source venv/bin/activate
   python src/app.py
   ```

3. Access the application at http://localhost:5000

## Docker Deployment

1. Build and start the Docker container:
   ```
   docker-compose up -d
   ```

2. Access the application at http://localhost:5000

## Cloud Deployment Options

### Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')" && python -m spacy download en_core_web_sm`
   - Start Command: `python src/app.py`

### Heroku

1. Create a new Heroku app
2. Add the Python buildpack
3. Deploy using the Heroku CLI:
   ```
   heroku login
   heroku git:remote -a your-app-name
   git push heroku main
   ```

### AWS Elastic Beanstalk

1. Install the EB CLI
2. Initialize your EB environment:
   ```
   eb init -p python-3.8 research-papers-classification
   ```
3. Create and deploy the environment:
   ```
   eb create research-papers-env
   ```

## Environment Variables

The following environment variables can be set to customize the application:

- `PORT`: The port to run the application on (default: 5000)
- `DEBUG`: Set to "True" to enable debug mode (default: False)
- `MAX_CONTENT_LENGTH`: Maximum upload size in bytes (default: 16MB)
"""
        
        with open(os.path.join(self.deployment_path, "deployment_instructions.md"), "w") as f:
            f.write(deployment_instructions)
        
        logger.info("Created deployment instructions")
        
        # Create .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Logs
*.log

# Data
data/raw/papers/*
data/raw/reference/*
!data/raw/papers/.gitkeep
!data/raw/reference/.gitkeep
!data/raw/reference/publishable/.gitkeep
!data/raw/reference/non_publishable/.gitkeep

# Models
models/*
!models/.gitkeep

# Results
results/*
!results/.gitkeep

# IDE
.idea/
.vscode/
*.swp
*.swo
"""
        
        with open(os.path.join(self.deployment_path, ".gitignore"), "w") as f:
            f.write(gitignore_content)
        
        logger.info("Created .gitignore")
        
        # Create empty directories with .gitkeep
        empty_dirs = [
            "data/raw/papers",
            "data/raw/reference/publishable",
            "data/raw/reference/non_publishable",
            "data/processed",
            "models",
            "results"
        ]
        
        for dir_path in empty_dirs:
            full_path = os.path.join(self.deployment_path, dir_path)
            os.makedirs(full_path, exist_ok=True)
            with open(os.path.join(full_path, ".gitkeep"), "w") as f:
                pass
        
        logger.info("Created empty directories with .gitkeep")
        
        logger.info("Deployment package prepared successfully")
        return True
    
    def create_deployment_zip(self):
        """Create a ZIP file of the deployment package."""
        logger.info("Creating deployment ZIP file...")
        
        # Create ZIP file
        zip_path = f"{self.deployment_path}.zip"
        shutil.make_archive(self.deployment_path, 'zip', os.path.dirname(self.deployment_path), os.path.basename(self.deployment_path))
        
        logger.info(f"Deployment ZIP file created: {zip_path}")
        return zip_path
    
    def run_deployment(self):
        """Run the deployment process."""
        logger.info("Starting deployment process...")
        
        # Prepare deployment package
        if not self.prepare_deployment_package():
            logger.error("Failed to prepare deployment package")
            return False
        
        # Create deployment ZIP
        zip_path = self.create_deployment_zip()
        if not os.path.exists(zip_path):
            logger.error("Failed to create deployment ZIP")
            return False
        
        logger.info("Deployment process completed successfully")
        return zip_path

if __name__ == "__main__":
    # Create and run deployment manager
    manager = DeploymentManager()
    manager.run_deployment()
