# Phish Guard Prediction

Phish Guard Prediction is an end-to-end machine learning system designed to detect and classify phishing websites based on network, domain, and page structure features. The application leverages a modular Python design pattern to construct a robust MLOps pipeline comprising data ingestion from MongoDB, schema validation and drift detection, data transformation with KNN imputation, multi-model evaluation with grid search, and experiment tracking via MLflow and DagsHub.

The system is fully containerized and includes a FastAPI web service providing interfaces for both single-instance and batch predictions from CSV files. Continuous Integration, Continuous Delivery, and Continuous Deployment (CI/CD) pipelines are established using GitHub Actions to automatically test, build, and deploy the application to AWS.

---

## Technical Architecture

The application is structured around a decoupled workflow:

1. **Feature Store Ingestion**: Pulls raw website data from a MongoDB database, splits it into training and testing partitions, and writes them to the local artifact store.
2. **Schema & Drift Validation**: Evaluates the ingested datasets against a predefined YAML schema to ensure column alignment, type matching, and identifies potential data drift.
3. **Data Preprocessing**: Replaces missing values using a K-Nearest Neighbors (KNN) Imputer, scales/standardizes numerical columns, and exports a reusable preprocessing pipeline object (`preprocessing.pkl`).
4. **Model Tuning & Selection**: Evaluates seven classification algorithms (Random Forest, XGBoost, CatBoost, Decision Tree, Gradient Boosting, AdaBoost, and Logistic Regression) using GridSearchCV.
5. **Experiment Tracking**: Tracks training metrics (Accuracy, F1-Score, Precision, Recall) and logs model files directly to MLflow via DagsHub.
6. **Cloud Artifact Syncer**: Automatically uploads artifact logs and final serialized model files to an AWS S3 bucket.
7. **FastAPI Web Service**: Serves a web interface allowing users to input individual site characteristics or upload a batch CSV file for phishing classification.

---

## Repository Structure

Below is the layout of the primary project components:

* `app.py`: Entry point for the FastAPI application, serving endpoints for UI pages, model training, single prediction, and batch CSV predictions.
* `push_data.py`: CLI script for converting raw CSV dataset records to JSON and seeding them into the MongoDB feature store.
* `inputs.py`: Defines the Pydantic validator model containing all 30 features expected by the prediction pipeline.
* `Dockerfile`: Container configuration file based on Python 3.12-slim that installs required dependencies and the AWS CLI.
* `.github/workflows/main.yaml`: Complete GitHub Actions workflow automating testing, ECR container builds, and deployment on a self-hosted AWS runner.
* `data_schema/`:
  * `schema.yaml`: Declarative YAML specification of all columns, data types, and target variable metadata.
* `src/`:
  * `components/`: Contains Python modules representing individual steps of the training pipeline:
    * `data_ingestion.py`: Pulls records from MongoDB and partitions the datasets.
    * `data_validation.py`: Performs schema alignment checks and writes validation reports.
    * `data_transformation.py`: Formulates preprocessing pipelines using scikit-learn.
    * `model_trainer.py`: Grid searches multiple classifiers and integrates with MLflow/DagsHub.
  * `constants/`: Configuration parameters, database details, default paths, and bucket names.
  * `entity/`: Defines configuration and artifact entities for type safety across pipeline stages.
  * `pipeline/`:
    * `training_pipeline.py`: Orchestrates the training sequence and initiates AWS S3 syncing.
    * `prediction_pipeline.py`: Implements single-record inference loading local models.
    * `batch_prediction_pipeline.py`: Implements batch file processing and handles schema validations.
  * `cloud/s3_syncer.py`: Helper class that uses the AWS CLI to sync directories to and from AWS S3 buckets.
  * `utils.py`: Reusable utility functions for YAML reading/writing, numpy data manipulation, model scoring, and object serialization.
  * `logger.py` & `exception.py`: Custom logging and error handling structures.

---

## Local Setup & Development

Follow these steps to run the training pipeline and web application locally.

### Prerequisites

* Python 3.12 installed on your system.
* MongoDB instance (such as MongoDB Atlas) with a valid connection string.
* DagsHub account and repository created for MLflow experiment tracking.

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone <repository_url>
cd phish_guard_prediction
python -m venv venv
```

Activate the virtual environment:
* Windows: `venv\Scripts\activate`
* macOS/Linux: `source venv/bin/activate`

Install dependencies and the project package:

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Environment Configuration

Create a `.env` file in the root directory and configure the following variables:

```env
DATABASE_URL=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MLFLOW_TRACKING_URI=https://dagshub.com/<dagshub_username>/phish_guard_prediction.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub_username>
MLFLOW_TRACKING_PASSWORD=<dagshub_token_or_password>
DAGSHUB_USER_TOKEN=<dagshub_token_or_password>
```

### 3. Seed Database

Push raw data to your MongoDB database using the seed script:

```bash
python push_data.py
```

This reads the dataset from `Network_Data/phishingData.csv` and inserts it into the `network_metrics` collection under the `network_data` database.

### 4. Running the Pipelines

To run the machine learning training pipeline locally:

```bash
python -c "from src.pipeline.training_pipeline import TrainingPipeline; TrainingPipeline().run_pipeline()"
```

To run the FastAPI server:

```bash
python app.py
```
Or use Uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

The application UI will be available at `http://localhost:8080/`. You can train the model via the `/train` endpoint and query predictions via the GUI or REST calls.

---

## REST API Reference

The FastAPI service exposes the following endpoints:

* `GET /`: Renders the index landing page.
* `GET /predict`: Renders the prediction query interface page.
* `GET /train`: Triggers the end-to-end `TrainingPipeline` on demand.
* `POST /predict`: Accepts a single phishing query in JSON format and outputs classification results.
* `POST /predict_batch`: Accepts an uploaded `.csv` file, validates it, runs inference, and returns predicted results.

### Sample Prediction Request Payload
```json
{
  "having_IP_Address": 1,
  "URL_Length": 1,
  "Shortining_Service": 0,
  "having_At_Symbol": 1,
  "double_slash_redirecting": 1,
  "Prefix_Suffix": -1,
  "having_Sub_Domain": -1,
  "SSLfinal_State": -1,
  "Domain_registeration_length": -1,
  "Favicon": 1,
  "port": 1,
  "HTTPS_token": -1,
  "Request_URL": 1,
  "URL_of_Anchor": -1,
  "Links_in_tags": 1,
  "SFH": -1,
  "Submitting_to_email": 1,
  "Abnormal_URL": 1,
  "Redirect": 0,
  "on_mouseover": 1,
  "RightClick": 1,
  "popUpWidnow": 1,
  "Iframe": 1,
  "age_of_domain": -1,
  "DNSRecord": -1,
  "web_traffic": -1,
  "Page_Rank": -1,
  "Google_Index": 1,
  "Links_pointing_to_page": 1,
  "Statistical_report": 1
}
```

---

## Complete AWS Deployment Guide

This application utilizes a GitHub Actions pipeline (`.github/workflows/main.yaml`) to automate testing, build a Docker image, push it to AWS ECR, and execute a deployment script on a self-hosted runner.

### Architectural Setup on AWS

The deployment target consists of:
* An **Amazon ECR Private Repository** to host built Docker images.
* An **Amazon S3 Bucket** (`networksecurityumer`) configured for archiving model/pipeline artifacts.
* An **Amazon EC2 Instance** (Ubuntu Server, t3.medium or larger recommended) configured as a **GitHub Actions Self-Hosted Runner**.

### Step 1: Set up the AWS S3 Bucket
1. Log into your AWS Console and navigate to the S3 service.
2. Create a bucket named `networksecurityumer` (or update `TRAINING_BUCKET_NAME` in `src/constants/training_pipeline_constants.py` to match your custom bucket name).
3. Ensure the bucket is in your designated AWS Region (e.g., `us-east-1`).

### Step 2: Set up the Amazon ECR Repository
1. Navigate to Amazon Elastic Container Registry (ECR).
2. Create a new private repository. Name it matching your project (e.g., `phish-guard-prediction`).
3. Copy the repository URL (e.g., `<aws_account_id>.dkr.ecr.<region>.amazonaws.com/phish-guard-prediction`).

### Step 3: Configure the EC2 Hosting Instance & Self-Hosted Runner
Since the CD step in the workflow runs on a `self-hosted` runner, you must configure your EC2 instance to execute GitHub Action runner tasks.

#### 1. Launch EC2 Instance
Launch an EC2 Instance using Ubuntu Server. Configure security groups to allow inbound HTTP traffic on port `8080` (or whichever port you map) and SSH traffic on port `22` for management.

#### 2. Create and Attach an IAM Instance Profile (Recommended)
Attach an IAM Role to the EC2 instance with the following permissions to ensure the runner and Docker containers can securely access S3 and ECR without needing hardcoded credentials inside the server environment:
* `AmazonEC2ContainerRegistryReadOnly` (to pull images from ECR)
* `AmazonS3FullAccess` (or write access restricted to the bucket `networksecurityumer` to upload training run artifacts)

#### 3. Install Docker on the EC2 Instance
Connect to your EC2 instance via SSH and execute the following commands to install Docker:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

To allow the GitHub Actions runner user to execute Docker commands without using `sudo`, add the user to the docker group:

```bash
sudo usermod -aG docker $USER
```

Log out and connect again for the group changes to take effect.

#### 4. Register the Self-Hosted Runner on the EC2 Instance
1. In your GitHub repository, go to **Settings** > **Actions** > **Runners**.
2. Click **New self-hosted runner** and select **Linux** as the runner OS.
3. Follow the provided setup instructions on your EC2 instance to download and extract the runner agent.
4. When configuring the runner (`./config.sh`), follow the prompts. You can accept default settings and labels. Ensure the runner successfully registers and is listening for jobs.
5. Install the runner as a background systemd service so it persists across reboots:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

### Step 4: Configure GitHub Repository Secrets
Navigate to **Settings** > **Secrets and variables** > **Actions** in your GitHub repository and define the following secrets:

* `AWS_ACCESS_KEY`: AWS access key for an IAM user with permissions to build/push to ECR.
* `AWS_SECRET_ACCESS_KEY`: AWS secret key corresponding to the access key.
* `AWS_REGION`: The AWS Region where your services are deployed (e.g., `us-east-1`).
* `AWS_ECR_REPOSITORY_NAME`: The name of the ECR repository created in Step 2 (e.g., `phish-guard-prediction`).
* `AWS_ECR_LOGIN_URI`: The ECR registry domain (e.g., `<aws_account_id>.dkr.ecr.<region>.amazonaws.com`).
* `DAGSHUB_PASSWORD`: DagsHub token or password used to authenticate MLflow tracking on the runner.
* `DATABASE_URL`: MongoDB connection string. *Note: Since this secret is required for database operations at runtime, ensure the Docker run step in `.github/workflows/main.yaml` passes this variable if running the application (e.g., by adding `-e DATABASE_URL=${{ secrets.DATABASE_URL }}`).*

### Step 5: Execute CI/CD Pipeline
Once your secrets are saved and the self-hosted runner is active, any push to the `main` branch (excluding modifications to `README.md`) will trigger the CI/CD pipeline:

1. **Continuous Integration**: Code is checked out, linted, and tests are run on a GitHub-hosted runner.
2. **Continuous Delivery**: The repository is checked out, a Docker image is compiled using the `Dockerfile`, and the image is tagged and pushed to your AWS ECR Registry.
3. **Continuous Deployment**: The self-hosted runner on the EC2 instance pulls the latest container image from ECR, stops and prunes any outdated Docker instances, and launches the new image exposing application port `8080` while injecting the required environment variables.
