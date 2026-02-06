# azure-mini-mlops-blob-functions

Production-style **mini MLOps** repo that trains a **spam detector** (scikit-learn + MLflow), stores the dataset in **Azure Blob**, deploys the model as an **Azure ML managed online endpoint**, and exposes an **Azure Functions** HTTP API that calls the endpoint and logs every request/response to Blob.

---

## Repository name & description

**Name:** `azure-mini-mlops-blob-functions`  
**Description:** Minimal, production-style Azure MLOps sample: Blob dataset + Azure ML training/registration/managed-online-endpoint + Azure Functions predictor with Blob logging.

---

## Overview

- **Use-case:** spam detection (binary classification)
- **Model:** `TfidfVectorizer + LogisticRegression`
- **Dataset:** `data/spam_sample.csv` (shipped in repo; uploaded to Blob container `datasets`)
- **Metrics:** accuracy, F1
- **Tracking:** MLflow (logs params/metrics; saves model via `mlflow.sklearn.save_model`)
- **Serving:** Azure ML **ManagedOnlineEndpoint** with key auth
- **API:** Azure Functions HTTP trigger `/api/predict` calls endpoint and logs JSON to Blob container `logs`

---

## Architecture

```
+------------------------+          +------------------------------+
| GitHub repo            |          | Azure Machine Learning       |
| - data/spam_sample.csv |          | Workspace + Compute Cluster  |
+-----------+------------+          |                              |
            | upload                 |  Command Job trains model    |
            v                        |  + MLflow logs + outputs     |
+------------------------+          +--------------+---------------+
| Azure Blob Storage     |                         |
| datasets/  logs/       |                         | register+deploy
+-----------+------------+                         v
            ^                        +-----------------------------+
            | write logs             | Managed Online Endpoint     |
            |                        | score.py (init/run)         |
+-----------+------------+           +--------------+--------------+
| Azure Functions        |                          ^
| /api/predict           |  HTTP (Bearer key)       |
| calls endpoint + logs  +--------------------------+
+------------------------+
```

---

## Folder structure

- `data/spam_sample.csv`
- `src/training/train.py`
- `src/serving/score.py`
- `src/azureml/` (bootstrap, data asset, job submit, register, deploy, test)
- `src/functions/predict_function/` (Azure Functions app)
- `infra/` (optional minimal IaC for Storage)
- `.github/workflows/ci.yml` (lint + tests)

---

## Prerequisites

**Local**
- Python **3.10+**
- (Optional) **Azurite** for Blob emulation (or run without it; logs fall back to `./local_blob_logs/`)

**Azure**
- Azure CLI
- Azure ML CLI extension v2:
  ```bash
  az extension add -n ml
  ```
- Azure Functions Core Tools (for `func start` and `func azure functionapp publish`)

---

## Local quickstart (no Azure required)

### 1) Create venv and install deps

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r src/training/requirements.txt
pip install -r src/azureml/requirements.txt
pip install -r src/functions/predict_function/requirements.txt
pip install -r requirements-dev.txt
```

### 2) Train locally

```bash
python -m src.training.train --data data/spam_sample.csv --output-dir artifacts/model
```

### 3) Start local scoring server

```bash
python -m src.serving.local_server --model-dir artifacts/model --port 8000
```

### 4) Run the Function locally (simplified local mode)

```bash
cd src/functions/predict_function
# Windows (PowerShell):
# $env:AML_SCORING_URI="http://127.0.0.1:8000/score"
# $env:AML_ENDPOINT_KEY="local-dev-key"
set AML_SCORING_URI=http://127.0.0.1:8000/score
set AML_ENDPOINT_KEY=local-dev-key

func start
```

Test:

```bash
curl -X POST http://localhost:7071/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"WIN a free gift card now!!!\"}"
```

Logs go to `./local_blob_logs/logs/` unless `AZURE_STORAGE_CONNECTION_STRING` is set.

---

## Azure deployment (end-to-end)

### 0) Set environment variables

Use `.env.example` as reference.

```bash
export AZURE_SUBSCRIPTION_ID="<sub-id>"
export AZURE_RESOURCE_GROUP="rg-mini-mlops"
export AZURE_LOCATION="westeurope"
export AZUREML_WORKSPACE_NAME="mlw-mini-mlops"

export STORAGE_ACCOUNT_NAME="<globally-unique-name>"
export BLOB_DATA_CONTAINER="datasets"
export BLOB_LOG_CONTAINER="logs"

export AML_ENDPOINT_NAME="spam-endpoint"
export AML_DEPLOYMENT_NAME="blue"
```

Login & select subscription:

```bash
az login
az account set --subscription "$AZURE_SUBSCRIPTION_ID"
```

### 1) Create resource group

```bash
az group create -n "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION"
```

### 2) Create Storage account + containers

**Option A (Bicep)**

```bash
az deployment group create           -g "$AZURE_RESOURCE_GROUP"           --template-file infra/main.bicep           --parameters storageAccountName="$STORAGE_ACCOUNT_NAME"
```

**Option B (CLI)**

```bash
az storage account create           -n "$STORAGE_ACCOUNT_NAME" -g "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION"           --sku Standard_LRS --kind StorageV2
```

Get the connection string:

```bash
export AZURE_STORAGE_CONNECTION_STRING="$(az storage account show-connection-string           -n "$STORAGE_ACCOUNT_NAME" -g "$AZURE_RESOURCE_GROUP" --query connectionString -o tsv)"
```

Create containers:

```bash
az storage container create --name "$BLOB_DATA_CONTAINER" --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
az storage container create --name "$BLOB_LOG_CONTAINER" --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
```

### 3) Create Azure ML workspace

```bash
az ml workspace create           -n "$AZUREML_WORKSPACE_NAME" -g "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION"
```

(Optional) Defaults:

```bash
az configure --defaults group="$AZURE_RESOURCE_GROUP" workspace="$AZUREML_WORKSPACE_NAME"
```

### 4) Upload CSV to Blob

```bash
python -m src.azureml.bootstrap_storage
```

### 5) Create AML datastore + data asset

```bash
python -m src.azureml.create_data_asset
```

### 6) Submit training job (Azure ML SDK v2)

```bash
python -m src.azureml.submit_train_job
```

Stream:

```bash
az ml job stream -n <JOB_NAME>
```

### 7) Register the model

```bash
python -m src.azureml.register_model --job-name <JOB_NAME> --model-name spam-detector
```

### 8) Deploy managed online endpoint

```bash
python -m src.azureml.deploy_endpoint
```

Capture output and set:

```bash
export AML_SCORING_URI="https://<...>.inference.ml.azure.com/score"
export AML_ENDPOINT_KEY="<primary-key>"
```

### 9) Call the endpoint

```bash
python -m src.azureml.endpoint_test "FREE gift card now!!!"
```

Or curl:

```bash
curl -X POST "$AML_SCORING_URI"           -H "Authorization: Bearer $AML_ENDPOINT_KEY"           -H "Content-Type: application/json"           -d '{"text":"FREE gift card now!!!"}'
```

---

## Deploy the Azure Function App

### 1) Create Function App

```bash
export FUNC_APP_NAME="fn-mini-mlops-<unique>"

az functionapp create           --resource-group "$AZURE_RESOURCE_GROUP"           --consumption-plan-location "$AZURE_LOCATION"           --runtime python           --runtime-version 3.10           --functions-version 4           --name "$FUNC_APP_NAME"           --storage-account "$STORAGE_ACCOUNT_NAME"
```

### 2) Configure app settings (secrets via env vars)

```bash
az functionapp config appsettings set           -g "$AZURE_RESOURCE_GROUP" -n "$FUNC_APP_NAME"           --settings           AML_SCORING_URI="$AML_SCORING_URI"           AML_ENDPOINT_KEY="$AML_ENDPOINT_KEY"           AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING"           BLOB_LOG_CONTAINER="$BLOB_LOG_CONTAINER"
```

### 3) Publish

```bash
cd src/functions/predict_function
func azure functionapp publish "$FUNC_APP_NAME" --python
```

### 4) Test the function

```bash
curl -X POST "https://$FUNC_APP_NAME.azurewebsites.net/api/predict"           -H "Content-Type: application/json"           -d '{"text":"URGENT: Verify your account now http://tinyurl.com/winfast"}'
```

Inspect logs in Blob:

```bash
az storage blob list --container-name "$BLOB_LOG_CONTAINER"           --connection-string "$AZURE_STORAGE_CONNECTION_STRING" -o table
```

---

## Troubleshooting

- **Missing ML extension:** `az extension add -n ml`
- **Auth errors (DefaultAzureCredential):** run `az login`; if using a service principal, ensure environment vars are set.
- **Quota/SKU issues:** set `AML_COMPUTE_SKU` or `AML_ENDPOINT_SKU` to something available in your region.
- **Endpoint unhealthy:** `az ml online-deployment get-logs --endpoint-name $AML_ENDPOINT_NAME --name $AML_DEPLOYMENT_NAME`
- **Function import errors:** publish from `src/functions/predict_function/` and ensure `requirements.txt` is there.
- **Blob errors:** verify `AZURE_STORAGE_CONNECTION_STRING` and container existence; in local mode, logger falls back to files.

---

## What this demonstrates (mapping)

| Requirement | Repo component |
|---|---|
| Azure ML (training) | `src/azureml/submit_train_job.py`, `src/training/train.py` |
| Register model | `src/azureml/register_model.py` |
| Deploy managed online endpoint | `src/azureml/deploy_endpoint.py`, `src/serving/score.py` |
| Azure Storage Blob (dataset) | `src/azureml/bootstrap_storage.py`, `src/azureml/create_data_asset.py` |
| Azure Storage Blob (prediction logging) | `src/functions/predict_function/shared_code/blob_logger.py` |
| Azure Functions HTTP trigger | `src/functions/predict_function/predict/__init__.py` |
| Cloud deployment concepts | `infra/main.bicep`, Azure CLI steps, GitHub Actions CI |
| Clean Python (typing/lint/tests) | `pyproject.toml`, `ruff`, `mypy`, `pytest`, `.github/workflows/ci.yml` |

---

## Cost control

Delete everything:

```bash
az group delete -n "$AZURE_RESOURCE_GROUP" --yes --no-wait
```
