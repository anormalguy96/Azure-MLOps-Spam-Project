# infra/

This folder contains a tiny Bicep template to provision the **Storage Account** and the two **Blob containers**
required by the project:

- `datasets` (training data)
- `logs` (inference request/response logs)

The Azure ML Workspace and Azure Function App are created via CLI commands in the root README to keep this sample minimal.

## Deploy

```bash
az deployment group create \
  -g $AZURE_RESOURCE_GROUP \
  --template-file infra/main.bicep \
  --parameters storageAccountName=$STORAGE_ACCOUNT_NAME
```
