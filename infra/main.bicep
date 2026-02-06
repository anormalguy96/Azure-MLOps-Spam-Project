// Minimal IaC: Storage account + blob containers for datasets and logs.
// AML workspace and Function App are created via az CLI in the README to keep this sample short.

param location string = resourceGroup().location
param storageAccountName string

@allowed([
  'Standard_LRS'
  'Standard_GRS'
  'Standard_ZRS'
])
param skuName string = 'Standard_LRS'

param datasetsContainerName string = 'datasets'
param logsContainerName string = 'logs'

resource sa 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: skuName
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  name: '${sa.name}/default'
}

resource datasets 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: '${sa.name}/default/${datasetsContainerName}'
  properties: {
    publicAccess: 'None'
  }
  dependsOn: [
    blobService
  ]
}

resource logs 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: '${sa.name}/default/${logsContainerName}'
  properties: {
    publicAccess: 'None'
  }
  dependsOn: [
    blobService
  ]
}

output storageAccountResourceId string = sa.id
