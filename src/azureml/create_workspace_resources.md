# Azure resource creation (CLI)

This repo is designed to be runnable locally and deployable to Azure.
The root README contains the full end-to-end steps.

Quick checklist:

- Resource group
- Storage account + containers: `datasets`, `logs`
- Azure ML workspace
- AML compute cluster
- Managed online endpoint
- Azure Function App (Python)

Secrets are handled via environment variables / app settings.
