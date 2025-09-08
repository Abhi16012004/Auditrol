# Auditrol Testing Arena - Installation & Usage Guide

## Overview

Auditrol Testing Arena is a platform for testing and evaluating audit trail and compliance monitoring solutions. This README provides detailed instructions for setting up and running the application.

## Requirements

- Python 3.12 or higher
- Poetry (dependency management)
- Internet connection for downloading dependencies and loading example datasets

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd auditrol_testing_arena
```

### 2. Install Dependencies

Using Poetry (recommended):

```bash
poetry install
```

Alternatively, using pip:

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Note: OpenAI API key is only required if you plan to use OpenAI embedding models for duplicate detection.

### 4. Run the Application

Using Poetry:

```bash
poetry run streamlit run backend/app.py
```

Or using streamlit directly:

```bash
streamlit run backend/app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Module Navigation

The application consists of the following modules:

1. **Main Dashboard** - Overview of available tools and modules
2. **Duplicate Detection Module** - For finding semantic duplicates in your data
3. **Run Code Module** - For executing and testing custom code snippets

Use the sidebar navigation to move between different modules.

## Working with Data

### Uploading Data

1. Navigate to the appropriate module
2. Use the file upload widget to upload CSV or Excel files
3. Data will be loaded and a preview displayed

### Using Example Data

The Run Code module provides example datasets that can be loaded directly:
- Iris Dataset
- Titanic Dataset
- Weather Data

## Embedding Models

The Duplicate Detection module supports multiple embedding models:

| Model | Type | API Key Required |
|-------|------|------------------|
| all-MiniLM-L6-v2 | Free (SentenceTransformers) | No |
| all-mpnet-base-v2 | Free (SentenceTransformers) | No |
| paraphrase-multilingual-MiniLM-L12-v2 | Free (SentenceTransformers) | No |
| text-embedding-3-large | Paid (OpenAI) | Yes |

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Ensure all dependencies are installed via `poetry install`
   - Verify you're running in the correct poetry environment

2. **API Key Errors**
   - Check that your `.env` file exists and contains the required API keys
   - For OpenAI models, verify your API key has sufficient quota

3. **File Upload Issues**
   - Ensure your CSV/Excel files are properly formatted
   - Check for special characters or encoding issues

### Logs

Application logs are available in the console when running the application.

## Additional Resources

- For technical details about the implemented models, see `README_TECHNICAL.md`
- For contribution guidelines, see `CONTRIBUTING.md` (if applicable)

## License

[Specify your license here]

## Contact

[Your contact information]
