# OCR

## Project Structure

```
OCR/
├── config.py                 # Centralized configuration
├── logger.py                 # Logging utilities
├── image_processor.py        # Image processing operations
├── data_processor.py         # CSV and data processing
├── similarity_matcher.py     # ID matching and similarity scoring
├── cache_manager.py          # Cache management
├── cost_tracker.py           # API usage cost tracking
├── Gemini.py                 # Gemini API client
├── Main.py                   # Main script
├── requirements.txt          # Dependencies
├── README.md
├── all_images/               # Input images directory
├── processed_images/         # Processed images directory
├── data/                     # Data files directory
└── cache.json                # Cache file
```

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. **Set up environment variables**

    Create a new file named `.env` in the root of the project. Then, open the file and add the following content, replacing `"your_api_key_here"` with your actual key:

```ini
GOOGLE_API_KEY="your_api_key_here"
```

## Usage

### Basic Usage

```bash
python Main.py path/to/your/data.csv
```

## Configuration

Edit `config.py` to modify:

-   API settings
-   Image processing parameters
-   Similarity matching thresholds
-   File paths and directories
-   Cost calculation parameters

## Modules Overview

### `config.py`

Centralized configuration for all application settings.

### `logger.py`

Provides consistent logging across the application.

### `image_processor.py`

Handles all image processing operations:

-   Image enhancement and preprocessing
-   Adaptive thresholding
-   Company section cropping

### `data_processor.py`

Manages CSV operations and data transformations:

-   Multi-level header processing
-   JSON response cleaning
-   Data computation and validation

### `similarity_matcher.py`

Handles ID matching and similarity scoring:

-   Fuzzy string matching
-   Similarity score calculations
-   Prompt generation for Gemini

### `cache_manager.py`

Manages cache operations:

-   Cache loading and saving
-   Image processing coordination
-   File management

### `cost_tracker.py`

Tracks API usage costs:

-   Token usage monitoring
-   Cost calculations
-   Usage reporting

### `Gemini_refactored.py`

-   Gemini API client:

### `Main.py`

-   Main application orchestrator:
