"""
Configuration settings for the OCR application.
"""
import os
from pathlib import Path

# API Configuration
DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_CACHE_PATH = "./cache.json"
DEFAULT_IMAGE_FOLDER = "./all_images"

# Image Processing Constants
MM_TO_MIL = 39.4
BRIGHTNESS_THRESHOLD = 150
PERCENTAGE_THRESHOLD = 0.80
REQUIRED_CONSECUTIVE_LINES = 3
SEARCH_AREA_RATIO = 0.30
BLOCK_SIZE = 81
C_VALUE = 7
CONTRAST_ENHANCEMENT = 3
TEXT_VISIBILITY_THRESHOLD = 180
MEDIAN_FILTER_SIZE = 3
IMAGE_CROP_RATIO = 0.975

# Similarity Matching Constants
SIMILARITY_THRESHOLD = 85

# File Extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

# Directory Names
PROCESSED_IMAGES_DIR = "processed_images"
UNDEFINED_TABLES_DIR = "undefined"

# Output Configuration
OUTPUT_PATH = "./data/new_data.csv"

# Cost Calculation (Gemini 2.5 Flash pricing)
INPUT_TOKEN_COST_PER_MILLION = 0.3
OUTPUT_TOKEN_COST_PER_MILLION = 2.5

# System Instructions
SYSTEM_INSTRUCTION = """
You are a data extraction AI. Your task is to extract information from a technical document into a raw JSON object.
Follow these rules:
    1.  **`id`**: Extract the complete identification string from the document's top header. If there are sub-headers, include them in the ID, wrapped in parentheses. This string might contain parentheses, slashes, or other characters.
    2.  **`table`**: Extract the table data in the bottom-left portion of the image. Process this table to create a JSON object.
    *   For each row in the table, use the label from the first column(e.g., 'A', 'A1', 'D/E', 'øb') as a key in your main `table` object.
    *   The value for each key depends on the row's content:
        * **Case 1 (Multiple Columns): ** If the row has values for "MIN", "NOM", and / or "MAX", the value should be a ** nested object ** with `MIN`, `NOM`, and `MAX` as keys. Extract the corresponding numeric values. If a value for MIN, NOM, or MAX is not present or is blank, its value in the JSON must be `null`.
        * **Case 2 (Single Value): ** If the row has only one primary value(like 'D/E' or 'M' in the example), the value should be that single ** number ** directly.
    3. ** Output Format**:
        *   Your entire response MUST be a single, raw JSON object.
        *   Do NOT include any surrounding text, explanations, or markdown formatting like ```json.
        *   The JSON object must have two top-level keys: `id` and `table`.
        *   The value for `id` should be the string extracted from the header.
        *   The value for `table` must be a single JSON object(not an array). The keys of this object are the labels from the table rows.
        *   All extracted text should be trimmed of leading/trailing whitespace.
    4. ** Error Handling**:
        * If the ID is not found, use `"id": null`.
        * If the table is not found, use `"table": {}`.

    Example of required output:
    "CP132/CPG132": {
        "id": "CP132/CPG132",
        "table": {
            "A": {
                "MIN": null,
                "NOM": 1.0,
                "MAX": 1.1
            },
            "A1": {
                "MIN": 0.15,
                "NOM": 0.2,
                "MAX": 0.25
            },
            "D/E": {
                "MIN": 7.9,
                "NOM": 8.0,
                "MAX": 8.1
            },
            "D1/E1": 6.5,
            "e": 0.5,
            "øb": {
                "MIN": 0.25,
                "NOM": 0.3,
                "MAX": 0.35
            },
            "ccc": {
                "MIN": null,
                "NOM": null,
                "MAX": 0.1
            },
            "ddd": {
                "MIN": null,
                "NOM": null,
                "MAX": 0.08
            },
            "eee": {
                "MIN": null,
                "NOM": null,
                "MAX": 0.15
            },
            "ZD/ZE": {
                "MIN": 0.6,
                "NOM": 0.75,
                "MAX": 0.9
            },
            "M": 14
        }
    }
"""

# Coplanarity field mapping
COPLANARITY_FIELDS = ['aaa', 'bbb', 'ccc', 'ddd', 'eee']

# Default values
DEFAULT_BALL_QUALITY = 75
DEFAULT_PITCH_ERROR = 0
DEFAULT_MIN_COPLANARITY = 0

# Column names for output
OUTPUT_COLUMNS = {
    'BGA_PKG_TYPE': 'BGA Pkg Type',
    'COPLANARITY_MIN': 'Coplanarity_min',
    'COPLANARITY_MAX': 'Coplanarity_max',
    'PITCH_ERROR': 'Pitch Error_+/-',
    'NOMINAL_PITCH': 'Nominal Pitch_nom',
    'BALL_WIDTH_ERROR': 'Ball Width_Error +/-',
    'NOM_BALL_WIDTH': 'Nom. Ball Width_nom',
    'BALL_QUALITY': 'Ball Quality_%'
}
