"""
Data processing utilities for CSV operations and data transformations.
"""
import pandas as pd
import numpy as np
import json
import re
import regex
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

from config import (
    MM_TO_MIL, COPLANARITY_FIELDS, DEFAULT_BALL_QUALITY,
    DEFAULT_PITCH_ERROR, DEFAULT_MIN_COPLANARITY, OUTPUT_COLUMNS
)
from logger import logger


class DataProcessor:
    """Handles CSV operations and data transformations."""

    @staticmethod
    def read_and_prepare_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Read a CSV with a multi-level header and prepare data for processing.

        Args:
            csv_path: Path to the CSV file

        Returns:
            Tuple containing:
                - processing_df: DataFrame with simplified, clean column names
                - original_df: The original DataFrame with its MultiIndex columns
                - column_map: A mapping from clean names to original MultiIndex tuples
        """
        try:
            df = pd.read_csv(csv_path, header=[0, 2])

            original_columns = df.columns
            top_level = original_columns.get_level_values(0)
            bottom_level = original_columns.get_level_values(1)

            # Clean column names
            top_level_filled = pd.Series(top_level).replace(
                to_replace=r'^Unnamed:.*', value=np.nan, regex=True
            ).ffill().str.strip()
            bottom_level_cleaned = pd.Series(bottom_level).str.strip()

            # Create new column names
            new_columns = []
            for top, bottom in zip(top_level_filled, bottom_level_cleaned):
                if (top == bottom or bottom.startswith('Unnamed') or
                        top.startswith('Unnamed')):
                    new_columns.append(top)
                else:
                    new_columns.append(f"{top}_{bottom}")

            # Create column mapping
            column_map = dict(zip(new_columns, original_columns))

            # Create processing DataFrame
            processing_df = df.copy()
            processing_df.columns = new_columns

            # Remove completely empty rows
            na_rows = processing_df.isnull().all(axis=1)
            processing_df = processing_df[~na_rows]
            df = df[~na_rows]

            logger.info(
                f"Successfully loaded CSV with {len(processing_df)} rows")
            return processing_df, df, column_map

        except Exception as e:
            logger.error(f"Error reading CSV file {csv_path}: {e}")
            raise

    @staticmethod
    def clean_json_response(response: str) -> Optional[Dict[str, Any]]:
        """
        Clean and parse JSON response from API.

        Args:
            response: Raw response string

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        start_marker = '```json\n'
        end_marker = '\n```'

        cleaned_json_string = response
        if cleaned_json_string.startswith(start_marker):
            cleaned_json_string = cleaned_json_string[len(start_marker):]
        if cleaned_json_string.endswith(end_marker):
            cleaned_json_string = cleaned_json_string[:-len(end_marker)]

        try:
            data_dict = json.loads(cleaned_json_string)
            logger.debug("Successfully parsed JSON string")
            return data_dict
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            logger.debug("The cleaned string might still be invalid JSON")
            return None

    @staticmethod
    def compute_data(id_value: str, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute derived data from extracted table data.

        Args:
            id_value: The ID value
            data_dict: Dictionary containing extracted table data

        Returns:
            Dictionary with computed values
        """
        try:
            # Calculate max coplanarity
            max_coplanarity = DataProcessor._calculate_max_coplanarity(
                data_dict['table'])

            # Calculate other values
            nom_ball_width = round(
                data_dict['table']['øb']['NOM'] * MM_TO_MIL, 5)
            nom_pitch = round(data_dict['table']['e'] * MM_TO_MIL, 5)
            ball_width_error = round(
                (data_dict['table']['øb']['MAX'] -
                 data_dict['table']['øb']['NOM']) * MM_TO_MIL, 5
            )

            result = {
                OUTPUT_COLUMNS['BGA_PKG_TYPE']: data_dict['id'],
                OUTPUT_COLUMNS['COPLANARITY_MIN']: DEFAULT_MIN_COPLANARITY,
                OUTPUT_COLUMNS['COPLANARITY_MAX']: max_coplanarity,
                OUTPUT_COLUMNS['PITCH_ERROR']: DEFAULT_PITCH_ERROR,
                OUTPUT_COLUMNS['NOMINAL_PITCH']: nom_pitch,
                OUTPUT_COLUMNS['BALL_WIDTH_ERROR']: ball_width_error,
                OUTPUT_COLUMNS['NOM_BALL_WIDTH']: nom_ball_width,
                OUTPUT_COLUMNS['BALL_QUALITY']: DEFAULT_BALL_QUALITY
            }

            return result

        except KeyError as e:
            logger.error(f"Missing required field in data_dict: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error computing data: {e}")
            return {}

    @staticmethod
    def _calculate_max_coplanarity(table_data: Dict[str, Any]) -> float:
        """
        Calculate maximum coplanarity from table data.

        Args:
            table_data: Table data dictionary

        Returns:
            Maximum coplanarity value
        """
        for field in COPLANARITY_FIELDS:
            if field in table_data and 'MAX' in table_data[field]:
                return round(table_data[field]['MAX'] * MM_TO_MIL, 5)

        logger.warning("No coplanarity field found in table data")
        return 0.0

    @staticmethod
    def clean_id(id_value: str) -> str:
        """
        Clean and standardize ID values.

        Args:
            id_value: Raw ID string

        Returns:
            Cleaned ID string
        """
        id_value = id_value.replace("(G)", "")
        stripped_id = regex.sub(r'\((?:[^()]|(?R))*\)', "", id_value)
        letters = re.findall(r'[a-zA-Z]', stripped_id)
        digits = re.findall(r'\d', stripped_id)

        if len(letters) >= 2 and len(digits) >= 4:
            bracket_match = regex.findall(r'\((?:[^()]|(?R))*\)', id_value)
            if bracket_match:
                brackets = re.sub(r"VIRTEX-.*?:s*", "", bracket_match[-1])
                cleaned_id = "".join(letters[:2]) + \
                    "".join(digits[:4]) + " " + brackets
            else:
                cleaned_id = "".join(letters[:2]) + "".join(digits[:4])
            return cleaned_id
        elif len(letters) >= 2:
            return "".join(letters[:2]) + "".join(digits)
        elif len(digits) >= 4:
            return "".join(letters) + "".join(digits[:4])
        else:
            return id_value

    @staticmethod
    def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            output_path: Output file path
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving DataFrame to {output_path}: {e}")
            raise

    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names by replacing Unnamed columns with empty strings.

        Args:
            df: DataFrame with MultiIndex columns

        Returns:
            DataFrame with cleaned column names
        """
        old_columns = df.columns
        new_levels = []

        for level in old_columns.levels:
            new_level = level.str.replace(r'^Unnamed.*', '', regex=True)
            new_levels.append(new_level)

        df.columns = old_columns.set_levels(new_levels)
        return df
