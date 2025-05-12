from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
from tqdm import tqdm
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import csv
import re


class Utils:
    @staticmethod
    def process_image(image):
        image = image.convert('L')  # Grayscale
        width, height = image.size
        image = image.resize((width * 2, height * 2),
                             Image.Resampling.LANCZOS)  # Upscale
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(3)  # Add contrast
        image = image.point(lambda p: 0 if p < 200 else 255)  # Text visibility
        return image

    @staticmethod
    def standardize_id(extracted_text: str) -> list:
        """
        Standardizes the extracted ID text based on predefined rules.
        Handles cleaning, splitting, and applying specific transformations.

        Args:
            extracted_text (str): The raw text containing ID information

        Returns:
            list: List of standardized ID parts
        """
        if not extracted_text:
            return []

        # 1. Remove trailing newline and potential extra whitespace
        cleaned_text = extracted_text.strip()

        # First, clean the input by removing all content within parentheses
        # This handles cases where there might be complex content including slashes inside parentheses
        cleaned_no_parentheses = re.sub(r'\([^)]*\)', '', cleaned_text)

        # Then standardize the cleaned result
        cleaned_no_parentheses = cleaned_no_parentheses.strip()

        # If the input was just something in parentheses, handle that case
        if not cleaned_no_parentheses:
            return []

        # Split by comma if present
        comma_parts = [part.strip()
                       for part in cleaned_no_parentheses.split(',')]
        standardized_parts = []

        # Process each comma-separated part
        for part in comma_parts:
            if not part:
                continue

            # Check if we have a prefix pattern (letters followed by numbers)
            prefix_match = re.match(r'^([A-Z]+)(\d+)', part)
            prefix = ""
            if prefix_match:
                prefix = prefix_match.group(1)

            # Split by slash if present
            slash_parts = [sub_part.strip() for sub_part in part.split('/')]

            # Process each slash part
            for i, sub_part in enumerate(slash_parts):
                if not sub_part:
                    continue

                # If this is not the first part and it's only digits, prepend the prefix
                if i > 0 and re.match(r'^\d+$', sub_part) and prefix:
                    sub_part = prefix + sub_part

                if sub_part:
                    standardized_parts.append(sub_part)

        # De-duplicate while preserving order
        final_standardized_list = []
        seen = set()
        for part in standardized_parts:
            if part not in seen:
                final_standardized_list.append(part)
                seen.add(part)

        return final_standardized_list

    @staticmethod
    def clean_json_response(response):
        # Process response
        start_marker = '```json\n'
        end_marker = '\n```'

        cleaned_json_string = response
        if cleaned_json_string.startswith(start_marker):
            cleaned_json_string = cleaned_json_string[len(start_marker):]
        if cleaned_json_string.endswith(end_marker):
            cleaned_json_string = cleaned_json_string[:-len(end_marker)]

        try:
            data_dict = json.loads(cleaned_json_string)
            print("Successfully parsed JSON string into Python dictionary.")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print("The cleaned string might still be invalid JSON.")
            data_dict = None

        return data_dict

    @staticmethod
    def compute_and_append(data_dict):
        # Append data into csv
        MM_TO_MIL = 39.4
        csv_file = "table_data.csv"

        try:
            id = data_dict['model_name']
            min_coplanarity = 0
            max_coplanarity = round(
                data_dict['table']['aaa']['MAX'] * MM_TO_MIL, 5)
            nom_ball_width = round(
                data_dict['table']['øb']['NOM'] * MM_TO_MIL, 5)
            nom_pitch = round(data_dict['table']['e'] * MM_TO_MIL, 5)
            ball_width = round((data_dict['table']['øb']['MAX'] -
                                data_dict['table']['øb']['NOM']) * MM_TO_MIL, 5)

            row = [id, min_coplanarity, max_coplanarity,
                   nom_ball_width, nom_pitch, ball_width]
            headers = ['id', 'min_coplanarity',
                       'max_coplanarity', 'nom_ball_width', 'nom_pitch', 'ball_width']

            # Write to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not os.path.exists(
                        csv_file) or os.path.getsize(csv_file) == 0:
                    writer.writerow(headers)
                writer.writerow(row)

        except Exception as e:
            print("Exception in", e)
            print(data_dict)
