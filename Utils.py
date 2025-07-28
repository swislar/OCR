from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import csv
import re
from rapidfuzz import fuzz
import cv2


class Utils:
    @staticmethod
    def process_image(image):
        image = image.convert('L')  # Grayscale
        width, height = image.size
        image = image.resize((width * 2, height * 2),
                             Image.Resampling.LANCZOS)  # Upscale
        width, height = image.size
        image = image.crop((0, 0, width, height * 0.975))

        open_cv_image = np.array(image)
        block_size = 61
        C = 6

        processed_cv_image = cv2.adaptiveThreshold(
            open_cv_image,
            255,  # Max value to assign
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            C
        )

        # 6. Convert the processed OpenCV image (NumPy array) back to a PIL Image
        image = Image.fromarray(processed_cv_image)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(3)  # Add contrast
        image = image.point(lambda p: 0 if p < 180 else 255)  # Text visibility
        image.filter(ImageFilter.MedianFilter(size=3))
        return Utils.crop_company_section(image)

    @staticmethod
    def crop_company_section(image,
                             brightness_threshold=150,
                             percentage_threshold=0.80,
                             required_consecutive_lines=3,
                             search_area_ratio=0.30):
        """
        Crops everything below a prominent white horizontal band from the bottom.

        The logic is dynamic:
        - If 2 or more white bands are found, it crops at the second band from the bottom.
        - If only 1 white band is found, it crops at that band.
        - If no bands are found, it returns the original image.

        Args:
            image (PIL.Image.Image): The input image to crop.
            brightness_threshold (int): Pixel value above which a pixel is considered 'white'.
            percentage_threshold (float): Ratio of white pixels required to classify a row as 'white'.
            required_consecutive_lines (int): How many consecutive white rows define a 'thick' band.
            search_area_ratio (float): The portion of the image from the bottom to search.

        Returns:
            PIL.Image.Image: The cropped image or the original if no suitable band is found.
        """
        image_array = np.array(image.convert('L'))
        height, width = image_array.shape

        start_search_y = int(height * (1 - search_area_ratio))

        consecutive_white_lines = 0
        # This list will store the top y-coordinate of each band found, from bottom to top.
        found_band_positions = []

        # Scan the entire search area from the bottom up to find all bands.
        for y in range(height - 1, start_search_y, -1):
            row = image_array[y, :]
            white_pixels_count = np.sum(row > brightness_threshold)
            percentage_white = white_pixels_count / width

            is_row_white = percentage_white > percentage_threshold

            if is_row_white:
                consecutive_white_lines += 1
            else:
                # If we just passed a thick white band, record its position.
                if consecutive_white_lines >= required_consecutive_lines:
                    # The top of the band is the first non-white line we hit.
                    band_top_y = y + 1
                    found_band_positions.append(band_top_y)

                # Reset the counter as the streak is broken.
                consecutive_white_lines = 0

        # Edge case: Check if the search ended while inside a white band.
        if consecutive_white_lines >= required_consecutive_lines:
            band_top_y = start_search_y + 1
            found_band_positions.append(band_top_y)

        # --- Decision Logic: Decide where to crop based on the number of bands found ---
        crop_y = height  # Default to no crop
        total_bands_found = len(found_band_positions)

        if total_bands_found >= 2:
            # We found 2 or more bands. Target the second one from the bottom.
            # Since we scanned bottom-up, the second band is at index 1.
            crop_y = found_band_positions[1]
        elif total_bands_found == 1:
            # We found exactly one band. Target that one.
            # It's the first (and only) element in our list, at index 0.
            crop_y = found_band_positions[0]
        else:
            print("\n[!] Did not find any prominent white bands matching criteria.")
            return image

        # Perform the crop
        crop_box = (0, 0, image.width, crop_y)
        cropped_image = image.crop(crop_box)
        return cropped_image

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
            print("Successfully parsed JSON string.")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print("The cleaned string might still be invalid JSON.")
            data_dict = None

        return data_dict

    @staticmethod
    def compute_data(id, data_dict):
        MM_TO_MIL = 39.4

        min_coplanarity = 0
        if 'aaa' in data_dict['table']:
            max_coplanarity = round(
                data_dict['table']['aaa']['MAX'] * MM_TO_MIL, 5)
        elif 'bbb' in data_dict['table']:
            max_coplanarity = round(
                data_dict['table']['bbb']['MAX'] * MM_TO_MIL, 5)
        elif 'ccc' in data_dict['table']:
            max_coplanarity = round(
                data_dict['table']['ccc']['MAX'] * MM_TO_MIL, 5)
        elif 'ddd' in data_dict['table']:
            max_coplanarity = round(
                data_dict['table']['ddd']['MAX'] * MM_TO_MIL, 5)
        else:
            max_coplanarity = round(
                data_dict['table']['eee']['MAX'] * MM_TO_MIL, 5)
        pitch_error = 0
        nom_ball_width = round(
            data_dict['table']['øb']['NOM'] * MM_TO_MIL, 5)
        nom_pitch = round(data_dict['table']['e'] * MM_TO_MIL, 5)
        ball_width_error = round((data_dict['table']['øb']['MAX'] -
                                  data_dict['table']['øb']['NOM']) * MM_TO_MIL, 5)
        ball_quality = 75

        result = dict()
        result['BGA Pkg Type'] = data_dict['id']
        result['Coplanarity_min'] = min_coplanarity
        result['Coplanarity_max'] = max_coplanarity
        result['Pitch Error_+/-'] = pitch_error
        result['Nominal Pitch_nom'] = nom_pitch
        result['Ball Width_Error +/-'] = ball_width_error
        result['Nom. Ball Width_nom'] = nom_ball_width
        result['Ball Quality_%'] = ball_quality
        return result

    @staticmethod
    def similarity_score(code_1, code_2):
        clean_code_1 = re.sub(r'[^a-zA-Z0-9]', '', code_1)
        clean_code_2 = re.sub(r'[^a-zA-Z0-9]', '', code_2)
        return fuzz.ratio(clean_code_1, clean_code_2)

    @staticmethod
    def read_and_prepare_data(csv_path):
        """
        Reads a CSV with a multi-level header and prepares data for processing.

        Args:
            csv_path (str): The path to the CSV file.

        Returns:
            tuple: A tuple containing:
                - processing_df (pd.DataFrame): DataFrame with simplified, clean column names.
                - original_df (pd.DataFrame): The original DataFrame with its MultiIndex columns.
                - column_map (dict): A mapping from clean names to original MultiIndex tuples.
        """
        df = pd.read_csv(csv_path, header=[0, 2])

        original_columns = df.columns
        top_level = original_columns.get_level_values(0)
        bottom_level = original_columns.get_level_values(1)

        top_level_filled = pd.Series(top_level).replace(
            to_replace=r'^Unnamed:.*', value=np.nan, regex=True).ffill().str.strip()
        bottom_level_cleaned = pd.Series(bottom_level).str.strip()

        new_columns = []
        for top, bottom in zip(top_level_filled, bottom_level_cleaned):
            if top == bottom or bottom.startswith('Unnamed') or top.startswith('Unnamed'):
                new_columns.append(top)
            else:
                new_columns.append(f"{top}_{bottom}")

        column_map = dict(zip(new_columns, original_columns))

        processing_df = df.copy()
        processing_df.columns = new_columns

        na_rows = processing_df.isnull().all(axis=1)
        processing_df = processing_df[~na_rows]
        df = df[~na_rows]

        return processing_df, df, column_map
