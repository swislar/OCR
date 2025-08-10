"""
Image processing utilities for OCR application.
"""
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple

from config import (
    BRIGHTNESS_THRESHOLD, PERCENTAGE_THRESHOLD, REQUIRED_CONSECUTIVE_LINES,
    SEARCH_AREA_RATIO, BLOCK_SIZE, C_VALUE, CONTRAST_ENHANCEMENT,
    TEXT_VISIBILITY_THRESHOLD, MEDIAN_FILTER_SIZE, IMAGE_CROP_RATIO
)
from logger import logger


class ImageProcessor:
    """Handles all image processing operations for OCR."""

    @staticmethod
    def process_image(image: Image.Image) -> Image.Image:
        """
        Process an image for OCR by applying various enhancements.

        Args:
            image: Input PIL Image

        Returns:
            Processed PIL Image
        """
        # Convert to grayscale
        image = image.convert('L')

        # Resize and crop
        width, height = image.size
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        image = image.crop((0, 0, width, int(height * IMAGE_CROP_RATIO)))

        # Apply adaptive thresholding
        open_cv_image = np.array(image)
        processed_cv_image = cv2.adaptiveThreshold(
            open_cv_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            BLOCK_SIZE,
            C_VALUE
        )

        # Convert back to PIL and enhance
        image = Image.fromarray(processed_cv_image)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(CONTRAST_ENHANCEMENT)

        # Apply text visibility threshold
        image = image.point(lambda p: 0 if p <
                            TEXT_VISIBILITY_THRESHOLD else 255)
        image = image.filter(ImageFilter.MedianFilter(size=MEDIAN_FILTER_SIZE))

        # Crop company section
        return ImageProcessor._crop_company_section(image)

    @staticmethod
    def _crop_company_section(
        image: Image.Image,
        brightness_threshold: int = BRIGHTNESS_THRESHOLD,
        percentage_threshold: float = PERCENTAGE_THRESHOLD,
        required_consecutive_lines: int = REQUIRED_CONSECUTIVE_LINES,
        search_area_ratio: float = SEARCH_AREA_RATIO
    ) -> Image.Image:
        """
        Crop everything below a prominent white horizontal band from the bottom.

        Args:
            image: Input PIL Image
            brightness_threshold: Pixel value above which a pixel is considered 'white'
            percentage_threshold: Ratio of white pixels required to classify a row as 'white'
            required_consecutive_lines: How many consecutive white rows define a 'thick' band
            search_area_ratio: The portion of the image from the bottom to search

        Returns:
            Cropped PIL Image or original if no suitable band is found
        """
        image_array = np.array(image.convert('L'))
        height, width = image_array.shape

        start_search_y = int(height * (1 - search_area_ratio))

        consecutive_white_lines = 0
        found_band_positions = []

        # Search for white bands from bottom to top
        for y in range(height - 1, start_search_y, -1):
            row = image_array[y, :]
            white_pixels_count = np.sum(row > brightness_threshold)
            percentage_white = white_pixels_count / width

            is_row_white = percentage_white > percentage_threshold

            if is_row_white:
                consecutive_white_lines += 1
            else:
                if consecutive_white_lines >= required_consecutive_lines:
                    band_top_y = y + 1
                    found_band_positions.append(band_top_y)
                consecutive_white_lines = 0

        # Check if we ended on a white band
        if consecutive_white_lines >= required_consecutive_lines:
            band_top_y = start_search_y + 1
            found_band_positions.append(band_top_y)

        # Determine crop position
        crop_y = height
        total_bands_found = len(found_band_positions)

        if total_bands_found >= 2:
            crop_y = found_band_positions[1]
        elif total_bands_found == 1:
            crop_y = found_band_positions[0]
        else:
            logger.warning(
                "Did not find any prominent white bands matching criteria.")
            return image

        # Crop the image
        crop_box = (0, 0, image.width, crop_y)
        return image.crop(crop_box)

    @staticmethod
    def load_and_process_image(image_path: str) -> Image.Image:
        """
        Load and process an image from file path.

        Args:
            image_path: Path to the image file

        Returns:
            Processed PIL Image
        """
        try:
            image = Image.open(image_path)
            return ImageProcessor.process_image(image)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

    @staticmethod
    def save_processed_image(image: Image.Image, output_path: str) -> None:
        """
        Save a processed image to file.

        Args:
            image: PIL Image to save
            output_path: Output file path
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            logger.debug(f"Saved processed image to {output_path}")
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            raise
