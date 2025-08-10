import sys
import pandas as pd
from typing import Optional

from Gemini import GeminiFlash
from data_processor import DataProcessor
from config import OUTPUT_PATH
from logger import logger


class OCRProcessor:
    """Main OCR processing orchestrator."""

    def __init__(self, csv_file_path: str):
        """
        Initialize OCR processor.

        Args:
            csv_file_path: Path to the CSV file to process
        """
        self.csv_file_path = csv_file_path
        self.bot = None
        self.data = None
        self.result_csv = None
        self.column_map = None

    def setup(self) -> None:
        """Setup the processor and load data."""
        # Initialize Gemini client
        self.bot = GeminiFlash(
            model_name="gemini-2.5-flash",
            cache_path='cache.json',
            cache_refresh=True
        )

        # Load and prepare data
        self.data, self.result_csv, self.column_map = DataProcessor.read_and_prepare_data(
            self.csv_file_path
        )

        logger.info("OCR processor setup complete")

    def process_ids(self) -> None:
        """Process all IDs in the dataset."""
        if (self.data is None or self.result_csv is None or self.column_map is None):
            raise ValueError(
                "Processor not properly initialized. Call setup() first.")

        ids_to_search = self.data['BGA Pkg Type']
        id_column_original = self.column_map['BGA Pkg Type']

        logger.info(f"Processing {len(ids_to_search)} IDs")

        for id_value in ids_to_search:
            self._process_single_id(id_value, id_column_original)

    def _process_single_id(self, id_value: str, id_column_original: str) -> None:
        """
        Process a single ID value.

        Args:
            id_value: ID value to process
            id_column_original: Original column name for the ID
        """
        response = self.bot.get_similar_id(id_value)

        if response:
            logger.info(f"Most similar ID for {id_value} found in images")

            new_row_data = DataProcessor.compute_data(id_value, response)

            if new_row_data:
                self._update_result_dataframe(
                    id_value, new_row_data, id_column_original)
            else:
                logger.warning(f"Failed to compute data for {id_value}")
        else:
            logger.warning(f"{id_value} not found in images!")

    def _update_result_dataframe(self, id_value: str, new_row_data: dict, id_column_original: str) -> None:
        """
        Update the result DataFrame with new data.

        Args:
            id_value: ID value being updated
            new_row_data: New data to add
            id_column_original: Original column name for the ID
        """
        row_index = self.result_csv[self.result_csv[id_column_original]
                                    == id_value].index

        for clean_col_name, value in new_row_data.items():
            if clean_col_name in self.column_map:
                original_col = self.column_map[clean_col_name]
                self.result_csv.loc[row_index, original_col] = value
            else:
                logger.warning(
                    f"Column '{clean_col_name}' from compute_data not in original CSV.")

    def save_results(self) -> None:
        """Save processed results to file."""
        if self.result_csv is None:
            raise ValueError("No results to save. Run process_ids() first.")

        # Clean column names
        self.result_csv = DataProcessor.clean_column_names(self.result_csv)

        # Save to file
        DataProcessor.save_dataframe(self.result_csv, OUTPUT_PATH)

        logger.info(f"Results saved to {OUTPUT_PATH}")

    def print_results(self) -> None:
        """Print processing results."""
        if self.result_csv is not None:
            logger.info("--- Original DataFrame (for output) ---")
            print(self.result_csv)
        else:
            logger.warning("No results to display")

    def print_cost_report(self) -> None:
        """Print cost estimation report."""
        if self.bot:
            self.bot.estimate_cost()

    def run(self) -> None:
        """Run the complete OCR processing pipeline."""
        try:
            logger.info("Starting OCR processing pipeline")

            # Setup
            self.setup()

            # Process IDs
            self.process_ids()

            # Save results
            self.save_results()

            # Print results
            self.print_results()

            # Print cost report
            self.print_cost_report()

            logger.info("OCR processing pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in OCR processing pipeline: {e}")
            raise


def main():
    """Main entry point."""
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Check command line arguments
    if len(sys.argv) != 2:
        logger.error("Usage: python Main_refactored.py <csv File Path>")
        sys.exit(1)

    csv_file_path = str(sys.argv[1])

    # Create and run processor
    processor = OCRProcessor(csv_file_path)
    processor.run()


if __name__ == "__main__":
    main()
