from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
from tqdm import tqdm
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import csv
import re
import shutil
from Utils import Utils


class GeminiFlash:
    def __init__(self, model_name="gemini-2.5-flash", cache='./cache.json', image_folder='./all_images', cache_refresh=False):
        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. Please set it.")

        genai.configure(api_key=api_key)

        # Make these variables private
        self.cache_path = cache
        self.image_folder = image_folder
        self.MODEL_NAME = model_name
        self.cache_refresh = cache_refresh

        system_instruction = """
        You are a data extraction AI. Your task is to extract information from a technical document into a raw JSON object.
        Follow these rules:
            1.  **`id`**: Extract the complete identification string from the document's top header. If there are sub-headers, include them in the ID. This string might contain parentheses, slashes, or other characters.
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

        self.model = genai.GenerativeModel(
            model_name=self.MODEL_NAME, system_instruction=system_instruction)
        # self.chat = self.model.start_chat(history=[])

        try:
            with open(cache, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            print(f"Cache file '{cache}' not found. Initializing a new cache.")
            self.cache = {}
        except json.JSONDecodeError:
            print(
                f"Cache file '{cache}' is empty or not valid JSON. Initializing a new cache.")
            self.cache = {}

        if self.refresh_cache:
            self.refresh_cache()
        return

    def get_similar_id(self, target_id):
        if (matched_id := self.match_id(target_id)) and self.cache.keys():
            return matched_id
        return None

    def match_id(self, target_id):
        best_match_id = []
        for id, dict_object in self.cache.items():
            best_match_id.append(
                (id, dict_object, Utils.similarity_score(id, target_id)))
        best_match_id.sort(key=lambda x: x[2], reverse=True)

        if best_match_id[0][2] >= 85:
            print("Best score:", best_match_id[0][2])
            return self.cache[best_match_id[0][0]]
        else:
            return None

    def refresh_cache(self):
        # Break the refresh if the ID is found with a high similarity score of > 85
        print("Looking through images...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        processed_image_folder = os.path.join(script_dir, "processed_images")
        os.makedirs(processed_image_folder, exist_ok=True)

        for filename in os.listdir(self.image_folder):
            path_to_check = os.path.join(processed_image_folder, filename)
            image_path = os.path.join(self.image_folder, filename)
            if os.path.exists(path_to_check) or os.path.isdir(image_path):
                continue
            else:
                img = Image.open(image_path)
                img = Utils.process_image(img)
                img.save(os.path.join(processed_image_folder, filename))

        cached_paths = set(value['image_path']
                           for value in self.cache.values())

        undefined_tables_path = os.path.join(self.image_folder, "undefined")
        os.makedirs(undefined_tables_path, exist_ok=True)
        files_to_remove = []
        files_to_move = []
        for filename in os.listdir(processed_image_folder):
            processed_image_path = os.path.join(
                processed_image_folder, filename)
            image_path = os.path.join(self.image_folder, filename)

            # Skip if mapping is already stored
            if image_path in cached_paths:
                print(f"{filename} already in cache!")
                continue

            img = Image.open(processed_image_path)

            print(f"Sending {filename} to bot...")
            response = self.model.generate_content([img])
            content = Utils.clean_json_response(response.text)

            if not content:
                print("Skipping:", filename)
                files_to_move.append(image_path)
                files_to_remove.append(processed_image_path)
                continue

            if not content['table']:
                print("Empty table:", filename)
                print(response.text)
                files_to_move.append(image_path)
                files_to_remove.append(processed_image_path)
                if content['id'] in self.cache:
                    continue

            # Update cache
            content["image_path"] = image_path
            self.cache[content['id']] = content

            # Save after every iteration for free tier only
            try:
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False,
                              indent=4, sort_keys=False)
                print("Save successful.")
            except IOError as e:
                print(
                    f"Error: Could not write to file {self.cache_path}. Reason: {e}")

        for file in files_to_move:
            shutil.move(file, undefined_tables_path)

        for file in files_to_remove:
            os.remove(file)

        return
