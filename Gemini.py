from PIL import Image
import os
import json
import google.generativeai as genai
from google.genai import types
from dotenv import load_dotenv
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
        self.__cache_path = cache
        self.__image_folder = image_folder
        self.__MODEL_NAME = model_name
        self.__cache_refresh = cache_refresh

        self.__prompt_token_count_img = 0
        self.__candidates_token_count_img = 0
        self.__query_count_img = 0

        self.__prompt_token_count_id = 0
        self.__candidates_token_count_id = 0
        self.__query_count_id = 0

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

        self.__model = genai.GenerativeModel(
            model_name=self.__MODEL_NAME, system_instruction=system_instruction)
        self.__id_model = genai.GenerativeModel(
            model_name=self.__MODEL_NAME)
        # self.chat = self.model.start_chat(history=[])

        try:
            with open(cache, 'r', encoding='utf-8') as f:
                self.__cache = json.load(f)
        except FileNotFoundError:
            print(f"Cache file '{cache}' not found. Initializing a new cache.")
            self.__cache = {}
        except json.JSONDecodeError:
            print(
                f"Cache file '{cache}' is empty or not valid JSON. Initializing a new cache.")
            self.__cache = {}

        if self.__cache_refresh:
            self.__refresh_cache()
        return

    def get_similar_id(self, target_id):
        if self.__cache.keys() and (matched_id := self.__fuzz_match_id(target_id)):
            return matched_id
        print("Verifying with Gemini...")
        if self.__cache.keys() and (matched_id := self.__gemini_match_id(target_id)):
            return matched_id
        return None

    def __gemini_match_id(self, target_id):
        """
        Finds a similar ID from the cache using a Gemini model.

        This function is designed to be robust against common LLM output variations
        and prevents KeyErrors by validating the model's response.
        """
        if not self.__cache:
            return None

        id_list = list(self.__cache.keys())

        prompt = f"""
        You are a precise and silent ID matching tool. Your purpose is to find the single best match for a target ID from a list.

        Your instructions are absolute:
        1.  You will analyze the 'ID to match' and compare it against the 'ID list'.
        3.  If the result is a clear, unambiguous match, you will return the original ID from the list.
        4.  If there is no clear match, you will return the exact string 'NA'.

        Your output must be ONLY the matched ID or the string 'NA'. Do not provide any explanation, preamble, justification, or any text other than the result.

        ---
        Example 1:
        ID list = ['FF(G)/EF1152', 'FF(G)1152 (VIRTEX-4: XQ4VFX100)', 'CLIENT-789-C']
        ID to match = 'FF1152'
        Matched ID from list = ?

        FF(G)/EF1152

        Example 2:
        ID list = ['FF(G)/EF1152', 'FF(G)1152 (VIRTEX-4: XQ4VFX100)', 'CLIENT-789-C']
        ID to match = 'FF1152 (XQ4VFX100)'
        Matched ID from list = ?

        FF(G)1152 (VIRTEX-4: XQ4VFX100)

        Example 3 (No Match):
        ID list = ['PROD-A-999', 'PROD-B-101', 'DEV-C-500']
        ID to match = 'cus'
        Matched ID from list = ?

        NA
        ---

        Here is your task:

        ID list = {id_list}
        ID to match = {target_id}
        Matched ID from list = ?
        """

        try:
            response = self.__id_model.generate_content([prompt])
            self.__query_count_id += 1
            self.__prompt_token_count_id += response.usage_metadata.prompt_token_count
            self.__candidates_token_count_id += response.usage_metadata.candidates_token_count
            derived_id = response.text.strip()
            print("Derived_id:", derived_id)
            if derived_id in self.__cache:
                return self.__cache[derived_id]
            else:
                return None

        except Exception as e:
            print(f"An error occurred during Gemini ID matching: {e}")
            return None

    def __fuzz_match_id(self, target_id):
        best_match_id = []
        for id, dict_object in self.__cache.items():
            clean_id = Utils.clean_id(id)
            best_match_id.append(
                (id, dict_object, Utils.similarity_score(clean_id, target_id)))
        best_match_id.sort(key=lambda x: x[2], reverse=True)

        if best_match_id[0][2] >= 85:
            print("Old similarity score:", Utils.similarity_score(
                best_match_id[0][0], target_id))
            print("Similarity score:", best_match_id[0][2])
            print("Original Id", best_match_id[0][0], "; Clean Id", Utils.clean_id(
                best_match_id[0][0]))
            return self.__cache[best_match_id[0][0]]
        else:
            print("NO MATCH - Similarity score:", best_match_id[0][2])
            # print(self.cache[best_match_id[0][0]])
            return None

    def __refresh_cache(self):
        print("Looking through images...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        processed_image_folder = os.path.join(script_dir, "processed_images")
        os.makedirs(processed_image_folder, exist_ok=True)

        for filename in os.listdir(self.__image_folder):
            path_to_check = os.path.join(processed_image_folder, filename)
            image_path = os.path.join(self.__image_folder, filename)
            if os.path.exists(path_to_check) or os.path.isdir(image_path):
                continue
            else:
                img = Image.open(image_path)
                img = Utils.process_image(img)
                img.save(os.path.join(processed_image_folder, filename))

        cached_paths = set(value['image_path']
                           for value in self.__cache.values())

        undefined_tables_path = os.path.join(self.__image_folder, "undefined")
        os.makedirs(undefined_tables_path, exist_ok=True)
        files_to_remove = []
        files_to_move = []
        for filename in os.listdir(processed_image_folder):
            processed_image_path = os.path.join(
                processed_image_folder, filename)
            image_path = os.path.join(self.__image_folder, filename)

            # Skip if mapping is already stored
            if image_path in cached_paths:
                # print(f"{filename} already in cache!")
                continue

            img = Image.open(processed_image_path)

            print(f"Sending {filename} to bot...")
            response = self.__model.generate_content([img])
            self.__query_count_img += 1
            self.__prompt_token_count_img += response.usage_metadata.prompt_token_count
            self.__candidates_token_count_img += response.usage_metadata.candidates_token_count
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
                if content['id'] in self.__cache:
                    continue

            # Update cache
            content["image_path"] = image_path
            self.__cache[content['id']] = content

            try:
                with open(self.__cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.__cache, f, ensure_ascii=False,
                              indent=4, sort_keys=False)
                print("Save successful.")
            except IOError as e:
                print(
                    f"Error: Could not write to file {self.__cache_path}. Reason: {e}")

        for file in files_to_move:
            shutil.move(file, undefined_tables_path)

        for file in files_to_remove:
            os.remove(file)

        print("Done looking through!\n")
        return

    def estimate_cost(self):
        if self.__MODEL_NAME == "gemini-2.5-flash":
            print("\n====================")
            if self.__cache_refresh:
                print("IMG PROCESSING")
                img_input_cost = self.__prompt_token_count_img * 0.3 / 1_000_000
                img_output_cost = self.__candidates_token_count_img * 2.5 / 1_000_000
                print(f"    Total cost - {self.__query_count_img} Queries")
                print("         US$", round(img_input_cost + img_output_cost, 8))
                print("")
                print("     Average cost per query")
                print("         US$", [round((img_input_cost +
                      img_output_cost)/self.__query_count_img, 8) if self.__query_count_img != 0 else 0][0])
                print("--------------------")
            else:
                img_input_cost = 0
                img_output_cost = 0
            print("ID MATCHING")
            id_input_cost = self.__prompt_token_count_id * 0.3 / 1_000_000
            id_output_cost = self.__candidates_token_count_id * 2.5 / 1_000_000
            print(f"    Total cost - {self.__query_count_id} Queries")
            print("         US$", round(id_input_cost + id_output_cost, 8))
            print("")
            print("     Average cost per query")
            print("         US$", [round((id_input_cost +
                  id_output_cost)/self.__query_count_id, 8) if self.__query_count_id != 0 else 0][0])
            print("--------------------")
            print("TOTAL COST")
            print("     US$", round(img_input_cost + img_output_cost +
                  id_input_cost + id_output_cost, 8))
            print("====================\n")
        return
