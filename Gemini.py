from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
from tqdm import tqdm
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import csv
import re
from Utils import Utils


class geminiFlashLite:
    def __init__(self, mappings_file=None):
        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. Please set it.")

        genai.configure(api_key=api_key)
        self.MODEL_NAME = 'gemini-2.0-flash-lite-001'
        self.model = genai.GenerativeModel(model_name=self.MODEL_NAME)
        if not mappings_file:
            self.mappings = {}
        else:
            with open(mappings_file, 'r') as f:
                loaded_mappings = json.load(f)
            self.mappings = loaded_mappings

    def getMapping(self, targetId):
        print("Processing mappings...")
        # Search through mappings
        print("Looking through cache...")
        targetIds = Utils.standardize_id(targetId)
        for key, val in self.mappings.items():
            if key in targetIds:
                return self.mappings[key][0]

        initial_history = [
            {'role': 'user', 'parts': [
                "I will be providing you an image. Help me to extract the model name found at the top of the image. Only include the model name in your reponse without any other text."]},
        ]
        chat = self.model.start_chat(history=initial_history)

        print("Looking through files...")
        cachedIds = set(item for sublist in self.mappings.values()
                        for item in sublist)

        idFound = False
        for filename in os.listdir("./jpgs"):
            image_path = os.path.join("./jpgs", filename)
            # Skip if mapping is already stored
            if image_path in cachedIds:
                continue

            img = Image.open(image_path)
            img = Utils.process_image(img)
            id_list = self.extractId(chat, img)

            # update mappings
            for id in id_list:
                if id in self.mappings:  # Appends the full pathname
                    self.mappings[id].append(image_path)
                else:
                    self.mappings[id] = [image_path]

            idFound = False
            for processedTargetId in targetIds:
                if processedTargetId in id_list:
                    idFound = True
                    break
            if idFound:
                break

        # save mappings
        save_dir = os.path.dirname("idFilenameMap.json")
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open("idFilenameMap.json", 'w') as f:
            json.dump(self.mappings, f, indent=4)

        if idFound:
            return image_path
        else:
            return None

    def extractId(self, chat, img):
        print("Querying Id from bot")
        current_message_content = [
            img
        ]
        response = chat.send_message(current_message_content)
        id = response.text
        return Utils.standardize_id(id)

    def extractImageTable(self, targetId):
        path = self.getMapping(targetId)
        if not path:
            print("Image with correct targetID not found.")
            return {"status": 404}

        initial_history = [
            {'role': 'user', 'parts': [
                "I will be providing you an image. Help me to extract the table (with the key for json as 'table') from the bottom left of the image in JSON for futher computation. Also include the model name found at the top of the image. The primary keys to use are the Symbols on the left column (A, A1, A2, D/E, D1/E1, e, øb, aaa, ccc, ddd, eee, M) and the secondary keys to use are MIN, NOM, MAX if there are multiple values for these fields. For example, primary key D/E, D_1/E_1, e & M only has 1 column and therefore, do not include secondary keys for them (Do not have nested keys of the same key). Ignore the last column named NOTE."]},
        ]

        current_text_input = "Only provide me with the table data in numerical form or null for columns with no values, without including any text in your response so that I can simply take your response and do json.loads(string)."

        print("Extracting image table...")
        try:
            chat = self.model.start_chat(history=initial_history)

            img = Image.open(path)
            img = Utils.process_image(img)

            current_message_content = [
                current_text_input,
                img
            ]

            response = chat.send_message(current_message_content)

            if not response.text:
                print("FAIL", path)
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    print(
                        f"Reason for blocking: {response.prompt_feedback.block_reason}")

        except Exception as e:
            print(f"An error occurred: {e}")

        # Log response
        with open("history.log", "a") as log_file:
            log_file.write(str(response))
            log_file.write("\n-----\n")

        return response
