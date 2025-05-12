from Gemini import geminiFlashLite
from Utils import Utils
import sys

if __name__ == "__main__":
    # FF900/FF901/RF900
    if len(sys.argv) != 2:
        print("Usage: python3 Main.py <BGA Pkg Type>")
        exit(1)
    bot = geminiFlashLite('idFilenameMap.json')
    response = bot.extractImageTable(str(sys.argv[1]))
    print(Utils.clean_json_response(response.text))
