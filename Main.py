from Gemini import GeminiFlash
from Utils import Utils
import sys


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Main.py <csv File Path>")
        exit(1)

    csv_file_path = str(sys.argv[1])

    bot = GeminiFlash(model_name="gemini-2.5-flash",
                      cache='cache.json', cache_refresh=False)

    data, result_csv, column_map = Utils.read_and_prepare_data(
        csv_file_path)

    ids_to_search = data['BGA Pkg Type']
    id_column_original = column_map['BGA Pkg Type']

    for id_value in ids_to_search:
        response = bot.get_similar_id(id_value)
        if response:
            print(
                f"Most similar ID for {id_value} found in images:\n{response}\n\n")

            new_row_data = Utils.compute_data(id_value, response)

            if new_row_data:
                row_index = result_csv[result_csv[id_column_original]
                                       == id_value].index

                # Update the result_csv DataFrame
                for clean_col_name, value in new_row_data.items():
                    if clean_col_name in column_map:
                        original_col = column_map[clean_col_name]
                        result_csv.loc[row_index, original_col] = value
                    else:
                        print(
                            f"Warning: Column '{clean_col_name}' from compute_data not in original CSV.")
        else:
            print(f"{id_value} not found in images!\n\n")

    old_columns = result_csv.columns
    new_levels = []

    # Replace Unnamed columns with ''
    for level in old_columns.levels:
        new_level = level.str.replace(r'^Unnamed.*', '', regex=True)
        new_levels.append(new_level)

    result_csv.columns = old_columns.set_levels(new_levels)
    output_path = "./data/new_data.csv"
    result_csv.to_csv(output_path, index=False)

    print(f"\nExported data to '{output_path}'")

    print("\n--- Original DataFrame (for output) ---")
    print(result_csv)

    print('\n')
    bot.estimate_cost()
