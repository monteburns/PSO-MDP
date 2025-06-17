import pandas as pd
import os

folder_path = "/content/drive/My Drive/Doktora Tezi/Analysis/PhD"

# List all CSV files in the folder
csv_files = ["Gercek_Zamanli_Tuketim-01012016-31122016.csv",
"Gercek_Zamanli_Tuketim-01012017-31122017.csv",
"Gercek_Zamanli_Tuketim-01012018-31122018.csv",
"Gercek_Zamanli_Tuketim-01012019-31122019.csv",
"Gercek_Zamanli_Tuketim-01012020-31122020.csv",
"Gercek_Zamanli_Tuketim-01012021-31122021.csv",
"Gercek_Zamanli_Tuketim-01012022-31122022.csv",
"Gercek_Zamanli_Tuketim-01012023-31122023.csv",
"Gercek_Zamanli_Tuketim-01012024-31122024.csv"]

# Read and concatenate all CSVs
df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Save merged file
output_path = "/content/drive/My Drive/Doktora Tezi/Analysis/PhD/Gerceklesen Tuketim-16-24.csv"
merged_df.to_csv(output_path, index=False)