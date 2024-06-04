import os
import pandas as pd

# Caminho para o arquivo CSV
csv_file_path = 'image_paths_labels.csv'  # Substitua pelo caminho do seu arquivo CSV

# Lê o arquivo CSV
df = pd.read_csv(csv_file_path)

replace_string = 'Negative for intraepithelial lesion'
replace_with = 'Negative'

#replace all occurrences of the string on the csv file
#replace ALL ocurrences of the string on the csv file

df = df.replace(replace_string, replace_with, regex=True)
#save into a csv file:
df.to_csv('image_paths_labels1.csv', index=False)


