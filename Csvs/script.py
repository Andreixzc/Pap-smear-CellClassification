import pandas as pd

# Ler o arquivo CSV
df = pd.read_csv("hu_moments.csv")

# Alterar a coluna 'label' para 0 ou 1
df["label"] = df["label"].apply(
    lambda x: 0 if x == "Negative for intraepithelial lesion" else 1
)

# Salvar o novo arquivo CSV
df.to_csv("hu_moments_modified.csv", index=False)
