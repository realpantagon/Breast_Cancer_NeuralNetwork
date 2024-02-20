import pandas as pd

# Load the dataset
column_names = [
    "id", "clump_thickness", "uniform_cell_size", "uniform_cell_shape",
    "marginal_adhesion", "single_epithelial_size", "bare_nuclei", "bland_chromatin",
    "normal_nucleoli", "mitoses", "class"
]

# Load the dataset with column names
df = pd.read_csv("breast-cancer-wisconsin.data", names=column_names)

# Replace '?' with NaN
df.replace('?', pd.NA, inplace=True)

# Convert 'bare_nuclei' to numeric (as it might be stored as strings)
df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'], errors='coerce')

# Exclude the 'id' column from the processing
df.drop('id', axis=1, inplace=True)

# Fill missing values with the mode of the respective column as integers
df = df.apply(lambda x: x.fillna(x.mode()[0]).astype(int), axis=0)

# Write the preprocessed dataset to a new CSV file
df.to_csv("preprocess.csv", index=False, header=False)

print("Preprocessing completed. New dataset saved to preprocess.csv")
