import pandas as pd
from sklearn.model_selection import train_test_split

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

# Separate features and target variable
X = df.drop('class', axis=1)
y = df['class']

# Split the dataset into training (70%) and test (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Concatenate features and target variable for both training and test sets
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save training and test datasets to new CSV files
train_df.to_csv("train_dataset.csv", index=False, header=False)
test_df.to_csv("test_dataset.csv", index=False, header=False)

print("Preprocessing completed. New dataset saved to preprocess.csv")
print("Training and test datasets created and saved.")
