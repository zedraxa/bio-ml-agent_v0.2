import pandas as pd
import os

# Dosya yolu, betik workspace/ dizininden çalıştırıldığında doğru olacak şekilde güncellendi.
# find komutunun sonucuna göre doğru yol: ./data/processed/kareler.csv
file_path = "data/processed/kareler.csv"

print(f"Current working directory: {os.getcwd()}")
print(f"Attempting to read file from: {file_path}")

try:
    if os.path.exists(file_path):
        print(f"File '{file_path}' exists.")
        df = pd.read_csv(file_path)
        print("File content:")
        print(df.to_string(index=False))
    else:
        print(f"File '{file_path}' does NOT exist at this path according to os.path.exists().")
        # 'data/processed' dizininin içeriğini listele
        processed_dir = os.path.dirname(file_path)
        if os.path.isdir(processed_dir):
            print(f"Contents of {processed_dir}: {os.listdir(processed_dir)}")
        else:
            print(f"Directory {processed_dir} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")