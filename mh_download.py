import os
import pandas as pd
import requests
from tqdm import tqdm

# Change this to the directory where you want everything to be saved
DOWNLOAD_FOLDER = "./Hu_Imagenet"

def main(csv_path: str):
    """
    Reads a CSV file from `csv_path` that contains columns:
    [image_link, transformation, ground_truth, human_label, IQA].

    Steps:
    1. Remove rows where 'transformation' == 'original'.
    2. Remove rows where 'human_label' != 'ground_truth'.
    3. Remove duplicates.
    4. Convert 'not_car' to 0 and 'car' to 1 in 'ground_truth', and store this in 'label'.
    5. Save filtered data to 'filtered_data.csv' in DOWNLOAD_FOLDER.
    6. Download all images into 'images' folder inside DOWNLOAD_FOLDER.
    """
    # Make sure the DOWNLOAD_FOLDER exists
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    images_folder = os.path.join(DOWNLOAD_FOLDER, "images")
    os.makedirs(images_folder, exist_ok=True)

    # 1. Read the CSV
    df = pd.read_csv(csv_path)

    # 2. Filter out rows where transformation == 'original'
    df = df[df["transformation"] != "original"]

    # 3. Keep rows where human_label == ground_truth
    df = df[df["human_label"] == df["ground_truth"]]

    # 4. Remove duplicates
    df.drop_duplicates(inplace=True)

    # 5. Map ground_truth to numeric labels
    label_map = {
        "not car": 0,
        "car": 1
    }
    df["label"] = df["ground_truth"].map(label_map)

    # 6. Save the filtered dataframe to CSV
    filtered_csv_path = os.path.join(DOWNLOAD_FOLDER, "filtered_data.csv")
    df.to_csv(filtered_csv_path, index=False)

    # 7. Download images
    print("Downloading images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_url = row["image_link"]
        # Use a consistent naming scheme. Here, we'll just use the CSV index as the filename.
        img_filename = os.path.join(images_folder, f"{idx}.jpg")

        # Download the image
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            with open(img_filename, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")

    print(f"Filtered CSV saved to: {filtered_csv_path}")
    print(f"Images saved in: {images_folder}")

if __name__ == "__main__":
    # Example usage
    # Replace 'your_csv_file.csv' with the path to your actual CSV file
    main("imagenet_experiment_results.csv")
