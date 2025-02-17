import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MHSyntheticDataset(Dataset):
    """
    A PyTorch-compatible dataset that uses the CSV generated
    by `mh_download.py`.
    
    This version is suitable when your downloaded image files
    are named with their original CSV index values (e.g. 2.jpg, 10.jpg, ...),
    and thus may have 'gaps' in the sequence.
    """
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Path to the DOWNLOAD_FOLDER containing
                        'filtered_data.csv' and the 'images/' directory.
            transform (callable, optional): Optional transform to be
                                            applied on a sample.
        """
        self.root = root
        self.transform = transform

        # Read the filtered CSV (which should contain a "label" column).
        csv_path = os.path.join(self.root, "filtered_data.csv")
        self.data = pd.read_csv(csv_path)

        
        # Directory where images are saved
        self.images_dir = os.path.join(self.root, "images")
        self.idx_list = sorted([int(f[:-4]) for f in os.listdir(self.images_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        """
        Returns:
            (image, label) where:
              - image is a PIL Image (RGB).
              - label is an integer (0 or 1).
        """
        # Map the dataset index i to the actual row index in the DataFrame
        row_idx = self.idx_list[i]

        # The file for this sample is e.g. "2.jpg" if row_idx == 2
        img_filename = f"{row_idx}.jpg"
        img_path = os.path.join(self.images_dir, img_filename)

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Get the label from the CSV
        label = self.data.loc[i, "label"]  # should be 0 or 1
        ##  if label is nan, return 0
        if pd.isna(label):
            label = 0

        # Apply any optional transform
        if self.transform:
            image = self.transform(image)

        return image, int(label)


if __name__ == "__main__":
    # Example usage:
    # Assuming you have run mh_download.py and now have
    # downloaded_data/filtered_data.csv and downloaded_data/images/*.jpg
    # dataset = MHSyntheticDataset(root="./Hu_Cifar10")
    dataset = MHSyntheticDataset(root="./Hu_Imagenet")

    print("Number of samples in dataset:", len(dataset))

    # for i in range(len(dataset)):
    #     sample_img, sample_label = dataset[i]
    #     print(f"Sample {i} label:", sample_label)

