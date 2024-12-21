import os
import shutil
import random


def split_folders(source_dir, dest_base_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    Split folders from source directory into train/val/test sets according to given ratio.
    Args:
        source_dir (str): Path to source directory containing subfolders
        dest_base_dir (str): Path to destination base directory
        split_ratio (tuple): Ratio for train/val/test split (default: 0.8, 0.1, 0.1)
    """
    # Create destination directories
    split_names = ["train", "val", "test"]
    dest_dirs = {}
    for name in split_names:
        dest_path = os.path.join(dest_base_dir, name)
        os.makedirs(dest_path, exist_ok=True)
        dest_dirs[name] = dest_path

    # Get list of all subfolders
    subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]

    # Shuffle the list to ensure random distribution
    random.shuffle(subfolders)

    # Calculate split sizes
    total = len(subfolders)
    train_size = int(total * split_ratio[0])
    val_size = int(total * split_ratio[1])

    # Split the folders list
    train_folders = subfolders[:train_size]
    val_folders = subfolders[train_size : train_size + val_size]
    test_folders = subfolders[train_size + val_size :]

    # Create a dictionary mapping split names to folder lists
    splits = {"train": train_folders, "val": val_folders, "test": test_folders}

    # Move folders to their respective destinations
    for split_name, folders in splits.items():
        for folder in folders:
            folder_name = os.path.basename(folder)
            dest_path = os.path.join(dest_dirs[split_name], folder_name)
            shutil.move(folder, dest_path)
            print(f"Moved {folder_name} to {split_name}")

    # Print summary
    print("\nSplit Summary:")
    print(f"Total folders: {total}")
    for split_name, folders in splits.items():
        print(f"{split_name}: {len(folders)} folders ({len(folders)/total:.1%})")


if __name__ == "__main__":
    # Example usage
    source_directory = (
        "./data/canada-2021"  # Source directory containing the 3000 subfolders
    )
    destination_base_directory = "./dataset"  # Where to create train/val/test folders

    split_folders(source_directory, destination_base_directory)
