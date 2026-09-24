# I created this file for all the preprocessing requiremnts to be done 

from pathlib import Path
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split

import shutil

DATA_DIR = Path("data/RealWaste")
TARGET_DIR = Path("data/RealWaste_resized")
TARGET_SIZE = (64,64)
SPLIT_DIR = Path("data/RealWaste_split") # splitted and resized images also added to separate folders for future use

# resizing logic
for class_folder in DATA_DIR.iterdir():
    if class_folder.is_dir():
        print(class_folder.name) # all the folders (classes)
        target_class_dir = TARGET_DIR / class_folder.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_folder.iterdir():
            #print(class_folder.name, img_path) # to check all image files are includes and in jpg extension
            if img_path.is_file():
                try:
                    with Image.open(img_path) as img:
                        img_resized = img.resize(TARGET_SIZE, Image.LANCZOS) #resizing command
                        save_path = target_class_dir / img_path.name
                        img_resized.save(save_path)
                except Exception as e:
                    print(f" {e} Error occured during resizing")

#in order to create the splited manifest I created an dataframe woth labeled data from resized folders 
records = []
for class_dir in TARGET_DIR.iterdir():
    if class_dir.is_dir():
        class_name = class_dir.name
        for img_path in class_dir.iterdir():
            if img_path.is_file():
                records.append({
                    "filepath": str(img_path),
                    "label": class_name
                }) 

df = pd.DataFrame(records)

print(df.shape)
print(df["label"].value_counts()) # confirmed data counts and shape

"""
splitting -- this was a bit tricky for us as in our data set all classes
didn't have equal instances (instance counts range from 318 to 921 per class), 
beacuse of this first we used stratify parameter in train_test_split to ensure
every catagory(test, train and val) has same proportions as the full dataset.
But still since a class like plastic (~19% of whole data set) can make the model biased in 
training loss function will be made to penalize mistakes on rare classes more heaviliy.
"""

# 70% train
train_df, temp_df = train_test_split(
    df, 
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

#remaining 30% divided into 15% test and validation

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)
print("Train:", train_df.shape);print("Val:", val_df.shape); print("Test:", test_df.shape)

print("\nTrain label distribution:\n", train_df["label"].value_counts(normalize=True))
print("\nVal label distribution:\n", val_df["label"].value_counts(normalize=True))
print("\nTest label distribution:\n", test_df["label"].value_counts(normalize=True))

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
manifest.to_csv("data/manifest.csv", index=False)
#final manifest that will be used to train the model
print(manifest["split"].value_counts())
print(f"\nManifest saved to data/manifest.csv ({len(manifest)} rows)")

# saving splitted, resized images in subfolders
for _, row in manifest.iterrows():
    src_path = Path(row["filepath"])
    split_class_dir = SPLIT_DIR / row["split"] / row["label"]
    split_class_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, split_class_dir / src_path.name)

print(f"\nImages organized into {SPLIT_DIR}/train, /val, /test")