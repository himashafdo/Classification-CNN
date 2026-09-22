# I created this file for all the preprocessing requiremnts to be done 

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/RealWaste")

records = []
for class_folder in DATA_DIR.iterdir():
    if class_folder.is_dir():
        print(class_folder.name) # all the folders (classes)
        for img_path in class_folder.iterdir():
            #print(class_folder.name, img_path) # to check all image files are includes and in jpg extension
            records.append({
                "filepath": str(img_path),
                "label": class_folder.name
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