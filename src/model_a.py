


from pathlib import Path
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/RealWaste")
TARGET_DIR = Path("data/RealWaste_resized")
TARGET_SIZE = (64,64)

sample = next(TARGET_DIR.rglob("*.jpg"))
with Image.open(sample) as img:
    print("Sample image size:", img.size)