# data_utils.py: Contains functions to load nutrition mappings and the unified dataset.
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def safe_convert(x):
    try:
        return float(x)
    except:
        if isinstance(x, str):
            if "/" in x:
                parts = x.split("/")
                try:
                    nums = [float(p.strip()) for p in parts if p.strip() != '']
                    return sum(nums) / len(nums) if nums else np.nan
                except:
                    return np.nan
            else:
                try:
                    x_clean = ''.join(c for c in x if c.isdigit() or c in ['.', '-'])
                    return float(x_clean) if x_clean != '' else np.nan
                except:
                    return np.nan
        return np.nan

def load_food_nutrition_mapping(nutrition_dir, target_columns=None):
    if target_columns is None:
        target_columns = ["Caloric Value", "Fat", "Carbohydrates"]
    dataframes = []
    for filename in os.listdir(nutrition_dir):
        if filename.endswith(".csv") and filename.startswith("FOOD-DATA-GROUP"):
            file_path = os.path.join(nutrition_dir, filename)
            df = pd.read_csv(file_path)
            df['Source'] = 'food_nutrition'
            dataframes.append(df)
    if len(dataframes) == 0:
        raise FileNotFoundError("No FOOD-DATA-GROUP*.csv files found in the directory.")
    combined_df = pd.concat(dataframes, ignore_index=True)
    if "food" in combined_df.columns:
        combined_df["FoodName"] = combined_df["food"].astype(str).str.lower().str.strip()
    elif "item" in combined_df.columns:
        combined_df["FoodName"] = combined_df["item"].astype(str).str.lower().str.strip()
    else:
        raise ValueError("Neither 'food' nor 'item' column found in food nutrition data.")
    for col in target_columns:
        if col not in combined_df.columns:
            raise ValueError(f"Column '{col}' not found in food nutrition data.")
    grouped = combined_df.groupby("FoodName")[target_columns].mean().reset_index()
    mapping = {}
    for _, row in grouped.iterrows():
        food_name = row["FoodName"]
        target_vector = row[target_columns].values.astype(np.float32)
        mapping[food_name] = target_vector
    return mapping, target_columns

def load_fv_mapping(fruits_path, vegetables_path, target_columns=None):
    if target_columns is None:
        target_columns = ["energy (kcal/kJ)", "water (g)", "protein (g)", "total fat (g)",
                          "carbohydrates (g)", "fiber (g)", "sugars (g)", "calcium (mg)", "iron (mg)"]
    fruits_df = pd.read_csv(fruits_path)
    fruits_df['Source'] = 'fv'
    vegetables_df = pd.read_csv(vegetables_path)
    vegetables_df['Source'] = 'fv'
    combined = pd.concat([fruits_df, vegetables_df], ignore_index=True)
    if "name" not in combined.columns:
        raise ValueError("Column 'name' not found in fruits/vegetables data.")
    combined["FoodName"] = combined["name"].astype(str).str.lower().str.strip()
    for col in target_columns:
        if col not in combined.columns:
            raise ValueError(f"Column '{col}' not found in fruits/vegetables data.")
        combined[col] = combined[col].apply(safe_convert)
    grouped = combined.groupby("FoodName")[target_columns].mean().reset_index()
    mapping = {}
    for _, row in grouped.iterrows():
        food_name = row["FoodName"]
        target_vector = row[target_columns].values.astype(np.float32)
        mapping[food_name] = target_vector
    return mapping, target_columns

def load_fastfood_mapping(fastfood_path, target_columns=None):
    if target_columns is None:
        target_columns = ["calories", "cal_fat", "total_fat", "sat_fat", 
                          "trans_fat", "cholesterol", "sodium", "total_carb"]
    df = pd.read_csv(fastfood_path)
    df['Source'] = 'fastfood'
    if "item" not in df.columns:
        raise ValueError("Column 'item' not found in fastfood data.")
    df["FoodName"] = df["item"].astype(str).str.lower().str.strip()
    for col in target_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in fastfood data.")
    grouped = df.groupby("FoodName")[target_columns].mean().reset_index()
    mapping = {}
    for _, row in grouped.iterrows():
        food_name = row["FoodName"]
        target_vector = row[target_columns].values.astype(np.float32)
        mapping[food_name] = target_vector
    return mapping, target_columns

class UnifiedFoodDataset(Dataset):
    def __init__(self, images_root, mapping_fastfood, mapping_fv, mapping_food, transform=None):
        self.samples = []
        self.transform = transform
        for folder in os.listdir(images_root):
            folder_path = os.path.join(images_root, folder)
            if os.path.isdir(folder_path):
                class_name = folder.lower().strip()
                if class_name in mapping_fastfood:
                    source = "fastfood"
                    target = mapping_fastfood[class_name]
                elif class_name in mapping_fv:
                    source = "fv"
                    target = mapping_fv[class_name]
                elif class_name in mapping_food:
                    source = "food_nutrition"
                    target = mapping_food[class_name]
                else:
                    continue
                for file in os.listdir(folder_path):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(folder_path, file)
                        self.samples.append((img_path, target, source, class_name))
        if len(self.samples) == 0:
            raise ValueError("No images found that match any nutrition mapping.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target, source, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return image, target_tensor, source, class_name
