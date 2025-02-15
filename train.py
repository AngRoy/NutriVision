# train.py: Contains the training loop and main execution for training the model.
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import load_food_nutrition_mapping, load_fv_mapping, load_fastfood_mapping, UnifiedFoodDataset
from models import NutriVisionNetMultiHead

def train_model(model, dataloader, num_epochs, optimizer, device):
    model.train()
    mse_loss = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images, targets, sources, class_names = batch
            batch_loss = 0.0
            optimizer.zero_grad()
            for i in range(len(images)):
                image = images[i]
                target = targets[i].to(device).unsqueeze(0)
                source = sources[i]
                pred, caption = model(image, source)
                loss = mse_loss(pred, target)
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()
            pbar.set_postfix({"loss": batch_loss/len(images)})
            running_loss += batch_loss
        avg_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
    print("Training complete.")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nutrition_dir = "/kaggle/input/food-nutrition-dataset/FINAL FOOD DATASET"
    fruits_path = "/kaggle/input/fruits-and-vegetables-nutritional-values/fruits.csv"
    vegetables_path = "/kaggle/input/fruits-and-vegetables-nutritional-values/vegetables.csv"
    fastfood_path = "/kaggle/input/fastfood-nutrition/fastfood.csv"
    images_root = "/kaggle/input/food41/images"
    mapping_food, cols_food = load_food_nutrition_mapping(nutrition_dir)
    print(f"Food Nutrition: {len(mapping_food)} items, target dim = {len(cols_food)}")
    mapping_fv, cols_fv = load_fv_mapping(fruits_path, vegetables_path)
    print(f"Fruits & Vegetables: {len(mapping_fv)} items, target dim = {len(cols_fv)}")
    mapping_fastfood, cols_fastfood = load_fastfood_mapping(fastfood_path)
    print(f"Fastfood: {len(mapping_fastfood)} items, target dim = {len(cols_fastfood)}")
    dataset = UnifiedFoodDataset(images_root, mapping_fastfood, mapping_fv, mapping_food, transform=None)
    print(f"Unified dataset size: {len(dataset)} samples")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    food_nutrition_dim = len(cols_food)
    fv_dim = len(cols_fv)
    fastfood_dim = len(cols_fastfood)
    model = NutriVisionNetMultiHead(food_nutrition_dim, fv_dim, fastfood_dim, device=device, fine_tune_clip_image=True)
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    num_epochs = 1
    train_model(model, dataloader, num_epochs, optimizer, device)
    torch.save(model.state_dict(), "nutrivision_multihand.pt")
    print("Model saved as nutrivision_multihand.pt")

if __name__ == "__main__":
    main()
