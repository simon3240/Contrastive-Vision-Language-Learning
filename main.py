import torch
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

# 1. Setup Device & Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Define Classes & Prompts
classes = ["sushi", "pizza", "ramen", "hamburger", "ice cream"]
prompts = [f"a photo of {c}, a type of food" for c in classes]

# 3. Load Sample Image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.pjg"
# Assuming food image examples are used：
image = Image.open("sample_pizza.jpg")

# 4. Zero-Shot Inference
inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # image-text feature similarity scores
probs = logits_per_image.softmax(dim=1)  # converted to probability values

# 5. Output Result
for class_name, prob in zip(classes, probs[0]):
    print(f"{class_name}: {prob.item()*100:.2f}%")
