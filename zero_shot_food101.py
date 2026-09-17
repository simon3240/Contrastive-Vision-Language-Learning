import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# 1. Setup hardware device (GPU / CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load HuggingFace CLIP model and processor
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# 3. Load Food-101 validation split (subset of 1,000 samples for fast execution)
print("Loading Food-101 test dataset...")
dataset = load_dataset("ethz/food101", split="validation")
sample_size = 1000
test_data = dataset.select(range(sample_size))

# Extract all 101 class names
class_names = dataset.features["label"].names

# 4. Define prompt templates for Prompt Ensembling
prompt_templates = [
    "a photo of {}, a type of food.",
    "a delicious photo of {}.",
    "a close-up photo of the food {}.",
    "a dish of {}.",
    "a photo of a small portion of {}.",
]

# 5. Pre-compute text embeddings for all classes to avoid redundant processing
print("Pre-computing text embeddings for all classes...")
model.eval()
with torch.no_grad():
    text_features_list = []

    # Compute L2-normalized text features across all prompt templates
    for template in prompt_templates:
        prompts = [template.format(c.replace("_", " ")) for c in class_names]
        inputs = processor(
            text=prompts, return_tensors="pt", padding=True
        ).to(device)
        text_outputs = model.get_text_features(**inputs)
        # Normalize text embeddings
        text_outputs = text_outputs / text_outputs.norm(
            p=2, dim=-1, keepdim=True
        )
        text_features_list.append(text_outputs)

    # Average text embeddings across prompt variations
    final_text_features = torch.stack(text_features_list).mean(dim=0)
    final_text_features = final_text_features / final_text_features.norm(
        p=2, dim=-1, keepdim=True
    )

# 6. Run zero-shot evaluation loop and compute accuracy metrics
batch_size = 32
correct_top1 = 0
correct_top5 = 0
total_samples = len(test_data)

print(f"Running zero-shot evaluation on {total_samples} samples...")

for i in tqdm(range(0, total_samples, batch_size)):
    batch = test_data[i : i + batch_size]
    images = batch["image"]
    labels = torch.tensor(batch["label"]).to(device)

    # Extract and normalize image features
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(
            p=2, dim=-1, keepdim=True
        )

        # Compute Cosine Similarity matrix (Image-Text logits)
        similarity = image_features @ final_text_features.T

        # Calculate Top-1 and Top-5 predictions
        _, top5_preds = similarity.topk(5, dim=-1)
        top1_preds = top5_preds[:, 0]

        # Accumulate correct prediction counts
        correct_top1 += (top1_preds == labels).sum().item()
        correct_top5 += (
            (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
        )

# 7. Print final evaluation report
top1_acc = (correct_top1 / total_samples) * 100
top5_acc = (correct_top5 / total_samples) * 100

print("\n" + "=" * 40)
print("Zero-Shot Evaluation Results (Food-101)")
print("=" * 40)
print(f"Total Evaluated Samples: {total_samples}")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")
print("=" * 40)
