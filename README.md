# Zero-Shot Generalization on Food-101: CLIP vs. Supervised Baseline

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/index)
[![Dataset](https://img.shields.io/badge/Dataset-Food--101-blue?style=for-the-badge)](https://huggingface.co/datasets/ethz/food101)

## Executive Summary
This project benchmarks OpenAI's **[CLIP (ViT-B/32)](https://huggingface.co/openai/clip-vit-base-patch32)** zero-shot capabilities against a fully supervised **[ResNet-50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)** baseline on the **[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)** dataset. We explore how text prompt context and ensembling impact zero-shot classification performance without requiring task-specific fine-tuning or retraining.

---

## Performance Comparison

| Model | Setup | Training Time | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | Supervised Fine-Tuning | ~45 mins | **84.1%** | **96.2%** |
| **CLIP (ViT-B/32)** | Zero-Shot (Basic Prompt) | **0 mins** | 78.2% | 93.5% |
| **CLIP (ViT-B/32)** | Zero-Shot (Prompt Ensemble) | **0 mins** | **82.5%** | **95.8%** |

---

## Key Takeaways
1. **Instant Deployment:** Zero-shot CLIP completely eliminates GPU training pipelines while maintaining highly competitive accuracy.
2. **Prompt Engineering:** Structuring domain context (e.g., `"a photo of {class}, a type of food"`) and averaging prompt embeddings provides a **+4.3% accuracy boost** without updating model weights.

---

## Quick Start & Reproduction

### 1. Installation
Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
