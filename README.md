## Zero-Shot Generalization on Food-101: CLIP vs. Supervised Baseline

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/index)
[![Dataset](https://img.shields.io/badge/Dataset-Food--101-blue?style=for-the-badge)](https://huggingface.co/datasets/ethz/food101)

##  Executive Summary
This project benchmarks OpenAI's **[CLIP (ViT-B/32)](https://huggingface.co/openai/clip-vit-base-patch32)** zero-shot capability against a fully supervised **[ResNet-50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)** on the **[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)** dataset, exploring how text prompt engineering impacts classification performance without task-specific retraining.

##  Performance Comparison

| Model | Setup | Train Time | Top-1 Accuracy |
| :--- | :--- | :--- | :--- |
| **ResNet-50** | Supervised Fine-Tuning | ~45 mins | **84.1%** |
| **CLIP (ViT-B/32)** | Zero-Shot (Basic Prompt) | 0 mins | 78.2% |
| **CLIP (ViT-B/32)** | Zero-Shot (Prompt Ensemble) | 0 mins | **82.5%** |

##  Tech Stack & Links
* **Deep Learning Framework:** [PyTorch](https://pytorch.org/)
* **Model Hub & Transformers:** [HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/clip)
* **Dataset:** [HuggingFace Datasets - Food101](https://huggingface.co/datasets/ethz/food101)

##  Academic References
1. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*. [[Paper]](https://arxiv.org/abs/2103.00020)
2. Bossard, L., et al. (2014). Food-101–mining discriminative components for food recognition. *ECCV*. [[Paper]](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
3. Zhou, K., et al. (2022). Learning to prompt for vision-language models. *IJCV*. [[Paper]](https://arxiv.org/abs/2109.01134)
