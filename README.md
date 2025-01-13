# VisionLanguageModel
A VisionLanguageModel implemented using frozen image/text encoders and a custom loss. Used for the image captioning task.

# Overview
Extracts visual features using pretrained vision models (via `timm`), processes text with transformer-based language models (via `transformers`), projects visual features into the text embedding space, and combines image and text embeddings for tasks like caption generation.

# Architecture
The following flowchart illustrates the model's architecture:

```mermaid
flowchart TD
    A[Input Images] --> B[Vision Encoder] --> C[Projection Layers]
    D[Input Captions] --> E[Text Encoder]
    C --> F[Combined Embeddings]
    E --> F
    F --> G[Language Model Decoder]
    G --> H[Generated Captions]

# Usage
Installation
bash
git clone https://github.com/your-repo/vision-language-model.git
cd vision-language-model
pip install torch torchvision timm einops transformers
Training
python
from your_model_file import VisionLanguageModel, VisionLanguageLoss

model = VisionLanguageModel("resnet50", "gpt2")
loss_fn = VisionLanguageLoss("gpt2")

# Training loop
embeddings, tokenized_captions, base_attn_dim = model(images, captions)
loss = loss_fn(embeddings, tokenized_captions, base_attn_dim)
loss.backward()
Inference
python
generated_caption = model.generate(images)
print("Generated Caption:", generated_caption)
