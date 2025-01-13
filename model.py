import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
import timm
from transformers import AutoModelForCausalLM, AutoTokenizer

class Projection(nn.Module):
    def __init__(self, d_in, d_out, device, p = 0.5, last_layer=False) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False).to(device)
        self.linear2 = nn.Linear(d_out, d_out, bias=False).to(device)
        self.layer_norm = nn.Identity() if last_layer else nn.LayerNorm(d_out).to(device)
        self.drop = nn.Dropout(p).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class VisionLanguageModel(nn.Module):
    def __init__(self, vision_model, text_model, num_projection_layers=4):
        super().__init__()
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.vision_model = timm.create_model(vision_model, pretrained=True).to(self.device)
        self.text_model = AutoModelForCausalLM.from_pretrained(text_model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.num_projection_layers = num_projection_layers
        self.prepend_embeddings, self.postpend_embeddings = self.get_boundary_embeddings()
    
    def get_boundary_embeddings(self):
        template = "<|im_start|>user**\n**Caption the following image.<|im_end|>**\n**<|im_start|>assistant"
        input_ids = self.tokenizer(template, return_tensors="pt").input_ids.to(self.device)
        self.tokenizer.eos_token = "<|im_end|>"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        eos_token_index = (
            input_ids[0] == self.tokenizer.eos_token_id
        ).nonzero(as_tuple=True)[0].item()
        self.text_embeddings = self.text_model.get_input_embeddings()(
            self.tokenizer(template, return_tensors="pt").input_ids.to(self.device)
        ).detach()

        return self.text_embeddings[:, :eos_token_index], self.text_embeddings[:, eos_token_index:]

    def projection_layers(self, d_in, d_out, layers):
        sequential_layers = []
        for i in range(layers - 1):
            sequential_layers.append(Projection(d_in, d_in, self.device))
            sequential_layers.append(nn.GELU())
            sequential_layers.append(nn.LayerNorm(d_in).to(self.device))
        sequential_layers.append(Projection(d_in, d_out, self.device, last_layer=True))
        return nn.Sequential(*sequential_layers)
        
    def forward(self, images, captions):
        images = images.to(self.device)
        tokenized_captions = self.tokenizer(
            captions,
            padding='max_length',  
            truncation=True,
            max_length=196,
            return_tensors="pt"
        ).to(self.device)
        caption_embeddings = self.text_model.get_input_embeddings()(tokenized_captions.input_ids).detach()
        original_image_tokens = einops.rearrange(self.vision_model.forward_features(images), "bs dim w h -> bs (w h) dim")
        project = self.projection_layers(original_image_tokens.shape[-1], self.text_embeddings.shape[-1], self.num_projection_layers)
        image_tokens = project(original_image_tokens)
        
        embeddings = torch.cat(
            [
                self.prepend_embeddings.to(self.device).expand(len(images), -1, -1),
                image_tokens,
                self.postpend_embeddings.to(self.device).expand(len(images), -1, -1),
                caption_embeddings,
            ],
            dim=1,
        )
        return embeddings, tokenized_captions, embeddings.shape[1] - caption_embeddings.shape[1]
    
    def generate(self, images, generator_kwargs={"max_new_tokens":25}):
        images = images.to(self.device)
        original_image_tokens = einops.rearrange(self.vision_model.forward_features(images), "bs dim w h -> bs (w h) dim")
        project = self.projection_layers(original_image_tokens.shape[-1], self.text_embeddings.shape[-1], self.num_projection_layers)
        image_tokens = project(original_image_tokens)
        embeddings = torch.cat(
            [
                self.prepend_embeddings.to(self.device).expand(len(images), -1, -1),
                image_tokens,
                self.postpend_embeddings.to(self.device).expand(len(images), -1, -1),
            ],
            dim=1,
        )
        attention_mask = torch.ones(1, self.text_embeddings.shape[1] + image_tokens.shape[1]).to(self.device).expand(len(images), -1)
        token_ids = self.text_model.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            **generator_kwargs
        )[0]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

class VisionLanguageLoss(nn.Module):
  def __init__(self, text_model):
    super().__init__()
    self.label_mask = -100
    self.device = "cpu" if not torch.cuda.is_available() else "cuda"
    self.text_model = AutoModelForCausalLM.from_pretrained(text_model).to(self.device)

  def forward(self, embeddings, tokenized_captions, base_attn_dim):
    attn_mask = torch.ones(1, base_attn_dim)
    labels = torch.full((1, attn_mask.shape[1]), self.label_mask)
    attention_mask = torch.cat(
        [
            attn_mask.to(self.device).expand(len(embeddings), -1), 
            tokenized_captions.attention_mask
        ], 
        dim=1
    )
    labels = torch.cat(
        [
            labels.to(self.device).expand(len(embeddings), -1), 
            tokenized_captions.input_ids.clone()
        ],
        dim=1,
    )
    labels[attention_mask == 0] = self.label_mask

    return self.text_model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        labels=labels,
    ).loss