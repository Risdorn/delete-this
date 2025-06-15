from BLIP.models.blip_itm import blip_itm
from transformers import BlipProcessor, BlipForConditionalGeneration

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size=384
sample_size = 1
df_path = f"kinetic_frames_{sample_size}.csv"
out_path = f"kinetic_text_embeddings_{sample_size}.csv"
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
model_url = "model_base_retrieval_coco.pth"
model = blip_itm(model_url, image_size=image_size)
processor = BlipProcessor.from_pretrained("blip-image-captioning-base", use_fast=True)
caption_model = BlipForConditionalGeneration.from_pretrained("blip-image-captioning-base")
caption_model.to(device)
caption_model.eval()

text_encoder = model.text_encoder
text_proj = model.text_proj
tokenizer = model.tokenizer
del model
text_encoder.eval()
text_proj.eval()
text_encoder.to(device)
text_proj.to(device)

dataset = pd.read_csv(df_path)
file_paths = dataset['file_path'].tolist()
classes = dataset['class'].tolist()
embeddings = []
captions = []

for file_path in tqdm(file_paths):
    img = Image.open(file_path).convert('RGB')
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = caption_model.generate(**inputs)[0]
        caption = processor.decode(out, skip_special_tokens=True)
        text = tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(device)
        text_output = text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text').last_hidden_state[:,0,:]
        text_embed = text_proj(text_output).squeeze(0)
        captions.append(caption)
        embeddings.append(text_embed.cpu().numpy())

df = pd.DataFrame({
    'file_path': file_paths,
    'label': classes,
    'caption': captions,
    **{f'dim_{i}': val for emb in embeddings for i, val in enumerate(emb)}
})

df.to_csv(out_path, index=False)
print(f"Total images processed: {len(file_paths)}")
print(f"Embeddings saved to {out_path}")