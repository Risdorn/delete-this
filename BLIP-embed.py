from BLIP.models.blip_itm import blip_itm

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
out_path = f"kinetic_vision_embeddings_{sample_size}.csv"
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
model_url = "model_base_retrieval_coco.pth"
model = blip_itm(model_url, image_size=image_size)

vision_encoder = model.visual_encoder
vision_proj = model.vision_proj
del model
vision_encoder.eval()
vision_proj.eval()
vision_encoder.to(device)
vision_proj.to(device)

dataset = pd.read_csv(df_path)
file_paths = dataset['file_path'].tolist()
classes = dataset['class'].tolist()
embeddings = []

for file_path in tqdm(file_paths):
    img = Image.open(file_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embeds = vision_encoder(img)[:,0,:]
        image_embeds = vision_proj(image_embeds).squeeze(0)
        embeddings.append(image_embeds.cpu().numpy())

df = pd.DataFrame({
    'file_path': file_paths,
    'label': classes,
    **{f'dim_{i}': val for emb in embeddings for i, val in enumerate(emb)}
})

df.to_csv(out_path, index=False)
print(f"Total images processed: {len(file_paths)}")
print(f"Embeddings saved to {out_path}")