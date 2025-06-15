import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter

state = random.randint(0, 1000)
print(f"Random state for reproducibility: {state}")
file_name = "kinetic_text_embeddings_1.csv"
df = pd.read_csv(file_name)
labels = df['label'].tolist()
embeddings = df.drop(columns=['file_path', 'label', 'caption']).values
label_to_idx = {label: idx for idx, label in enumerate(set(labels))}
labels_idx = [label_to_idx[label] for label in labels]
print(f"Embeddings shape: {embeddings.shape}, Labels shape: {len(labels)}")
scaler = MinMaxScaler()
embeddings = scaler.fit_transform(embeddings)
pca = PCA(n_components=50, random_state=state)
embeddings_2d = pca.fit_transform(embeddings)
tsne = TSNE(n_components=2, random_state=state)
embeddings_2d = tsne.fit_transform(embeddings)
print(f"t-SNE completed. Shape of 2D embeddings: {embeddings_2d.shape}")


writer = SummaryWriter(log_dir='runs')
writer.add_embedding(embeddings_2d, metadata=labels_idx, tag='tSNE_Embeddings')
folder = 'graphs'
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_idx, cmap='jet', alpha=0.5)
plt.title('t-SNE Visualization of Video Frame Embeddings')
plt.savefig(folder + "/" + file_name.replace('.csv', f'_tsne_{state}.png'))
#plt.show()