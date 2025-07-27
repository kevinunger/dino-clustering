import os
import argparse
import logging
import time
from datetime import datetime
from collections import deque
from functools import partial
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, DistributedSampler
from torch.nn.utils import parametrizations
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

import torchvision.transforms as transforms

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

import timm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import umap.umap_ as umap
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA

import gurobipy as gp
from optuna import create_study

import tqdm


# clear cuda / gpu memory
def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

clear_gpu_memory()

'''
python dinov1.py --random_search --n_trials 100 >/dev/null 2>&1 & disown
'''

# Logging (nicht für Tensorboard sonder für stdout allg.)
os.makedirs("out", exist_ok=True)
log_filename = datetime.now().strftime("out/%H-%M-%S_%d-%m.out")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import builtins
def print(*args, **kwargs):
    logging.info(' '.join(str(arg) for arg in args))
    builtins.print(*args, **kwargs)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class Organoid(Dataset):
    def __init__(
        self,
        root,
        n_local_crops=8,
        apply_transforms=True,
        augmentations=None,
    ):
        """
        Args:
            root (str): Pfad zum Ordner, der die Organoid-Bilder enthält.
            n_local_crops (int): Anzahl an 'lokalen' Crops, die erstellt werden sollen.
            apply_transforms (bool): Flag, ob überhaupt Augmentations angewendet werden sollen.
            augmentations (dict or None): Falls ein dict übergeben wird, kann es eigene
                                          Transformation-Pipelines für 'global_1', 'global_2' 
                                          und 'local' enthalten. Ansonsten werden Defaults verwendet.
        """
        self.root = root
        self.imgs = sorted([img for img in os.listdir(root) if "mask" not in img])
        self.apply_transforms = apply_transforms

        # Force size to 224x224 for Vision Transformer
        self.size = 224
        self.n_local_crops = n_local_crops

        # Standard-Skalierungsfaktoren gemäß BYOL-/DINO-Ähnlichen Setups
        global_crops_scale = (0.5, 0.9)
        local_crops_scale = (0.05, 0.4)

        # Gemeinsamer Normalisierungs-Transform
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Beispiel für leichte Augmentation: horizontales Flip & Gaussian Blur
        flip_and_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # Keine Farb- oder Grayscale-Änderungen,
            # da Farben vom Modell gelernt werden sollen
        ])

        # Häufig in DINO verwendete Operation: selektives Anwenden von GaussianBlur
        RandomGaussianBlur = lambda p: transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2))], p=p
        )

        # -- Default-Augmentationen --
        # "Global View" 1
        default_global_1 = transforms.Compose([
            transforms.Resize((self.size, self.size)),  # Ensure base size
            transforms.RandomResizedCrop(
                self.size, scale=global_crops_scale, interpolation=Image.BICUBIC
            ),
            flip_and_jitter,
            normalize
        ])

        # "Global View" 2 (mit leicht unterschiedlicher Wahrscheinlichkeit für Blur)
        default_global_2 = transforms.Compose([
            transforms.Resize((self.size, self.size)),  # Ensure base size
            transforms.RandomResizedCrop(
                self.size, scale=global_crops_scale, interpolation=Image.BICUBIC
            ),
            RandomGaussianBlur(0.1),
            normalize
        ])

        # "Local View" (kleinere Crops, häufiger Blur)
        default_local = transforms.Compose([
            transforms.Resize((self.size, self.size)),  # Ensure base size
            transforms.RandomResizedCrop(
                self.size, scale=local_crops_scale, interpolation=Image.BICUBIC
            ),
            flip_and_jitter,
            RandomGaussianBlur(0.5),
            normalize
        ])

        # Wenn von außen kein eigenes Augmentations-Dict übergeben wird,
        # verwende die Default-Augmentations
        self.augmentations = augmentations or {
            "global_1": default_global_1,
            "global_2": default_global_2,
            "local": default_local,
            "n_local_crops": self.n_local_crops,
        }

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Bild öffnen
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # Label extrahieren (z. B. aus Dateinamen '_XYZ.jpg' => XYZ-100 => Label)
        img_name = os.path.basename(img_path)
        num_part = int(img_name.split('_')[-1].split('.')[0])
        label = (num_part // 100) - 1

        if self.apply_transforms:
            # Wende definierte Augmentations an (entweder Default oder von außen übergeben)
            all_crops = [
                self.augmentations["global_1"](img),
                self.augmentations["global_2"](img),
            ]
            all_crops.extend([
                self.augmentations["local"](img)
                for _ in range(self.augmentations["n_local_crops"])
            ])

            # Originalbild als Referenz (ohne Augmentation, nur Resize + ToTensor)
            orig_img_tensor = transforms.ToTensor()(img)
            orig_img_tensor = transforms.Resize((self.size, self.size))(orig_img_tensor)

            return all_crops, orig_img_tensor, label

        else:
            # Falls man keine Transforms anwenden will (reines Laden + Normalisieren/Resize)
            orig_img_tensor = transforms.ToTensor()(img)
            orig_img_tensor = transforms.Resize((self.size, self.size))(orig_img_tensor)
            return orig_img_tensor, label
        



class DinoLoss(nn.Module):
    """The original DINO loss function.

    We subclass the `nn.Module` becuase we want to create a buffer for the
    logits center of the teacher.

    Parameters
    ----------
    out_dim : int
        The dimensionality of the final layer (we computed the softmax over).

    teacher_temp, student_temp : float
        Softmax temperature of the teacher resp. student.

    center_momentum : float
        Hyperparameter for the exponential moving average that determines
        the center logits. The higher the more the running average matters.
    """
    def __init__(
        self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.out_dim = out_dim  # Store out_dim
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """Evaluate loss.

        Parameters
        ----------
        student_output, teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` representing
            logits. The length is equal to number of crops.
            Note that student processed all crops and that the two initial crops
            are the global ones.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the average loss.
        """
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]

        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        total_loss = 0
        n_loss_terms = 0

        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue

                loss = torch.sum(-t * s, dim=-1)  # (n_samples,)
                total_loss += loss.mean()  # scalar
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.

        Compute the exponential moving average.

        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        # Ensure center has correct dimension
        if self.center.shape[1] != self.out_dim:
            self.center = torch.zeros(1, self.out_dim, device=self.center.device)
            
        batch_center = torch.cat(teacher_output).mean(
            dim=0, keepdim=True
        )  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

class CorrelationLoss(nn.Module):
    """Combined DINO and correlation clustering loss function.

    This loss combines the original DINO loss with a correlation clustering objective
    that encourages embeddings to be more clusterable.

    Parameters
    ----------
    out_dim : int
        The dimensionality of the final layer (we computed the softmax over).

    teacher_temp, student_temp : float
        Softmax temperature of the teacher resp. student.

    center_momentum : float
        Hyperparameter for the exponential moving average that determines
        the center logits. The higher the more the running average matters.

    margin : float
        Threshold for cosine similarity in correlation clustering loss.
        Higher values require embeddings to be more similar to be considered
        in the same cluster.
        
    correlation_weight : float
        Weight for the correlation clustering loss component.
    """
    def __init__(
        self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9, margin=0.7, correlation_weight=5.0
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.margin = margin
        self.correlation_weight = correlation_weight
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """Evaluate combined loss."""
        # 1. Original DINO Loss Components
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]
        
        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        dino_loss = 0
        n_dino_terms = 0

        # 2. DINO Loss Computation
        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue
                loss = torch.sum(-t * s, dim=-1)  # Cross-entropy between distributions
                dino_loss += loss.mean()
                n_dino_terms += 1

        dino_loss = dino_loss / n_dino_terms

        # 3. Correlation Clustering Loss
        correlation_loss = 0
        n_correlation_terms = 0

        # Normalize embeddings for cosine similarity
        student_norm = [F.normalize(s, dim=-1) for s in student_output]
        
        # Move normalized embeddings to CPU and convert to numpy for cosine similarity
        student_norm_cpu = [s.detach().cpu().numpy() for s in student_norm]
        
        # Concatenate all embeddings for similarity matrix
        all_embeddings = np.concatenate(student_norm_cpu, axis=0)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(all_embeddings)
        
        # Process each pair of embeddings
        n_embeddings = len(student_norm_cpu)
        for i in range(n_embeddings):
            for j in range(i+1, n_embeddings):
                # Get similarity between embeddings i and j
                sim = similarity_matrix[i, j]
                
                # Convert similarity to tensor for loss computation
                sim_tensor = torch.tensor(sim, device=student_output[0].device)
                
                # For positive pairs (same crop), encourage high similarity
                # For negative pairs (different crops), encourage low similarity
                if i == j:
                    # Positive pair loss: encourage similarity > margin
                    correlation_loss += torch.mean(torch.relu(self.margin - sim_tensor))
                else:
                    # Negative pair loss: encourage similarity < (1-margin)
                    correlation_loss += torch.mean(torch.relu(sim_tensor - (1-self.margin)))
                
                n_correlation_terms += 1

        correlation_loss = correlation_loss / n_correlation_terms

        # 4. Combine losses with weighting
        total_loss = dino_loss + self.correlation_weight * correlation_loss
        # print("CORRELATION LOSS: ", correlation_loss.item())
        # print("DINO LOSS: ", dino_loss.item())
        # print("WEIGHTED CORRELATION LOSS: ", (self.correlation_weight * correlation_loss).item())

        # Update center
        self.update_center(student_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.

        Compute the exponential moving average.

        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        batch_center = torch.cat(teacher_output).mean(
            dim=0, keepdim=True
        )  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


def clip_gradients(model, clip=2.0):
    """Rescale norm of computed gradients.

    Parameters
    ----------
    model : nn.Module
        Module.

    clip : float
        Maximum norm.
    """
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        if norm_last_layer:
            self.last_layer = parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight.data.fill_(1)
        else:
            self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def _init_weights(self, m):
        """Initialize learnable parameters."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape `(n_samples, in_dim)`.

        Returns
        -------
        torch.Tensor
            Of shape `(n_samples, out_dim)`.
        """
        x = self.mlp(x)  # (n_samples, bottleneck_dim)
        x = nn.functional.normalize(x, dim=-1, p=2)  # (n_samples, bottleneck_dim)
        x = self.last_layer(x)  # (n_samples, out_dim)

        return x
    
class MultiCropWrapper(nn.Module):
    """Convenience class for forward pass of multiple crops.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Instantiated Vision Transformer. Note that we will take the `head`
        attribute and replace it with `nn.Identity`.

    new_head : Head
        New head that is going to be put on top of the `backbone`.
    """
    def __init__(self, backbone, new_head):
        super().__init__()
        backbone.head = nn.Identity()  # deactivate original head
        self.backbone = backbone
        self.new_head = new_head

    def forward(self, x):
        """Run the forward pass.

        The different crops are concatenated along the batch dimension
        and then a single forward pass is fun. The resulting tensor
        is then chunked back to per crop tensors.

        Parameters
        ----------
        x : list
            List of `torch.Tensor` each of shape `(n_samples, 3, size, size)`.

        Returns
        -------
        tuple
            Tuple of `torch.Tensor` each of shape `(n_samples, out_dim)` where
            `output_dim` is determined by `Head`.
        """
        n_crops = len(x)
        concatenated = torch.cat(x, dim=0)  # (n_samples * n_crops, 3, size, size)
        cls_embedding = self.backbone(concatenated)  # (n_samples * n_crops, in_dim)
        logits = self.new_head(cls_embedding)  # (n_samples * n_crops, out_dim)
        chunks = logits.chunk(n_crops)  # n_crops * (n_samples, out_dim)

        return chunks


# EVAL
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def compute_knn(backbone, data_loader):
    """Get CLS embeddings and use KNN classifier on them.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer whose head is just an identity mapping.

    data_loader : torch.utils.data.DataLoader
        DataLoader that does not apply any augmentations.

    Returns
    -------
    accuracy : float
        KNN classification accuracy.
    """
    # Get embeddings and labels
    embs, _, labels = compute_embedding(backbone, data_loader)
    
    if embs is None or len(embs) == 0:
        print("Warning: No embeddings computed for KNN")
        return 0.0
    
    # Split data into train and validation sets
    n_samples = len(embs)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    
    X_train = embs[indices[:n_train]]
    y_train = labels[indices[:n_train]]
    X_val = embs[indices[n_train:]]
    y_val = labels[indices[n_train:]]
    
    # Train KNN classifier
    estimator = KNeighborsClassifier()
    estimator.fit(X_train, y_train)
    y_val_pred = estimator.predict(X_val)

    # Compute accuracy
    acc = accuracy_score(y_val, y_val_pred)
    return acc

def compute_embedding(backbone, data_loader):
    device = next(backbone.parameters()).device

    embs_l = []
    imgs_l = []
    labels = []

    for batch_idx, (img, label) in enumerate(data_loader):
        try:
            # 🛠️ Fix: Falls img eine Liste ist, in Tensor stapeln
            if isinstance(img, list):
                img = torch.stack(img, dim=0)

            img = img.to(device).float()

            with torch.no_grad():
                embs = backbone(img)
                embs = embs.detach().cpu()

            embs_l.append(embs.numpy())
            imgs_l.append(img.cpu().numpy())
            labels.extend(label.numpy())

        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not embs_l:
        print("Warning: No embeddings were computed")
        return None, None, None

    embs = np.concatenate(embs_l, axis=0)
    imgs = np.concatenate(imgs_l, axis=0)
    labels = np.array(labels)

    return embs, imgs, labels


def evaluate_cosine_similarity(embeddings, pos_pairs, neg_pairs):
    pos_sim = [F.cosine_similarity(embeddings[i], embeddings[j], dim=0).item() for i, j in pos_pairs]
    neg_sim = [F.cosine_similarity(embeddings[i], embeddings[j], dim=0).item() for i, j in neg_pairs]
    avg_pos_sim = np.mean(pos_sim)
    avg_neg_sim = np.mean(neg_sim)
    return avg_pos_sim, avg_neg_sim


def simple_kmeans(embeddings, labels, n_clusters=10) -> dict:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans.labels_


def compute_cc_metrics(embeddings, labels) -> dict:
    """Helper function to compute correlation clustering metrics."""
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create correlation clustering problem
    n_samples = len(embeddings)
    model = gp.Model("correlation_clustering")
    
    # Create binary variables for each pair (only upper triangular)
    x = model.addMVar((n_samples, n_samples), vtype=gp.GRB.BINARY, name="x")
    
    # Add reflexivity constraints (diagonal)
    for i in range(n_samples):
        model.addConstr(x[i,i] == 1)
    
    # Add symmetry constraints (only upper triangular)
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            model.addConstr(x[i,j] == x[j,i])
    
    # Add transitivity constraints (only necessary combinations)
    for i in range(n_samples-2):
        for j in range(i+1, n_samples-1):
            for k in range(j+1, n_samples):
                # If i and j are in same cluster and j and k are in same cluster,
                # then i and k must be in same cluster
                model.addConstr(x[i,k] >= x[i,j] + x[j,k] - 1)
    
    # Set objective: maximize agreement with similarity matrix
    # Positive terms: high similarity within clusters
    # Negative terms: low similarity between clusters
    obj = gp.quicksum(
        similarity_matrix[i,j] * x[i,j]  # Positive term for high similarity within clusters
        for i in range(n_samples)
        for j in range(i+1, n_samples)
    ) - gp.quicksum(
        similarity_matrix[i,j] * (1 - x[i,j])  # Negative term for low similarity between clusters
        for i in range(n_samples)
        for j in range(i+1, n_samples)
    )
    
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    try:
        # Optimize with early stopping
        model.optimize()
        
        if model.Status == gp.GRB.OPTIMAL or model.Status == gp.GRB.TIME_LIMIT:
            # Get clustering from solution
            clustering = np.zeros(n_samples, dtype=int)
            current_cluster = 0
            
            # Extract clusters using a more robust method
            for i in range(n_samples):
                if clustering[i] == 0:
                    current_cluster += 1
                    clustering[i] = current_cluster
                    # Find all samples that should be in the same cluster as i
                    for j in range(i+1, n_samples):
                        if x[i,j].X > 0.5:  # If i and j are in same cluster
                            clustering[j] = current_cluster
                            # Also check if j has any other connections
                            for k in range(j+1, n_samples):
                                if x[j,k].X > 0.5 and clustering[k] == 0:
                                    clustering[k] = current_cluster
            
            # Compute metrics
            nmi = normalized_mutual_info_score(labels, clustering)
            ari = adjusted_rand_score(labels, clustering)
            
            # Compute similarity gap efficiently
            cluster_pairs = []
            non_cluster_pairs = []
            
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    if clustering[i] == clustering[j]:
                        cluster_pairs.append(similarity_matrix[i,j])
                    else:
                        non_cluster_pairs.append(similarity_matrix[i,j])
            
            cluster_pairs = np.array(cluster_pairs)
            non_cluster_pairs = np.array(non_cluster_pairs)
            
            if len(cluster_pairs) > 0 and len(non_cluster_pairs) > 0:
                similarity_gap = np.mean(cluster_pairs) - np.mean(non_cluster_pairs)
            else:
                similarity_gap = 0.0
            
            return {
                'nmi': float(nmi),
                'ari': float(ari),
                'similarity_gap': float(similarity_gap),
                'clustering': clustering
            }
        else:
            print(f"Optimization failed with status {model.Status}")
            return {
                'nmi': 0.0,
                'ari': 0.0,
                'similarity_gap': 0.0,
                'clustering': np.zeros(n_samples, dtype=int)
            }
            
    except Exception as e:
        print(f"Error in optimization: {str(e)}")
        return {
            'nmi': 0.0,
            'ari': 0.0,
            'similarity_gap': 0.0,
            'clustering': np.zeros(n_samples, dtype=int)
        }

def evaluate(model, data_loader):
    """Evaluate the model and compute all metrics."""
    model.eval()
    
    # Get embeddings and labels
    embs, imgs, labels = compute_embedding(model, data_loader)

    reducer_3d = umap.UMAP(
    n_components=3,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    )

    reducer_2d = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    )

    embs_3d = reducer_3d.fit_transform(embs)
    embs_2d = reducer_2d.fit_transform(embs)

    # === PCA-Reduktion für HDBSCAN-Metrik ===
    pca_2d = PCA(n_components=2)
    embs_pca_2d = pca_2d.fit_transform(embs)
    pca_3d = PCA(n_components=3)
    embs_pca_3d = pca_3d.fit_transform(embs)

    # Ensure labels is a 1D array
    labels = np.asarray(labels).flatten()
    
    # First compute metrics on original embeddings
    # metrics = compute_cc_metrics(embeddings_orig, labels)

    kmeans_labels = simple_kmeans(embs, labels)
    kmeans_labels_2d = simple_kmeans(embs_2d, labels)
    kmeans_labels_3d = simple_kmeans(embs_3d, labels)

    clusterer = HDBSCAN(min_cluster_size=5, metric='euclidean')
    labels_orig = clusterer.fit_predict(embs)
    labels_2d = clusterer.fit_predict(embs_2d)
    labels_3d = clusterer.fit_predict(embs_3d)
    # PCA -> HDBSCAN
    labels_pca_2d = clusterer.fit_predict(embs_pca_2d)
    labels_pca_3d = clusterer.fit_predict(embs_pca_3d)


    #### ZUM testen kurz deaktivieren
    metrics = {
        'nmi': 0.0,
        'ari': 0.0,
        'similarity_gap': 0.0,
        'clustering': np.zeros(len(labels), dtype=int),
        'simple_kmeans': simple_kmeans(embs, labels)
    }

    metrics['kmeans_rand'] = evaluate_clustering(labels, kmeans_labels)['rand_index']
    metrics['kmeans_rand_2d'] = evaluate_clustering(labels, kmeans_labels_2d)['rand_index']
    metrics['kmeans_rand_3d'] = evaluate_clustering(labels, kmeans_labels_3d)['rand_index']

    metrics['hdbscan_rand_orig'] = evaluate_clustering(labels, labels_orig)['rand_index']
    metrics['hdbscan_rand_2d'] = evaluate_clustering(labels, labels_2d)['rand_index']
    metrics['hdbscan_rand_3d'] = evaluate_clustering(labels, labels_3d)['rand_index']
    metrics['hdbscan_rand_pca_2d'] = evaluate_clustering(labels, labels_pca_2d)['rand_index']
    metrics['hdbscan_rand_pca_3d'] = evaluate_clustering(labels, labels_pca_3d)['rand_index']
    
    pos_pairs = []
    neg_pairs = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                pos_pairs.append((i, j))
            else:
                neg_pairs.append((i, j))
    
    embeddings_torch = torch.from_numpy(embs).float()
    cosine_sim_pos, cosine_sim_neg = evaluate_cosine_similarity(embeddings_torch, pos_pairs, neg_pairs)
    knn_accuracy = compute_knn(model, data_loader)

    return {
        'full_nmi': metrics['nmi'],
        'full_ari': metrics['ari'],
        'similarity_gap': metrics['similarity_gap'],
        'clustering': metrics['clustering'],
        'kmeans_rand': metrics['kmeans_rand'],
        'kmeans_rand_2d': metrics['kmeans_rand_2d'],
        'kmeans_rand_3d': metrics['kmeans_rand_3d'],
        'hdbscan_rand_orig': metrics['hdbscan_rand_orig'],
        'hdbscan_rand_2d': metrics['hdbscan_rand_2d'],
        'hdbscan_rand_3d': metrics['hdbscan_rand_3d'],
        'hdbscan_rand_pca_2d': metrics['hdbscan_rand_pca_2d'],
        'hdbscan_rand_pca_3d': metrics['hdbscan_rand_pca_3d'],
        'cosine_sim_pos': cosine_sim_pos,
        'cosine_sim_neg': cosine_sim_neg,
        'knn_accuracy': knn_accuracy,
        'embeddings': embs,
        'images': imgs,
        'labels': labels
    }


def evaluate_clustering(true_labels, pred_labels):
    print("=== Clustering Evaluation ===")
    print(f"Rand Index:               {rand_score(true_labels, pred_labels):.4f}")
    print(f"Adjusted Rand Index (ARI):{adjusted_rand_score(true_labels, pred_labels):.4f}")
    print(f"NMI (Normalized MI):      {normalized_mutual_info_score(true_labels, pred_labels):.4f}")

    return {
        'rand_index': rand_score(true_labels, pred_labels),
        'ari': adjusted_rand_score(true_labels, pred_labels),
        'nmi': normalized_mutual_info_score(true_labels, pred_labels)
    }



def load_checkpoint(checkpoint_path, device='cuda'):
    """Load a model checkpoint and reconstruct the model.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load the model to ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, trial_args, trial_number)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract information
    trial_args = checkpoint['trial_args']
    trial_number = checkpoint['trial_number']
    
    # Reconstruct model
    vit_name, dim = "vit_base_patch16_224", 768
    vit = timm.create_model(vit_name, pretrained=trial_args["pretrained"])
    
    # Initialize DINOHead
    dino_head = DINOHead(
        dim,
        trial_args["out_dim"],
        norm_last_layer=trial_args["norm_last_layer"],
    )
    
    # Create model
    model = MultiCropWrapper(vit, dino_head)
    
    # Handle DataParallel state dict
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix from keys
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load state dict
    model.load_state_dict(state_dict)
    
    # Move to device
    model = model.to(device)
    
    # If multiple GPUs available, use DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model, trial_args, trial_number


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    
    # Only keep random search parameters
    parser.add_argument('--random_search', action='store_true',
                       help='Whether to perform random search')
    parser.add_argument('--n_trials', type=int, default=None,
                       help='Number of trials for random search')
    
    return parser

def convert_to_tensorboard_compatible(value):
    """Convert a value to a TensorBoard compatible type."""
    if isinstance(value, (int, float, str, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return str(value)
    elif isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else str(value.tolist())
    elif isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else str(value.tolist())
    else:
        return str(value)

def save_augmented_images(data_loader, dir_to_save):
    """Save augmented images from the first batch of the data loader.
    
    Args:
        data_loader: DataLoader containing augmented images
        dir_to_save: Directory to save the images (can be string or Path)
    """
    # Convert to Path if string
    if isinstance(dir_to_save, str):
        dir_to_save = pathlib.Path(dir_to_save)
    
    # Create directory for images in the log directory
    img_dir = dir_to_save
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # ImageNet mean and std for denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Get first batch
    try:
        batch = next(iter(data_loader))
        all_crops = batch[0]
        
        # Save global crops
        if len(all_crops) >= 2:
            for i, crop in enumerate(all_crops[:2]):  # First two are global crops
                try:
                    crop = crop[0].cpu()  # Get first image from batch
                    # Denormalize
                    crop = crop * std + mean
                    crop = torch.clamp(crop, 0, 1)  # Ensure values are in [0,1]
                    crop = (crop * 255).byte()  # Convert to uint8
                    crop = crop.permute(1, 2, 0).numpy()  # Convert to HWC format
                    crop = Image.fromarray(crop)
                    crop.save(img_dir / f"global_crop_{i+1}.png")
                except Exception as e:
                    print(f"Error saving global crop {i+1}: {str(e)}")
        
        # Save local crops
        if len(all_crops) > 2:
            for i, crop in enumerate(all_crops[2:]):  # Rest are local crops
                try:
                    crop = crop[0].cpu()  # Get first image from batch
                    # Denormalize
                    crop = crop * std + mean
                    crop = torch.clamp(crop, 0, 1)  # Ensure values are in [0,1]
                    crop = (crop * 255).byte()  # Convert to uint8
                    crop = crop.permute(1, 2, 0).numpy()  # Convert to HWC format
                    crop = Image.fromarray(crop)
                    crop.save(img_dir / f"local_crop_{i+1}.png")
                except Exception as e:
                    print(f"Error saving local crop {i+1}: {str(e)}")
            
            # If we have additional elements in the batch, try to save them
            if len(batch) > 1:
                print("\nProcessing additional batch elements:")
                # Save original images
                if isinstance(batch[1], torch.Tensor) and len(batch[1].shape) == 4:  # Check if it's an image tensor
                    try:
                        img = batch[1][0].cpu()  # Get first image
                        # Denormalize
                        img = img * std + mean
                        img = torch.clamp(img, 0, 1)
                        img = (img * 255).byte()
                        img = img.permute(1, 2, 0).numpy()
                        img = Image.fromarray(img)
                        img.save(img_dir / "original_image.png")
                    except Exception as e:
                        print(f"Error saving original image: {str(e)}")
                
                # Save labels
                if len(batch) > 2 and isinstance(batch[2], torch.Tensor):
                    try:
                        labels = batch[2].cpu().numpy()
                        with open(img_dir / "labels.txt", "w") as f:
                            f.write("Labels for the batch:\n")
                            for i, label in enumerate(labels):
                                f.write(f"Image {i}: {label}\n")
                    except Exception as e:
                        print(f"Error saving labels: {str(e)}")
            
            print(f"\nSuccessfully saved all augmented images to {img_dir}")
        else:
            print("Unexpected batch format")
            return
            
    except Exception as e:
        print(f"Error saving augmented images: {str(e)}")
        import traceback
        traceback.print_exc()

def objective(trial, args, student, teacher, dino_loss, data_loader_train, optimizer, epoch,device, data_loader_val_plain, writer, trial_dir):
    """Objective function for random search optimization."""
    try:
        # Start timing the trial
        trial_start_time = time.time()
        
        print(f"\n=== Starting trial {trial.number} ===")
        
        # Randomly choose between DinoLoss and CorrelationLoss
        use_correlation_loss = trial.suggest_categorical('use_correlation_loss', [True, False])
        print(f"Using {'CorrelationLoss' if use_correlation_loss else 'DinoLoss'}")
        
        # Save augmented images for this trial
        save_augmented_images(data_loader_train, trial_dir)
        
        # Crop parameters
        global_crops_scale = trial.suggest_float('global_crops_scale', 0.5, 0.7)  # Adjusted range
        # local_crops_number = trial.suggest_int('local_crops_number', 6, 10)
        local_crops_number = 8
        local_crops_size = trial.suggest_int('local_crops_size', 80, 112)
        local_crops_scale = trial.suggest_float('local_crops_scale', 0.2, 0.4)  # Adjusted range
        
        # Temperature parameters
        teacher_temp = trial.suggest_float('teacher_temp', 0.07, 0.09)  # Adjusted range
        student_temp = trial.suggest_float('student_temp', 0.06, 0.08)  # Adjusted range
        
        # Training parameters
        start_lr = trial.suggest_float('start_lr', 1e-7, 1e-5, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.04, 0.06)  # Adjusted range
        momentum_teacher = trial.suggest_float('momentum_teacher', 0.99, 0.999)
        center_momentum = trial.suggest_float('center_momentum', 0.85, 0.95)  # Adjusted range
        #clip_grad = trial.suggest_float('clip_grad', 1.5, 2.5)
        clip_grad = 2.0 
        
        # Model parameters
        out_dim = trial.suggest_int('out_dim', 32768, 131072, step=32768)  # Adjusted range
        norm_last_layer = trial.suggest_categorical('norm_last_layer', [True])
        norm_in_head = trial.suggest_categorical('norm_in_head', [True])  # Changed to only True
        pretrained = trial.suggest_categorical('pretrained', [True])

        
        # Update args with trial parameters
        trial_args = args.copy()
        trial_args.update({
            "global_crops_scale": [global_crops_scale, 1.0],
            "local_crops_number": local_crops_number,
            "local_crops_size": local_crops_size,
            "local_crops_scale": [local_crops_scale, 0.5],
            "teacher_temp": teacher_temp,
            "student_temp": student_temp,
            "start_lr": start_lr,
            "weight_decay": weight_decay,
            "momentum_teacher": momentum_teacher,
            "center_momentum": center_momentum,
            "clip_grad": clip_grad,
            "out_dim": out_dim,
            "norm_last_layer": norm_last_layer,
            "norm_in_head": norm_in_head,
            "pretrained": pretrained,
            "use_correlation_loss": use_correlation_loss
        })
        
        # Reinitialize models
        vit_name, dim = "vit_base_patch16_224", 768
        student_vit = timm.create_model(vit_name, pretrained=pretrained)
        teacher_vit = timm.create_model(vit_name, pretrained=pretrained)
        
        dino_head = DINOHead(
            dim,
            trial_args["out_dim"],
            norm_last_layer=trial_args["norm_last_layer"],
        )
        
        trial_student = MultiCropWrapper(student_vit, dino_head)
        trial_teacher = MultiCropWrapper(teacher_vit, dino_head)
        
        for p in trial_teacher.parameters():
            p.requires_grad = False
            
        trial_student = trial_student.to(device)
        trial_teacher = trial_teacher.to(device)
        
        if isinstance(student, nn.DataParallel):
            trial_student = nn.DataParallel(trial_student)
            trial_teacher = nn.DataParallel(trial_teacher)
            
        # Reinitialize loss based on random choice
        if use_correlation_loss:
            trial_dino_loss = CorrelationLoss(
                out_dim=trial_args["out_dim"],
                teacher_temp=trial_args["teacher_temp"],
                student_temp=trial_args["student_temp"],
                margin=0.7,
                correlation_weight=5.0
            )
        else:
            trial_dino_loss = DinoLoss(
                out_dim=trial_args["out_dim"],
                teacher_temp=trial_args["teacher_temp"],
                student_temp=trial_args["student_temp"],
                center_momentum=trial_args["center_momentum"],
            )
        trial_dino_loss = trial_dino_loss.to(device)
        
        # Reinitialize optimizer
        lr = trial_args["start_lr"] * trial_args["batch_size"] / 256
        trial_optimizer = torch.optim.AdamW(
            trial_student.parameters(),
            lr=lr,
            weight_decay=trial_args["weight_decay"],
        )
        
        # Training loop
        total_loss = 0
        for epoch in range(trial_args["n_epochs"]):
            stats = train_one_epoch(
                student=trial_student,
                teacher=trial_teacher,
                dino_loss=trial_dino_loss,
                data_loader=data_loader_train,
                optimizer=trial_optimizer,
                epoch=epoch,
                args=trial_args,
                device=device,
                writer=writer,
            )
            
            writer.add_scalar('train/loss', stats['loss'], epoch)
            total_loss += stats['loss']

            print(f"Trial {trial.number} - Epoch {epoch}: Loss={stats['loss']:.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / trial_args["n_epochs"]
        
        # Evaluation
        trial_student.eval()
        model_for_embedding = trial_student.module.backbone if isinstance(trial_student, nn.DataParallel) else trial_student.backbone
        
        metrics = evaluate(model_for_embedding, data_loader_val_plain)
        
        # Log metrics
        writer.add_scalar('metrics/nmi', metrics['full_nmi'], 0)
        writer.add_scalar('metrics/ari', metrics['full_ari'], 0)
        writer.add_scalar('metrics/similarity_gap', metrics['similarity_gap'], 0)
        writer.add_scalar('metrics/final_loss', avg_loss, 0)
        writer.add_scalar('metrics/cosine_sim_pos', metrics['cosine_sim_pos'], 0)
        writer.add_scalar('metrics/cosine_sim_neg', metrics['cosine_sim_neg'], 0)
        writer.add_scalar('metrics/knn_accuracy', metrics['knn_accuracy'], 0)
        writer.add_scalar('metrics/hdbscan_rand_orig', metrics['hdbscan_rand_orig'], 0)
        writer.add_scalar('metrics/hdbscan_rand_2d', metrics['hdbscan_rand_2d'], 0)
        writer.add_scalar('metrics/hdbscan_rand_3d', metrics['hdbscan_rand_3d'], 0)
        writer.add_scalar('metrics/hdbscan_rand_pca_2d', metrics['hdbscan_rand_pca_2d'], 0)
        writer.add_scalar('metrics/hdbscan_rand_pca_3d', metrics['hdbscan_rand_pca_3d'], 0)
        writer.add_scalar('metrics/kmeans_rand', metrics['kmeans_rand'], 0)
        writer.add_scalar('metrics/kmeans_rand_2d', metrics['kmeans_rand_2d'], 0)
        writer.add_scalar('metrics/kmeans_rand_3d', metrics['kmeans_rand_3d'], 0)
        
        # Save embeddings
        projector_dir = trial_dir / "projector"
        projector_dir.mkdir(exist_ok=True)
        
        # Convert images to proper format for TensorBoard
        if metrics['images'] is not None:
            # Ensure images are in the correct format (N, C, H, W)
            images = torch.from_numpy(metrics['images'])
            if len(images.shape) == 3:
                images = images.unsqueeze(1)  # Add channel dimension if grayscale
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)  # Convert to RGB if grayscale
                
            # Save embeddings with proper image format
            writer.add_embedding(
                metrics['embeddings'],
                metadata=metrics['labels'],
                label_img=images,
                tag='embeddings/final'
            )
        else:
            # If no images available, save embeddings without images
            writer.add_embedding(
                metrics['embeddings'],
                metadata=metrics['labels'],
                tag='embeddings/final'
            )
        
        # Log hyperparameters
        hparams_dict = {k: convert_to_tensorboard_compatible(v) for k, v in trial_args.items()}
        metric_dict = {
            'hparam/loss': avg_loss,
            'hparam/cosine_sim_pos': metrics['cosine_sim_pos'],
            'hparam/cosine_sim_neg': metrics['cosine_sim_neg'],
            'hparam/knn_accuracy': metrics['knn_accuracy'],
            'hparam/hdbscan_rand_orig': metrics['hdbscan_rand_orig'],
            'hparam/hdbscan_rand_2d': metrics['hdbscan_rand_2d'],
            'hparam/hdbscan_rand_3d': metrics['hdbscan_rand_3d'],
            'hparam/hdbscan_rand_pca_2d': metrics['hdbscan_rand_pca_2d'],
            'hparam/hdbscan_rand_pca_3d': metrics['hdbscan_rand_pca_3d'],
            'hparam/kmeans_rand': metrics['kmeans_rand'],
            'hparam/kmeans_rand_2d': metrics['kmeans_rand_2d'],
            'hparam/kmeans_rand_3d': metrics['kmeans_rand_3d'],
        }
        writer.add_hparams(hparams_dict, metric_dict)
        
        # Flush writer
        writer.flush()
        
        # Calculate trial duration
        trial_duration = time.time() - trial_start_time
        hours = int(trial_duration // 3600)
        minutes = int((trial_duration % 3600) // 60)
        seconds = int(trial_duration % 60)
        
        print(f"\nTrial {trial.number} duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Log duration to tensorboard
        writer.add_scalar('time/trial_duration_seconds', trial_duration, 0)
        writer.add_text('time/trial_duration_formatted', f"{hours:02d}:{minutes:02d}:{seconds:02d}", 0)


        # Save model checkpoint
        checkpoint = {
            'epoch': trial_args["n_epochs"],
            'model_state_dict': trial_student.state_dict(),
            'trial_args': {
                'out_dim': trial_args["out_dim"],
                'norm_last_layer': trial_args["norm_last_layer"],
                'norm_in_head': trial_args["norm_in_head"],
                'pretrained': trial_args["pretrained"],
                'use_correlation_loss': trial_args["use_correlation_loss"]
            },
            'trial_number': trial.number
        }
        
        # Create checkpoints directory if it doesn't exist
        checkpoints_dir = trial_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = checkpoints_dir / f"trial_{trial.number}_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")
        
        print("\n=== Trial completed successfully ===")
        return metric_dict["hparam/kmeans_rand_3d"]
        
    except Exception as e:
        print(f"\n[ERROR] Error in trial: {str(e)}")
        print("[ERROR] Full error traceback:")
        import traceback
        traceback.print_exc()
        return 0.0

def train_one_epoch(student, teacher, dino_loss, data_loader, optimizer, epoch, args, device, writer):
    """Train one epoch of the model."""
    total_loss = 0
    n_batches = 0
    
    for data_iter_step, data in enumerate(data_loader):
        try:

            all_crops, original_img, label = data

            images = []
            for im in all_crops:
                im = im.to(device).float()
                images.append(im)
            
            # Teacher forward pass - only the 2 global views pass through the teacher
            teacher_output = teacher(images[:2])
            # Student forward pass
            student_output = student(images)
            # get loss
            student_loss = dino_loss(student_output, teacher_output)

            student_loss.backward()
            if args["clip_grad"] is not None:
                param_norms = clip_gradients(student, args["clip_grad"])
            optimizer.step()

            writer.add_scalar('train/loss', student_loss.item(), data_iter_step)

            m = 0.999
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # Logging
            total_loss += student_loss.item()
            n_batches += 1

            if data_iter_step % 10 == 0:
                print(f"Epoch [{epoch}] Batch [{data_iter_step}/{len(data_loader)}] Loss: {student_loss.item():.4f}")



        except Exception as e:
            print(f"Error in batch {data_iter_step}: {str(e)}")
            print(f"Debug: Full error traceback:")
            import traceback
            traceback.print_exc()
            continue
    
    if n_batches == 0:
        print("No valid batches in this epoch!")
        return {"loss": float('inf')}
        
    avg_loss = total_loss / n_batches
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")
    return {"loss": avg_loss}

def train(args) -> dict:
    # Create base logging directory with timestamp
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    suffix = "_random"
    log_dir = pathlib.Path("logs") / f"{base_timestamp}{suffix}"
    # Store timestamp for final training
    args["random_search_timestamp"] = base_timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a single writer for all trials
    writer = SummaryWriter(log_dir=str(log_dir))

    # hyperparams
    # Training Setup
    vit_name, dim = "vit_base_patch16_224", 768

    # Dataset paths
    val_path =  "./ds/organoid/test-100"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()


    print(f"Number of available GPUs: {num_gpus}")

    # Set environment based on GPU count
    is_hpc = num_gpus > 1
    if is_hpc:
        print("HPC environment detected (multiple GPUs), using full dataset")
        imgs_path = "./ds/organoid/neu/single_organoids-224"
        dataset_train_aug = Organoid(imgs_path, apply_transforms=True)
        dataset_val_plain = Organoid(val_path, apply_transforms=False)
        batch_size = 128
        num_workers = 8
        train_indices = range(len(dataset_train_aug))
        val_indices = range(len(dataset_val_plain))

    else:
        print("Local environment detected (single/no GPU), using subset of data")
        imgs_path = "./ds/organoid/train-100-224"
        dataset_train_aug = Organoid(imgs_path, apply_transforms=True)
        dataset_val_plain = Organoid(val_path, apply_transforms=False)

        batch_size = 3
        num_workers = 0
        train_indices = range(len(dataset_train_aug))
        val_indices = range(len(dataset_val_plain))

    print("Training on {} samples".format(train_indices))
    print("Validation on {} samples".format(val_indices))
    
    args.update({
        "batch_size": batch_size,
        "batch_size_eval": batch_size,
        "n_workers": num_workers
    })

    # DataLoader configuration
    dataloader_kwargs = {
        "pin_memory": True,
        "num_workers": num_workers,
    }
    
    # Only add persistent_workers if we have workers
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    data_loader_train_aug = DataLoader(
        dataset_train_aug,
        batch_size=args["batch_size"],
        drop_last=True,
        sampler=SubsetRandomSampler(train_indices),
        **dataloader_kwargs
    )

    data_loader_val_plain = DataLoader(
        dataset_val_plain,
        batch_size=args["batch_size_eval"],
        drop_last=False,
        sampler=SubsetRandomSampler(val_indices),
        **dataloader_kwargs
    )

    # Neural network related
    student_vit = timm.create_model(vit_name, pretrained=args["pretrained"])
    teacher_vit = timm.create_model(vit_name, pretrained=args["pretrained"])

    # Initialize DINOHead with consistent dimensions
    dino_head = DINOHead(
          dim,
          args["out_dim"],
          norm_last_layer=args["norm_last_layer"],
    )

    student = MultiCropWrapper(
        student_vit,
        dino_head,
    )
    teacher = MultiCropWrapper(
        teacher_vit,
        dino_head,
    )

    for p in teacher.parameters():
        p.requires_grad = False

    # Move models to device
    student = student.to(device)
    teacher = teacher.to(device)

    # hpc: multiple GPUs, use DataParallel
    if num_gpus > 1:
        student = nn.DataParallel(student)
        teacher = nn.DataParallel(teacher)

    # Loss
    dino_loss_correlation = CorrelationLoss(
        out_dim=args["out_dim"],
        teacher_temp=args["teacher_temp"],
        student_temp=args["student_temp"],
        margin=0.7,
        correlation_weight=5.0
    )

    dino_loss = DinoLoss(
        out_dim=args["out_dim"],
        teacher_temp=args["teacher_temp"],
        student_temp=args["student_temp"],
        center_momentum=args["center_momentum"],
    )

    dino_loss = dino_loss.to(device)

    # Initialize optimizer and schedulers
    lr = args["start_lr"] * args["batch_size"] / 256
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=args["weight_decay"],
    )

    # Initialize epoch counter
    epoch = 0

    # Perform random search if requested
    if args["random_search"]:
        print(f"\nStarting RANDOM SEARCH with {args['n_trials']} trials!")
        study = create_study(direction="maximize")
        
        # Create a closure that captures all necessary variables
        def objective_wrapper(trial):
            # Create a new writer for each trial in a separate subdirectory
            trial_dir = log_dir / f"trial_{trial.number}"
            trial_dir.mkdir(exist_ok=True)
            trial_writer = SummaryWriter(log_dir=str(trial_dir))
            
            try:
                result = objective(
                    trial=trial,
                    args=args,
                    student=student,
                    teacher=teacher,
                    dino_loss=dino_loss,
                    data_loader_train=data_loader_train_aug,
                    optimizer=optimizer,
                    epoch=epoch,
                    device=device,
                    data_loader_val_plain=data_loader_val_plain,
                    writer=trial_writer,
                    trial_dir=trial_dir
                )
                
                trial_writer.close()
                return result
                
            except Exception as e:
                print(f"Error in trial {trial.number}: {str(e)}")
                trial_writer.close()
                return 0.0
        
        study.optimize(objective_wrapper, n_trials=args["n_trials"])
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Store best parameters for final training
        args["best_params"] = best_params
        args["best_value"] = best_value
        return best_params, best_value
    
    else:
        # Save augmented images in the log directory
        save_augmented_images(data_loader_train_aug, log_dir)
        train_one_epoch(student, teacher, dino_loss, data_loader_train_aug, optimizer, epoch, args, device)


    return {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    parsed_args = parser.parse_args()
    
    # Default arguments - only keep what's needed
    args = {
        # Model parameters
        "out_dim": 65536,
        "norm_last_layer": True,
        "norm_in_head": True,
        "pretrained": True,
        
        # Temperature parameters
        "teacher_temp": 0.08,
        "student_temp": 0.07,
        
        # Training parameters
        "batch_size": 128,
        "batch_size_eval": 128,
        "n_epochs": 3,
        "weight_decay": 0.05,
        "momentum_teacher": 0.996,
        "center_momentum": 0.9,
        "clip_grad": 2.0,
        "start_lr": 1e-5,
        
        # Multi-crop parameters
        "global_crops_scale": [0.8, 1.0],
        "local_crops_number": 8,
        "local_crops_size": 96,
        "local_crops_scale": [0.2, 0.5],
        
        # Random search parameters
        "random_search": parsed_args.random_search,
        "n_trials": parsed_args.n_trials if parsed_args.n_trials else 3
    }

    if args["random_search"]:
        best_params, best_value = train(args)
        
        # Update args with best parameters and set epochs for final training
        args.update(best_params)
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"{key}: {value}")
        print(f"Best value: {best_value}")

        # train with best parameters
        args["n_epochs"] = args["n_epochs"] * 2
        train(args)
        
    else:
        start_time = time.time()
        print(f"Starting training at {start_time}")
        # Just do normal training
        train(args)
        end_time = time.time()
        print(f"Training completed at {end_time}")
        print(f"Total training time: {end_time - start_time:.2f} seconds")