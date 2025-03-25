import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # 2N
    z = F.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T)

    # 去掉自身相似度
    mask = torch.eye(2*N, dtype=torch.bool).to(z.device)
    similarity_matrix = similarity_matrix[~mask].view(2*N, -1)

    positives = torch.sum(z1 * z2, dim=-1)
    positives = torch.cat([positives, positives], dim=0)

    logits = torch.cat([positives.unsqueeze(1), similarity_matrix], dim=1)
    labels = torch.zeros(2*N, dtype=torch.long).to(z.device)

    return F.cross_entropy(logits / temperature, labels)
