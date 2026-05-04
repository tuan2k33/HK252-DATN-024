import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import MNIST
import numpy as np
import copy
import random
import logging
import os
import time
import datetime

# ==========================================
# 1. R-MNIST DATASET
# ==========================================

class RotatedMNIST(Dataset):
    """Single rotation domain of MNIST."""
    def __init__(self, root, train, degree, transform=None, download=True):
        self.mnist      = MNIST(root=root, train=train, download=download, transform=None)
        self.degree     = degree
        self.transform  = transform
        self._to_tensor = transforms.ToTensor()
        self.targets    = self.mnist.targets.tolist()

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        img = transforms.functional.rotate(img, self.degree)
        if self.transform:
            img = self.transform(img)
        else:
            img = self._to_tensor(img)
        return img, label


class RMNISTSequential:
    """
    R-MNIST: 20 tasks, each a different rotation degree in [0, 180).
    Domain-IL: same 10 digit classes across all tasks.
    Fixed seed to match Co2L paper setup.
    """
    N_TASKS   = 20
    N_CLASSES = 10

    def __init__(self, root, download=True):
        self.root     = root
        self.download = download
        self.degrees = [i * 9 for i in range(self.N_TASKS)]
        logging.info(f"R-MNIST degrees: {[round(d,1) for d in self.degrees]}")

    def get_task_datasets(self, task_id, test_transform):
        degree   = self.degrees[task_id]
        train_ds = RotatedMNIST(self.root, train=True,  degree=degree, download=self.download)
        test_ds  = RotatedMNIST(self.root, train=False, degree=degree,
                                transform=test_transform, download=self.download)
        return train_ds, test_ds


# ==========================================
# 2. CONTRASTIVE TRANSFORM (no rotation aug)
# ==========================================

class ContrastiveTransformMNIST(nn.Module):
    """
    SimCLR-style augmentation for MNIST 28x28.
    Rotation intentionally excluded to avoid conflict with domain-split rotations.
    Input: 3-channel tensor (grayscale repeated).
    """
    def __init__(self):
        super().__init__()
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.7, 1.0), antialias=True),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
        ])
        self._to_tensor = transforms.ToTensor()

    def _to3ch(self, x):
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x

    def forward(self, x):
        x = self._to3ch(x)
        return self.aug(x), self.aug(x)


# ==========================================
# 3. CNN ENCODER (Co2L paper for R-MNIST)
# ==========================================

class CNNEncoder(nn.Module):
    """
    Architecture from Co2L Appendix A.2 for R-MNIST:
    - Conv 20 filters 5x5 -> ReLU -> MaxPool 2x2
    - Conv 50 filters 5x5 -> ReLU -> MaxPool 2x2
    - FC 500 -> ReLU
    - MLP projector: 500 -> BN -> ReLU -> 500 -> BN
    Input: 3x28x28
    """
    def __init__(self, output_dim=500):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, stride=1),  # -> 20x24x24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # -> 20x12x12
            nn.Conv2d(20, 50, kernel_size=5, stride=1), # -> 50x8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # -> 50x4x4 = 800
            nn.Flatten(),
            nn.Linear(800, 500),
            nn.ReLU(),
        )
        self.feature_dim = 500
        self.projector = nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        features    = self.backbone(x)
        projections = self.projector(features)
        return features, projections


# ==========================================
# 4. LOSS FUNCTIONS
# ==========================================

class Co2LLoss(nn.Module):
    def __init__(self, t_contrast=0.07, t_student=0.07, t_teacher=0.07):
        super().__init__()
        self.t_contrast = t_contrast
        self.t_student  = t_student
        self.t_teacher  = t_teacher

    def contrastive_loss(self, features):
        batch_size = features.shape[0] // 2
        features   = F.normalize(features, dim=1)
        sim        = torch.matmul(features, features.T) / self.t_contrast
        mask_diag  = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        sim_nd     = sim[~mask_diag].view(2 * batch_size, -1)

        labels    = torch.cat([torch.arange(batch_size)] * 2).to(features.device)
        lbl_mat   = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        lbl_nd    = lbl_mat[~mask_diag].view(2 * batch_size, -1)

        positives = sim_nd[lbl_nd.bool()].view(lbl_nd.shape[0], -1)
        negatives = sim_nd[~lbl_nd.bool()].view(sim_nd.shape[0], -1)
        logits    = torch.cat([positives, negatives], dim=1)
        target    = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)
        return F.cross_entropy(logits, target)

    def ird_loss(self, curr_proj, past_proj):
        curr_proj = F.normalize(curr_proj, dim=1)
        past_proj = F.normalize(past_proj, dim=1)
        sim_c = torch.matmul(curr_proj, curr_proj.T) / self.t_student
        sim_p = torch.matmul(past_proj, past_proj.T) / self.t_teacher
        mask  = torch.eye(sim_c.shape[0], dtype=torch.bool, device=sim_c.device)
        sim_c = sim_c[~mask].view(sim_c.shape[0], -1)
        sim_p = sim_p[~mask].view(sim_p.shape[0], -1)
        return F.kl_div(F.log_softmax(sim_c, dim=1), F.softmax(sim_p, dim=1), reduction='batchmean')


# ==========================================
# 5. REPLAY BUFFER (Domain-IL)
# ==========================================

class ReplayBuffer:
    """
    Domain-IL buffer: store samples per (task_id, label) since same digit
    class appears across multiple domains.
    """
    def __init__(self, buffer_size=200):
        self.buffer_size = buffer_size
        self.data        = {}   # (task_id, label) -> list[Tensor 3xHxW]
        self._to_tensor  = transforms.ToTensor()

    def add_task(self, rmnist_ds, task_id, degree, slots_per_task):
        """Store up to slots_per_task samples from this task, balanced per class."""
        by_label = {}
        for idx in range(len(rmnist_ds.mnist)):
            lbl = rmnist_ds.targets[idx]
            by_label.setdefault(lbl, []).append(idx)

        per_class = max(1, slots_per_task // len(by_label))
        for lbl, idxs in by_label.items():
            random.shuffle(idxs)
            imgs = []
            for i in idxs[:per_class]:
                img, _ = rmnist_ds.mnist[i]
                img    = transforms.functional.rotate(img, degree)
                t      = self._to_tensor(img)
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
                imgs.append(t)
            self.data[(task_id, lbl)] = imgs

    def sample(self, batch_size):
        all_imgs, all_lbls = [], []
        for (_, lbl), imgs in self.data.items():
            for img in imgs:
                all_imgs.append(img)
                all_lbls.append(lbl)
        if not all_imgs:
            return None, None
        chosen = np.random.choice(len(all_imgs), min(batch_size, len(all_imgs)), replace=False)
        return torch.stack([all_imgs[i] for i in chosen]), \
               torch.tensor([all_lbls[i] for i in chosen])

    def lump(self, raw_imgs, alpha, device):
        if len(self) == 0:
            return None
        buf_imgs, _ = self.sample(raw_imgs.shape[0])
        if buf_imgs is None:
            return None
        lam = float(np.random.beta(alpha, alpha))
        n = buf_imgs.shape[0]
        return lam * raw_imgs[:n].to(device) + (1.0 - lam) * buf_imgs.to(device)

    def __len__(self):
        return sum(len(v) for v in self.data.values())


# ==========================================
# 6. AUG WRAPPER DATASET
# ==========================================

class AugWrapper(Dataset):
    """Returns ((v1, v2), raw_tensor, label) for training loop."""
    def __init__(self, rmnist_ds, aug_transform):
        self.ds         = rmnist_ds
        self.aug        = aug_transform
        self._to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, lbl = self.ds.mnist[idx]
        img      = transforms.functional.rotate(img, self.ds.degree)
        raw      = self._to_tensor(img)
        if raw.shape[0] == 1:
            raw = raw.repeat(3, 1, 1)
        v1, v2   = self.aug(raw)
        return (v1, v2), raw, lbl


# ==========================================
# 7. EVALUATION — Domain-IL linear probe
# ==========================================

def eval_domain_il(model, task_datasets, device, task_id, epochs_eval=100, buffer=None, current_train_ds=None):
    """
    Train a linear classifier on frozen features.
    Co2L eval protocol: use buffer samples (past) + current task samples.
    - SGD lr=1.0, momentum=0.9
    - MultiStep decay at 60, 75, 90
    - 100 epochs
    """
    model.eval()

    # Extract features from current task
    X_all, y_all = [], []
    if current_train_ds is not None:
        loader = DataLoader(current_train_ds, batch_size=256, num_workers=4)
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                feats, _ = model(imgs)
                X_all.append(feats.cpu())
                y_all.append(labels)

    # Extract features from buffer (past tasks)
    if buffer is not None and len(buffer) > 0:
        buf_imgs, buf_lbls = buffer.sample(len(buffer))
        if buf_imgs is not None:
            with torch.no_grad():
                feats, _ = model(buf_imgs.to(device))
            X_all.append(feats.cpu())
            y_all.append(buf_lbls)

    X_all = torch.cat(X_all)
    y_all = torch.cat(y_all)

    # Train linear classifier
    classifier = nn.Linear(X_all.shape[1], 10).to(device)
    opt        = optim.SGD(classifier.parameters(), lr=1.0, momentum=0.9)
    criterion  = nn.CrossEntropyLoss()

    classifier.train()
    for _ in range(epochs_eval):
        perm = torch.randperm(X_all.size(0))
        for i in range(0, X_all.size(0), 256):
            idx = perm[i:i+256]
            bx  = X_all[idx].to(device)
            by  = y_all[idx].to(device)
            opt.zero_grad()
            criterion(classifier(bx), by).backward()
            opt.step()

    # Evaluate per domain
    classifier.eval()
    domain_accs = []
    _, curr_test_ds = task_datasets[task_id]
    for t in range(task_id + 1):
        loader = DataLoader(curr_test_ds, batch_size=256, num_workers=4)
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(device)
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                labels = labels.to(device)
                feats, _ = model(imgs)
                preds    = classifier(feats).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        domain_accs.append(100.0 * correct / total if total > 0 else 0.0)
    return domain_accs


# ==========================================
# 8. MAIN LOOP
# ==========================================

def main(lmbd, t_contrast, t_student, t_teacher,
         epochs_train, epochs_eval,
         alpha_lump, buffer_size, data_root):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    rmnist  = RMNISTSequential(root=data_root, download=True)
    N_TASKS = rmnist.N_TASKS

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])
    aug_transform = ContrastiveTransformMNIST()

    # Pre-load all task datasets
    task_datasets = [rmnist.get_task_datasets(t, test_transform) for t in range(N_TASKS)]

    model     = CNNEncoder(output_dim=500).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-10)
    loss_fn   = Co2LLoss(t_contrast=t_contrast, t_student=t_student, t_teacher=t_teacher)
    buffer    = ReplayBuffer(buffer_size=buffer_size)
    prev_model = None
    acc_matrix = []
    avg_per_task = []

    # Fixed buffer slots per task
    buf_per_task = max(10, buffer_size // N_TASKS)

    for task_id in range(N_TASKS):
        degree = rmnist.degrees[task_id]
        epochs = epochs_train * 5 if task_id == 0 else epochs_train
        logging.info(f"=== Task {task_id+1}/{N_TASKS} | degree={degree:.1f} | epochs={epochs} ===")

        train_ds, _ = task_datasets[task_id]
        train_loader = DataLoader(
            AugWrapper(train_ds, aug_transform),
            batch_size=256, shuffle=True, drop_last=True, num_workers=4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20
        )

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (v1, v2), raw_imgs, _ in train_loader:
                v1, v2 = v1.to(device), v2.to(device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    mixed = buffer.lump(raw_imgs, alpha_lump, device)
                    if mixed is not None:
                        mv1, mv2 = aug_transform(mixed)
                        cat_v1   = torch.cat([v1, mv1], dim=0)
                        cat_v2   = torch.cat([v2, mv2], dim=0)
                    else:
                        cat_v1, cat_v2 = v1, v2

                    _, curr_proj = model(torch.cat([cat_v1, cat_v2], dim=0))
                    loss_con     = loss_fn.contrastive_loss(curr_proj)

                    loss_ird = torch.tensor(0.0, device=device)
                    if prev_model is not None and len(buffer) > 0:
                        buf_imgs, _ = buffer.sample(batch_size=60)
                        buf_imgs    = buf_imgs.to(device)
                        bv1, bv2    = aug_transform(buf_imgs)
                        buf_input   = torch.cat([bv1, bv2], dim=0)
                        _, z_curr   = model(buf_input)
                        with torch.no_grad():
                            _, z_prev = prev_model(buf_input)
                        loss_ird = loss_fn.ird_loss(z_curr, z_prev)

                    loss = loss_con + lmbd * loss_ird

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            if (epoch + 1) % 10 == 0:
                logging.info(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

        # Update buffer and snapshot model
        buffer.add_task(train_ds, task_id, degree, buf_per_task)
        prev_model = copy.deepcopy(model).eval()

        # Evaluate
        domain_accs = eval_domain_il(model, task_datasets, device, task_id, epochs_eval,
                                     buffer=buffer, current_train_ds=train_ds)
        acc_matrix.append(domain_accs)
        avg = np.mean(domain_accs)
        avg_per_task.append(float(avg))
        logging.info(f"After Task {task_id+1} | Domain accs: {[round(a,2) for a in avg_per_task]}")
        logging.info(f"After Task {task_id+1} | Current Domain-IL: {avg:.2f}%")

    # Final metrics
    domain_il_aa = float(np.mean(avg_per_task))

    bwt_vals = [acc_matrix[N_TASKS-1][t] - acc_matrix[t][t] for t in range(N_TASKS-1)]
    bwt      = float(np.mean(bwt_vals)) if bwt_vals else 0.0

    fm_vals = [max(acc_matrix[k][t] for k in range(t, N_TASKS-1)) - acc_matrix[N_TASKS-1][t]
               for t in range(N_TASKS-1)]
    fm      = float(np.mean(fm_vals)) if fm_vals else 0.0

    logging.info("=" * 50)
    logging.info(">>>> FINAL DOMAIN-IL METRICS <<<<")
    logging.info(f"Domain-IL Accuracy : {domain_il_aa:.2f}%")
    logging.info(f"Backward Transfer  : {bwt:.4f}")
    logging.info(f"Forgetting Measure : {fm:.4f}")
    logging.info("=" * 50)
    return domain_il_aa, bwt, fm


def calculate_stats(data_list):
    n = len(data_list)
    if n == 0:
        return 0.0, 0.0
    mean = sum(data_list) / n
    std  = (sum((x - mean)**2 for x in data_list) / (n-1))**0.5 if n > 1 else 0.0
    return mean, std


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R-MNIST Domain-IL Benchmark")
    parser.add_argument('--lmbd',          type=float, default=0.6,     help='IRD loss weight')
    parser.add_argument('--trials',        type=int,   default=5,       help='Number of independent runs')
    parser.add_argument('--t_contrast',    type=float, default=0.2,    help='Temperature contrastive loss')
    parser.add_argument('--t_student',     type=float, default=0.1,    help='Temperature student IRD')
    parser.add_argument('--t_teacher',     type=float, default=0.01,    help='Temperature teacher IRD')
    parser.add_argument('--epochs_train',  type=int,   default=20,      help='Epochs per task (task 1 auto x5)')
    parser.add_argument('--epochs_eval',   type=int,   default=100,     help='Epochs linear probe')
    parser.add_argument('--alpha_lump',    type=float, default=0.1,     help='Beta alpha LUMP mixup')
    parser.add_argument('--buffer_size',   type=int,   default=500,     help='Buffer size (Co2L: 200/500)')
    parser.add_argument('--data_root',     type=str,   default='./data',help='MNIST data root')
    parser.add_argument('--log_file',      type=str,   default=None,    help='Log file path')
    parser.add_argument('--gpu',           type=int,   default=1,       help='GPU id')  # lmbd=1.0
    args = parser.parse_args()

    log_dir  = './log_rmnist'
    os.makedirs(log_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(
        log_dir, f"rmnist_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file)]
    )

    logging.info("=" * 50)
    logging.info(">>>> R-MNIST DOMAIN-IL BENCHMARK <<<<")
    logging.info("=" * 50)
    for k, v in vars(args).items():
        logging.info(f"  {k:<15} = {v}")
    logging.info("=" * 50)

    all_acc, all_bwt, all_fm = [], [], []
    for i in range(args.trials):
        t0 = time.time()
        logging.info(f">>>> TRIAL {i+1}/{args.trials} <<<<")
        acc, bwt, fm = main(
            lmbd=args.lmbd,
            t_contrast=args.t_contrast,
            t_student=args.t_student,
            t_teacher=args.t_teacher,
            epochs_train=args.epochs_train,
            epochs_eval=args.epochs_eval,
            alpha_lump=args.alpha_lump,
            buffer_size=args.buffer_size,
            data_root=args.data_root,
        )
        all_acc.append(acc)
        all_bwt.append(bwt)
        all_fm.append(fm)
        logging.info(f"Trial {i+1} done in {time.time()-t0:.1f}s")
        torch.cuda.empty_cache()

    ma, sa = calculate_stats(all_acc)
    mb, sb = calculate_stats(all_bwt)
    mf, sf = calculate_stats(all_fm)

    logging.info("=" * 50)
    logging.info(f">>>> SUMMARY <<<<")
    logging.info("=" * 50)
    logging.info(f"Domain-IL Accuracy : {ma:.2f} ± {sa:.2f}")
    logging.info(f"Backward Transfer  : {mb:.4f} ± {sb:.4f}")
    logging.info(f"Forgetting Measure : {mf:.4f} ± {sf:.4f}")
    logging.info("=" * 50)