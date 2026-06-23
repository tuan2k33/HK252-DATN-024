import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
import numpy as np
import copy
import random
import logging
import os
import time
import datetime
import yaml

# ==========================================
# 1. MODEL & TRANSFORM
# ==========================================

class ContrastiveTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size        = cfg['size']
        mean        = cfg['mean']
        std         = cfg['std']
        blur_prob   = cfg['blur_prob']
        kernel_size = cfg['blur_kernel_size']
        scale_min   = cfg['crop_scale_min']

        aug_list = [
            transforms.RandomResizedCrop(size, scale=(scale_min, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=blur_prob),
            transforms.Normalize(mean, std),
        ]

        self.aug      = transforms.Compose(aug_list)
        self.to_tensor = transforms.ToTensor()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.to_tensor(x)
        return self.aug(x), self.aug(x)

class Encoder(nn.Module):
    def __init__(self, output_dim=128):
        super(Encoder, self).__init__()
        self.backbone         = models.resnet18(weights=None)
        self.backbone.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        features    = self.backbone(x)
        projections = self.projector(features)
        return features, projections

# ==========================================
# 2. LOSS FUNCTIONS
# ==========================================
class Co2LLoss(nn.Module):
    def __init__(self, t_contrast=0.5, t_student=0.2, t_teacher=0.01):
        super(Co2LLoss, self).__init__()
        self.t_contrast = t_contrast
        self.t_student  = t_student
        self.t_teacher  = t_teacher

    def contrastive_loss(self, features):
        batch_size = features.shape[0] // 2
        features   = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T) / self.t_contrast
        mask_diag = torch.eye(2 * batch_size, dtype=torch.bool).to(features.device)
        sim_no_diag = similarity_matrix[~mask_diag].view(2 * batch_size, -1)

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)
        labels_no_diag = labels[~mask_diag].view(2 * batch_size, -1)

        positives = sim_no_diag[labels_no_diag.bool()].view(labels_no_diag.shape[0], -1)
        negatives = sim_no_diag[~labels_no_diag.bool()].view(sim_no_diag.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        target = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        return F.cross_entropy(logits, target)

    def ird_loss(self, curr_projections, past_projections):
        curr_projections = F.normalize(curr_projections, dim=1)
        past_projections = F.normalize(past_projections, dim=1)

        sim_curr = torch.matmul(curr_projections, curr_projections.T) / self.t_student
        sim_past = torch.matmul(past_projections, past_projections.T) / self.t_teacher

        mask = torch.eye(sim_curr.shape[0], dtype=torch.bool).to(sim_curr.device)
        sim_curr = sim_curr[~mask].view(sim_curr.shape[0], -1)
        sim_past = sim_past[~mask].view(sim_past.shape[0], -1)

        log_prob_curr = F.log_softmax(sim_curr, dim=1)
        prob_past     = F.softmax(sim_past, dim=1)

        return F.kl_div(log_prob_curr, prob_past, reduction='batchmean')

# ==========================================
# 3. MEMORY BUFFER
# ==========================================
class ReplayBuffer:
    def __init__(self, buffer_size=500):
        self.buffer_size = buffer_size
        self.task_imgs   = {}   # task_id -> list[Tensor]
        self._to_tensor  = transforms.ToTensor()

    def add_task(self, raw_dataset, indices, task_id):
        n_tasks    = task_id + 1
        per_task   = self.buffer_size // n_tasks
        chosen_idx = np.random.choice(indices, min(per_task, len(indices)), replace=False)
        imgs = []
        for idx in chosen_idx:
            img, _ = raw_dataset[idx]
            if not isinstance(img, torch.Tensor):
                img = self._to_tensor(img)
            imgs.append(img)
        self.task_imgs[task_id] = imgs
        # Rebalance old tasks
        for tid in list(self.task_imgs):
            if tid != task_id:
                self.task_imgs[tid] = self.task_imgs[tid][:per_task]

    def sample(self, batch_size):
        all_imgs = [img for imgs in self.task_imgs.values() for img in imgs]
        if not all_imgs:
            return None, None
        chosen = np.random.choice(len(all_imgs), min(batch_size, len(all_imgs)), replace=False)
        return torch.stack([all_imgs[i] for i in chosen]), None

    def lump(self, raw_imgs, alpha, device):
        if len(self) == 0 or alpha <= 0:
            return None
        buf_imgs, _ = self.sample(batch_size=raw_imgs.shape[0])
        lam = float(np.random.beta(alpha, alpha))
        return lam * raw_imgs.to(device) + (1.0 - lam) * buf_imgs.to(device)

    def __len__(self):
        return sum(len(v) for v in self.task_imgs.values())

# ==========================================
# 4. UTILS & DATA
# ==========================================
class CustomSubset(Dataset):
    def __init__(self, ds, indices, transform=None):
        self.ds = ds
        self.indices = indices
        self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        img, lbl = self.ds[self.indices[idx]]
        if self.transform:
            raw = transforms.ToTensor()(img)
            return self.transform(img), raw, lbl
        return img, lbl

def eval_fwt(model, next_train_indices, next_task_classes, raw_dataset, test_dataset,
             test_transform, device, cls_per_task=2, epochs_eval=100):
    """Train a fresh linear probe on the next task's data (before training on it)
    to measure forward transfer. Returns task-IL accuracy on next task's test set."""
    model.eval()
    probe_loader = DataLoader(
        CustomSubset(raw_dataset, next_train_indices, transform=test_transform),
        batch_size=128, num_workers=4
    )
    X, y = [], []
    with torch.no_grad():
        for imgs, _, labels in probe_loader:
            feats, _ = model(imgs.to(device))
            X.append(feats.cpu())
            y.append(labels)
    X, y = torch.cat(X), torch.cat(y)

    min_label = int(y.min().item())
    y_local   = y - min_label

    classifier = nn.Linear(X.shape[1], cls_per_task).to(device)
    opt        = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
    criterion  = nn.CrossEntropyLoss()
    classifier.train()
    for _ in range(epochs_eval):
        perm = torch.randperm(X.size(0))
        for i in range(0, X.size(0), 256):
            idx  = perm[i:i+256]
            b_x  = X[idx].to(device)
            b_y  = y_local[idx].to(device)
            opt.zero_grad()
            criterion(classifier(b_x), b_y).backward()
            opt.step()

    t_test_idx  = np.where(np.isin(np.array(test_dataset.targets), next_task_classes))[0]
    test_loader = DataLoader(Subset(test_dataset, t_test_idx), batch_size=128)
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats  = model(imgs)[0]
            preds  = classifier(feats).argmax(dim=1) + min_label
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0

def eval_linear_probe(model, train_loader, test_loaders, task_id, device, cls_per_task=2, epochs_eval=100):
    model.eval()
    X_train, y_train = [], []
    num_classes_seen = (task_id + 1) * cls_per_task

    with torch.no_grad():
        for imgs, _, labels in train_loader:
            feats, _ = model(imgs.to(device))
            X_train.append(feats.cpu())
            y_train.append(labels)
    X_train, y_train = torch.cat(X_train), torch.cat(y_train)

    classifier = nn.Linear(X_train.shape[1], num_classes_seen).to(device)
    optimizer  = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
    criterion  = nn.CrossEntropyLoss()

    classifier.train()
    for _ in range(epochs_eval):
        perm = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), 256):
            idx  = perm[i:i+256]
            b_x  = X_train[idx].to(device)
            b_y  = y_train[idx].to(device)
            optimizer.zero_grad()
            criterion(classifier(b_x), b_y).backward()
            optimizer.step()

    classifier.eval()
    class_il_corr = [0.] * num_classes_seen
    class_il_cnt  = [0.] * num_classes_seen
    task_il_corr  = [0.] * (task_id + 1)
    task_il_cnt   = [0.] * (task_id + 1)

    with torch.no_grad():
        for t_idx, loader in enumerate(test_loaders):
            t_start = t_idx * cls_per_task
            t_end   = t_start + cls_per_task
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats, _ = model(imgs)
                logits   = classifier(feats)

                # class-IL: global argmax
                preds = logits.argmax(dim=1)
                for i in range(labels.shape[0]):
                    lbl = labels[i].item()
                    class_il_cnt[lbl] += 1
                    if preds[i] == lbl:
                        class_il_corr[lbl] += 1

                # task-IL: restricted argmax trong task slice
                task_preds = logits[:, t_start:t_end].argmax(dim=1) + t_start
                task_il_corr[t_idx] += (task_preds == labels).sum().item()
                task_il_cnt[t_idx]  += labels.shape[0]

    class_il_accs = [class_il_corr[i] / class_il_cnt[i] * 100 if class_il_cnt[i] > 0 else 0
                     for i in range(num_classes_seen)]
    task_il_accs  = [task_il_corr[t] / task_il_cnt[t] * 100 if task_il_cnt[t] > 0 else 0
                     for t in range(task_id + 1)]
    return class_il_accs, task_il_accs

def compute_cl_metrics(acc_matrix, pre_task_accs=None, cls_per_task=2):
    if len(acc_matrix) == 0:
        return 0.0, 0.0, 0.0, 0.0
    T = len(acc_matrix)

    # Class-IL Average Accuracy (AA)
    class_il_aa = float(np.mean(acc_matrix[-1]))

    # Backward Transfer (BWT)
    bwt = 0.0
    if T > 1:
        bwt_vals = []
        for i in range(T - 1):
            acc_after = np.mean(acc_matrix[T-1][i*cls_per_task:(i+1)*cls_per_task])
            acc_at_i  = np.mean(acc_matrix[i][i*cls_per_task:(i+1)*cls_per_task])
            bwt_vals.append(float(acc_after - acc_at_i))
        bwt = float(np.mean(bwt_vals))

    # Forward Transfer (FWT)
    fwt = 0.0
    if pre_task_accs:
        baseline = 100.0 / cls_per_task
        fwt = float(np.mean([a - baseline for a in pre_task_accs]))

    # Forgetting Measure (FM)
    fm = 0.0
    if T > 1:
        fm_vals = []
        for i in range(T - 1):
            max_acc = max(
                np.mean(acc_matrix[k][i*cls_per_task:(i+1)*cls_per_task])
                for k in range(i, T - 1)
            )
            acc_now = np.mean(acc_matrix[T-1][i*cls_per_task:(i+1)*cls_per_task])
            fm_vals.append(float(max_acc - acc_now))
        fm = float(np.mean(fm_vals))

    return class_il_aa, bwt, fwt, fm

def calculate_stats(data_list):
    n = len(data_list)
    if n == 0:
        return 0.0, 0.0
    mean = sum(data_list) / n
    std  = (sum((x - mean)**2 for x in data_list) / (n - 1))**0.5 if n > 1 else 0.0
    return mean, std

# ==========================================
# 5. MAIN LOOP
# ==========================================

def load_dataset_cfg(dataset_name, yml_path='data.yml'):
    with open(yml_path) as f:
        all_cfg = yaml.safe_load(f)
    if dataset_name not in all_cfg:
        raise ValueError(f"Dataset '{dataset_name}' not found in {yml_path}")
    return all_cfg[dataset_name]

def build_datasets(cfg, test_transform):
    name     = cfg['_name']
    root     = cfg['data_root']
    download = cfg.get('download', False)
    if name == 'cifar10':
        train_raw = CIFAR10(root=root, train=True,  download=download, transform=None)
        test_ds   = CIFAR10(root=root, train=False, download=download, transform=test_transform)
    elif name == 'tinyimagenet':
        from datasets import TinyImagenet
        train_raw = TinyImagenet(root=root, train=True,  download=download)
        test_ds   = TinyImagenet(root=root, train=False, download=download, transform=test_transform)
    elif name == 'eurosat':
        from datasets import EuroSAT
        train_raw = EuroSAT(root=root, train=True)
        test_ds   = EuroSAT(root=root, train=False, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train_raw, test_ds

def main(lmbd, t_contrast, t_student, t_teacher, epochs_train, epochs_eval, metrics=False,
         alpha_lump=1.0, dataset='cifar10', buffer_size=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    cfg = load_dataset_cfg(dataset)
    cfg['_name'] = dataset

    aug_transform  = ContrastiveTransform(cfg)
    test_transform = transforms.Compose([
        transforms.Resize(cfg['size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std']),
    ])

    train_dataset_raw, test_dataset = build_datasets(cfg, test_transform)

    targets      = np.array(train_dataset_raw.targets)
    cls_per_task = cfg['cls_per_task']
    n_classes    = cfg['n_classes']
    task_classes  = [list(range(i * cls_per_task, (i + 1) * cls_per_task)) for i in range(n_classes // cls_per_task)]
    train_indices = [np.where(np.isin(targets, cls))[0] for cls in task_classes]
    epochs        = epochs_train

    model     = Encoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-10)
    loss_fn   = Co2LLoss(t_contrast=t_contrast, t_student=t_student, t_teacher=t_teacher)

    buffer        = ReplayBuffer(buffer_size=buffer_size)
    prev_model    = None
    task_il_accs  = []
    acc_matrix    = []
    pre_task_accs = []
    task_il_acc   = 0.0

    for task_id, indices in enumerate(train_indices):
        logging.info(f"=== Task {task_id+1} / {len(task_classes)} ===")

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

        task_ds      = CustomSubset(train_dataset_raw, indices, transform=aug_transform)
        train_loader = DataLoader(task_ds, batch_size=256, shuffle=True, drop_last=True, num_workers=4)

        task_epochs = epochs * 5 if task_id == 0 else epochs
        model.train()
        for epoch in range(task_epochs):
            total_loss = 0

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

                    loss_ird = torch.tensor(0.).to(device)
                    if prev_model is not None and len(buffer) > 0:
                        all_buf, _ = buffer.sample(batch_size=128)
                        all_buf    = all_buf.to(device)
                        b_v1, b_v2 = aug_transform(all_buf)
                        buf_input  = torch.cat([b_v1, b_v2], dim=0)
                        _, z_curr  = model(buf_input)
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
                logging.info(f"Task {task_id+1}, Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

        buffer.add_task(train_dataset_raw, indices, task_id)

        prev_model = copy.deepcopy(model).eval()

        if metrics and task_id < len(train_indices) - 1:
            fwt_acc = eval_fwt(
                model,
                train_indices[task_id + 1], task_classes[task_id + 1],
                train_dataset_raw, test_dataset, test_transform, device,
                epochs_eval=epochs_eval
            )
            pre_task_accs.append(fwt_acc)
            logging.info(f"Task {task_id+1} -> FWT probe on task {task_id+2}: {fwt_acc:.4f}")

        seen_train_idx    = np.concatenate(train_indices[:task_id+1])
        eval_train_loader = DataLoader(
            CustomSubset(train_dataset_raw, seen_train_idx, transform=test_transform), batch_size=128
        )
        test_loaders = []
        for i in range(task_id + 1):
            t_idx = np.where(np.isin(np.array(test_dataset.targets), task_classes[i]))[0]
            test_loaders.append(DataLoader(Subset(test_dataset, t_idx), batch_size=128))

        class_il_accs, task_il_accs = eval_linear_probe(model, eval_train_loader, test_loaders, task_id, device,
                                                         cls_per_task=cls_per_task, epochs_eval=epochs_eval)
        acc_matrix.append(class_il_accs)
        logging.info(f"After Task {task_id+1} - Task accs : {[round(a, 4) for a in task_il_accs]}")
        logging.info(f"After Task {task_id+1} - Class accs: {[round(a, 4) for a in class_il_accs]}")

    task_il_acc = float(np.mean(task_il_accs)) if task_il_accs else 0.0

    if not metrics:
        class_il_aa = float(np.mean(acc_matrix[-1])) if acc_matrix else 0.0
        return class_il_aa, task_il_acc, 0.0, 0.0, 0.0

    class_il_aa, bwt, fwt, fm = compute_cl_metrics(acc_matrix, pre_task_accs, cls_per_task=cls_per_task)
    logging.info("="*50)
    logging.info(">>>> FINAL CL METRICS <<<<")
    logging.info("="*50)
    logging.info(f"Class-IL Accuracy : {class_il_aa:.4f}")
    logging.info(f"Task-IL Accuracy  : {task_il_acc:.4f}")
    logging.info(f"Backward Transfer : {bwt:.4f}")
    logging.info(f"Forward Transfer  : {fwt:.4f}")
    logging.info(f"Forgetting Measure: {fm:.4f}")
    logging.info("="*50)

    logging.info(">>>> ACCURACY MATRIX <<<<")
    for row in acc_matrix:
        logging.info(row)

    return class_il_aa, task_il_acc, bwt, fwt, fm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmbd',         type=float, default=0.6,   help='Weight for IRD loss (default: 1.0)')
    parser.add_argument('--trials',       type=int,   default=5,     help='Number of independent runs to average results over')
    parser.add_argument('--t_contrast',   type=float, default=0.07,  help='Temperature for contrastive loss')
    parser.add_argument('--t_student',    type=float, default=0.07,  help='Temperature for student in IRD loss')
    parser.add_argument('--t_teacher',    type=float, default=0.07,  help='Temperature for teacher in IRD loss')
    parser.add_argument('--epochs_train', type=int,   default=100,   help='Epochs for contrastive training per task')
    parser.add_argument('--epochs_eval',  type=int,   default=100,   help='Epochs for linear probe classifier')
    parser.add_argument('--metrics',      action='store_true', default=False, help='Compute CL metrics (linear probe, FWT, BWT, FM) — disabled by default for speed')
    parser.add_argument('--alpha_lump',   type=float, default=0.1,  help='Beta distribution alpha for LUMP mixup')
    parser.add_argument('--dataset',      type=str,   default='eurosat', choices=['cifar10', 'tinyimagenet', 'eurosat'], help='Dataset to use (defined in data.yml)')
    parser.add_argument('--buffer_size',  type=int,   default=500,   help='Replay buffer size (default: 500)')
    parser.add_argument('--log_file',     type=str,   default='./log_tuning/eurosat.log',  help='Log file path (default: <datetime>.log)')
    parser.add_argument('--gpu',          type=int,   default=1,     help='GPU id to use (default: 0)')
    args = parser.parse_args()

    log_dir  = './log_tuning'
    os.makedirs(log_dir, exist_ok=True)
    log_file = args.log_file if args.log_file else os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
    logging.info("="*50)
    logging.info(">>>> STARTING EXPERIMENT <<<<")
    logging.info(f"  dataset       = {args.dataset}")
    logging.info(f"  lmbd          = {args.lmbd}")
    logging.info(f"  trials        = {args.trials}")
    logging.info(f"  epochs_train  = {args.epochs_train}")
    logging.info(f"  epochs_eval   = {args.epochs_eval}")
    logging.info(f"  t_contrast    = {args.t_contrast}")
    logging.info(f"  t_student     = {args.t_student}")
    logging.info(f"  t_teacher     = {args.t_teacher}")
    logging.info(f"  metrics       = {args.metrics}")
    logging.info(f"  alpha_lump    = {args.alpha_lump}")
    logging.info(f"  buffer_size   = {args.buffer_size}")
    logging.info(f"  log_file      = {log_file}")
    logging.info("="*50)

    all_cil, all_til, all_bwt, all_fwt, all_fm = [], [], [], [], []

    for i in range(args.trials):
        start_time = time.time()
        logging.info(f">>>> TRIAL {i+1}/{args.trials} <<<<")

        cil, til, bwt, fwt, fm = main(args.lmbd, args.t_contrast, args.t_student, args.t_teacher,
                                               args.epochs_train, args.epochs_eval, args.metrics,
                                               args.alpha_lump, args.dataset, args.buffer_size)

        all_cil.append(cil)
        all_til.append(til)
        if args.metrics:
            all_bwt.append(bwt)
            all_fwt.append(fwt)
            all_fm.append(fm)

        logging.info(f"Trial {i+1} completed in {time.time() - start_time:.2f} seconds")
        torch.cuda.empty_cache()

    mean_cil, std_cil = calculate_stats(all_cil)
    mean_til, std_til = calculate_stats(all_til)
    
    logging.info("="*50)
    logging.info(">>>> ENDING EXPERIMENT <<<<")
    logging.info(f"  lmbd          = {args.lmbd}")
    logging.info(f"  trials        = {args.trials}")
    logging.info(f"  epochs_train  = {args.epochs_train}")
    logging.info(f"  epochs_eval   = {args.epochs_eval}")
    logging.info(f"  t_contrast    = {args.t_contrast}")
    logging.info(f"  t_student     = {args.t_student}")
    logging.info(f"  t_teacher     = {args.t_teacher}")
    logging.info(f"  metrics       = {args.metrics}")
    logging.info(f"  alpha_lump    = {args.alpha_lump}")
    logging.info(f"  buffer_size   = {args.buffer_size}")
    logging.info(f"  log_file      = {log_file}")
    logging.info("="*50)
    logging.info("="*50)
    logging.info(f">>>> SUMMARY lmbd={args.lmbd} <<<<")
    logging.info("="*50)
    logging.info(f"Class-IL Accuracy : {mean_cil:.2f} ± {std_cil:.2f}")
    logging.info(f"Task-IL Accuracy  : {mean_til:.2f} ± {std_til:.2f}")
    if args.metrics and all_bwt:
        mean_bwt, std_bwt = calculate_stats(all_bwt)
        mean_fwt, std_fwt = calculate_stats(all_fwt)
        mean_fm,  std_fm  = calculate_stats(all_fm)
        logging.info(f"Backward Transfer : {mean_bwt:.4f} ± {std_bwt:.4f}")
        logging.info(f"Forward Transfer  : {mean_fwt:.4f} ± {std_fwt:.4f}")
        logging.info(f"Forgetting Measure: {mean_fm:.4f} ± {std_fm:.4f}")
    logging.info("="*50)
