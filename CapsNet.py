import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.model_selection import StratifiedShuffleSplit

# Import global configuration and data.
import config
from data_config import X_train, y_train, X_test, y_test, signal_length

# ---------------------------
# CapsuleLayer Definition
# ---------------------------
# Capsule Layer with Enhanced Dynamic Routing
class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, num_routes, kernel_size=None, stride=None, routing_iters=5):
        super().__init__()
        self.num_capsules  = num_capsules
        self.num_routes    = num_routes
        self.routing_iters = routing_iters

        if num_routes == -1:
            self.capsules = nn.ModuleList([
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=stride)
                for _ in range(num_capsules)
            ])
        else:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_routes, in_channels, out_channels))

    def squash(self, input_tensor, dim=-1):
        norm = torch.norm(input_tensor, dim=dim, keepdim=True)
        scale = norm ** 2 / (1 + norm ** 2)
        return scale * input_tensor / (norm + 1e-8)

    def forward(self, x):
        if hasattr(self, 'capsules'):
            u = [capsule(x) for capsule in self.capsules]
            u = torch.stack(u, dim=2)
            batch_size = u.size(0)
            u = u.permute(0, 3, 2, 1).contiguous()
            u = u.view(batch_size, -1, u.size(-1))
            return self.squash(u)
        else:
            batch_size = x.size(0)
            x = x.unsqueeze(1).unsqueeze(-1)
            x = x.expand(-1, self.num_capsules, -1, -1, -1)
            weights = self.route_weights.unsqueeze(0)
            u_hat = torch.matmul(weights.transpose(-1, -2), x).squeeze(-1)
            b_ij = torch.zeros(batch_size, self.num_capsules, self.num_routes, device=x.device)
            for iteration in range(self.routing_iters):
                c_ij = F.softmax(b_ij, dim=1).unsqueeze(-1)
                s_j = (c_ij * u_hat).sum(dim=2)
                v_j = self.squash(s_j)
                if iteration < self.routing_iters - 1:
                    b_ij += (u_hat * v_j.unsqueeze(2)).sum(-1)
            return v_j

# Multi-head Capsule Attention
class MultiHeadCapsuleAttention(nn.Module):
    def __init__(self, capsule_dim, heads=4):
        super(MultiHeadCapsuleAttention, self).__init__()
        self.heads = heads
        self.attentions = nn.ModuleList([CapsuleAttention(capsule_dim) for _ in range(heads)])
        self.fc_out = nn.Linear(capsule_dim * heads, capsule_dim)

    def forward(self, x, return_attention=False):
        attn_outputs, attn_weights_all = [], []
        for att in self.attentions:
            if return_attention:
                out, weights = att(x, return_attention=True)
                attn_outputs.append(out)
                attn_weights_all.append(weights)
            else:
                out = att(x)
                attn_outputs.append(out)
        out = torch.cat(attn_outputs, dim=-1)
        if return_attention:
            attn_weights_all = torch.stack(attn_weights_all)  # (heads, batch, M, M)
            attn_weights_all = attn_weights_all.permute(1, 0, 2, 3)  # (batch, heads, M, M)
            return self.fc_out(out), attn_weights_all
        return self.fc_out(out)

# Capsule Attention
class CapsuleAttention(nn.Module):
    def __init__(self, capsule_dim):
        super(CapsuleAttention, self).__init__()
        self.query = nn.Linear(capsule_dim, capsule_dim)
        self.key = nn.Linear(capsule_dim, capsule_dim)
        self.value = nn.Linear(capsule_dim, capsule_dim)
        self.scale = capsule_dim ** -0.5

    def forward(self, x, return_attention=False):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn_weights = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        output = (attn_weights @ V) + x
        if return_attention:
            return output, attn_weights
        return output


# Enhanced ECGCapsNet
class ECGCapsNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Retrieve signal length from data_config
        self.signal_length = signal_length

        # --- 1) First convolutional layer with padding to preserve sequence length ---
        k1, s1 = 9, 1
        self.conv1 = nn.Conv1d(1, 64, kernel_size=k1, stride=s1)
        self.bn1   = nn.BatchNorm1d(64)
        conv1_out  = (self.signal_length - k1) // s1 + 1  # = 180 for signal_length=188

        # --- 2) Primary capsules: Valid conv-> 16-d maps, stride 2
        num_primary_caps = 8
        k2, s2 = 9, 2
        self.primary_capsules = CapsuleLayer(
            num_capsules=num_primary_caps,
            in_channels=64,
            out_channels=16,
            num_routes=-1,
            kernel_size=k2,
            stride=s2
        )
        primary_out    = (conv1_out - k2) // s2 + 1  # 86
        num_primary_routes = primary_out * num_primary_caps  # = 86 * 8 = 688

        # --- 3) Secondary capsules: 10 capsules routing from primary routes ---
        self.secondary_capsules = CapsuleLayer(
            num_capsules=10,
            in_channels=16,
            out_channels=32,
            num_routes=num_primary_routes,  # = 688
            routing_iters=5
        )

        # --- 4) Class capsules: one capsule per class (5), 32-d output each ---
        self.class_capsules = CapsuleLayer(
            num_capsules=5,
            in_channels=32,
            out_channels=32,
            num_routes=10,       # routing from 10 secondary capsules
            routing_iters=5
        )

        # --- 5) Multi-head attention over the 5 class capsules ---
        self.capsule_attention = MultiHeadCapsuleAttention(capsule_dim=32, heads=2)

        # --- 6) Final decoder/classifier ---
        self.decoder = nn.Sequential(
            nn.Linear(32 * 5, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)
        )

        # --- 7) Loss function selection ---
        if config.CAPS_LOSS_FUNCTION == "eamcl":
            from EAMCL import AdaptiveMarginContextLoss
            self.loss_fn = AdaptiveMarginContextLoss(y_train, max_m=0.5, lambda_coef=0.1)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: [batch, 188]
        x = x.unsqueeze(1)                              # → [batch, 1, 188]
        x = F.relu(self.bn1(self.conv1(x)))             # → [batch, 64, 188]
        x = self.primary_capsules(x)                    # → [batch, primary_out*8, 16]
        x = self.secondary_capsules(x)                  # → [batch, 10, 32]
        class_caps = self.class_capsules(x)             # → [batch, 5, 32]
        class_caps = self.capsule_attention(class_caps) # → [batch, 5, 32]
        flat = class_caps.view(class_caps.size(0), -1)  # → [batch, 32*5]
        return self.decoder(flat)                       # → [batch, 5]

    def training_step(self, batch, batch_idx):
        x, y     = batch
        logits   = self(x)
        loss     = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y     = batch
        logits   = self(x)
        loss     = self.loss_fn(logits, y)
        preds    = torch.argmax(logits, dim=1)
        self.log_dict({
            'val_loss':      loss,
            'val_acc':       Accuracy(task='multiclass', num_classes=5).to(self.device)(preds, y),
            'val_precision': Precision(task='multiclass', num_classes=5).to(self.device)(preds, y),
            'val_recall':    Recall(task='multiclass', num_classes=5).to(self.device)(preds, y),
            'val_f1':        F1Score(task='multiclass', num_classes=5).to(self.device)(preds, y),
        }, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # reuse validation logic for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

# ---------------------------
# 6. Training, Cross Validation, and Final Testing
# ---------------------------
if __name__ == '__main__':
    """
    # 10-Fold Stratified Shuffle Split (80% train, 20% validation per fold) on training data.
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    cv_results = {}
    fold_metrics_list = []
    
    for fold, (train_idx, val_idx) in enumerate(sss.split(X_train, y_train)):
        print(f"\n=== CV Fold {fold+1} ===")
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        train_dataset = TensorDataset(torch.tensor(X_train_fold).float(), torch.tensor(y_train_fold).long())
        val_dataset = TensorDataset(torch.tensor(X_val_fold).float(), torch.tensor(y_val_fold).long())
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, persistent_workers=True)
        
        model = ECGCapsNet()  # No parameters needed here.
        trainer = pl.Trainer(
            max_epochs=config.EPOCHCAPS,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_checkpointing=False
        )
        trainer.fit(model, train_loader, val_loader)
        fold_metric = {
            "val_loss": trainer.callback_metrics.get("val_loss"),
            "val_acc": trainer.callback_metrics.get("val_acc"),
            "val_precision": trainer.callback_metrics.get("val_precision"),
            "val_recall": trainer.callback_metrics.get("val_recall"),
            "val_f1": trainer.callback_metrics.get("val_f1")
        }
        cv_results[f"Fold {fold+1}"] = fold_metric
        fold_metrics_list.append(fold_metric)
    
    avg_loss = np.mean([m["val_loss"].item() for m in fold_metrics_list if m["val_loss"] is not None])
    avg_acc = np.mean([m["val_acc"].item() for m in fold_metrics_list if m["val_acc"] is not None])
    avg_prec = np.mean([m["val_precision"].item() for m in fold_metrics_list if m["val_precision"] is not None])
    avg_rec = np.mean([m["val_recall"].item() for m in fold_metrics_list if m["val_recall"] is not None])
    avg_f1 = np.mean([m["val_f1"].item() for m in fold_metrics_list if m["val_f1"] is not None])
    
    print("\n=== Cross Validation Results (Average over 10 folds) ===")
    print(f"Loss Function: {config.CAPS_LOSS_FUNCTION}")
    print(f"  Avg Validation Loss : {avg_loss}")
    print(f"  Avg Accuracy        : {avg_acc}")
    print(f"  Avg Precision       : {avg_prec}")
    print(f"  Avg Recall          : {avg_rec}")
    print(f"  Avg F1 Score        : {avg_f1}")
    

    """
    # ---- Final Training on Full Training Data & Testing on Test Dataset ----
    full_train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    full_train_loader = DataLoader(full_train_dataset, batch_size=128, shuffle=True, num_workers=8, persistent_workers=True)
    
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, persistent_workers=True)
    
    model = ECGCapsNet()
    trainer = pl.Trainer(
        max_epochs=config.EPOCHCAPS,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False
    )
    # Use the full training set as both training and validation for metric monitoring.
    trainer.fit(model, full_train_loader, full_train_loader)
    
    # Save the final trained model.
    torch.save(model.state_dict(), 'high_level_capsnet_adaptive.pth')
    
    trainer.test(model, test_loader)
    
    final_metrics = trainer.callback_metrics
    print("\n=== Final Test Metrics ===")
    print(f"- Accuracy: {final_metrics.get('test_acc')}")
    print(f"- Precision: {final_metrics.get('test_precision')}")
    print(f"- Recall: {final_metrics.get('test_recall')}")
    print(f"- F1 Score: {final_metrics.get('test_f1')}")
