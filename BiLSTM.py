import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.model_selection import StratifiedShuffleSplit
from EAMCL import AdaptiveMarginContextLoss
from sklearn.model_selection import train_test_split

# Import data configuration and global config.
from data_config import X_train, y_train, X_test, y_test, signal_length
import config

# Set torch environment settings.
torch.set_float32_matmul_precision('medium')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# ---------------------------------------------------
# 1) Define the GradNormCallback
# ---------------------------------------------------
class GradNormCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.last_grad_norm = None

    def on_after_backward(self, trainer, pl_module):
        # Calculate total gradient norm (L2) for all parameters
        total_norm_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2).item()
                total_norm_sq += param_norm ** 2
        grad_norm = total_norm_sq ** 0.5
        self.last_grad_norm = grad_norm
        pl_module.log("grad_norm", grad_norm, prog_bar=False, on_step=False, on_epoch=True)


# ---------------------------
# Helper Functins
# ---------------------------
def get_attention_stats(model, dataloader, device):
    model.eval()
    model = model.to(device)
    attn_means, attn_stds, attn_mins, attn_maxs, attn_entropies = [], [], [], [], []
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            _, attn_weights = model(xb, return_attention=True)  # [batch, 188]
            attn_weights = attn_weights.cpu().numpy()
            # Calculate stats per sample, then average over batch
            attn_means.append(attn_weights.mean(axis=1))
            attn_stds.append(attn_weights.std(axis=1))
            attn_mins.append(attn_weights.min(axis=1))
            attn_maxs.append(attn_weights.max(axis=1))
            # Entropy: -sum(p*log(p)) over the 188 dim
            attn_entropies.append((-attn_weights * np.log(attn_weights + 1e-8)).sum(axis=1))
            break  # Only need one batch for monitoring!
    return (
        float(np.mean(attn_means)), float(np.mean(attn_stds)),
        float(np.min(attn_mins)), float(np.max(attn_maxs)),
        float(np.mean(attn_entropies))
    )

def get_grad_norm(model):
    norm = 0.
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            norm += param_norm ** 2
    return norm ** 0.5


# ---------------------------
# Metric Logger Callback
# ---------------------------
class MetricLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses     = []
        self.val_losses       = []
        self.train_accuracies = []   # ← ensure this list exists
        self.val_accuracies   = []

    def on_train_epoch_end(self, trainer, pl_module):
        # 1) Get the latest train_loss (you already do this)
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(
                loss.cpu().item() if hasattr(loss, 'cpu') else float(loss)
            )

        # 2) Compute train accuracy by doing one pass over the **training** DataLoader
        #    (we assume `pl_module.train_dataloader()` works or you pass in train_dataloader)
        preds, targs = [], []
        pl_module.eval()
        with torch.no_grad():
            for xb, yb in trainer.train_dataloader:    # iterate over entire train set
                xb, yb = xb.to(pl_module.device), yb.to(pl_module.device)
                out = pl_module(xb)
                preds.append(out.argmax(dim=1).cpu())
                targs.append(yb.cpu())
            preds = torch.cat(preds, dim=0)
            targs = torch.cat(targs, dim=0)
            acc = (preds == targs).float().mean().item()
            self.train_accuracies.append(acc)
        pl_module.train()  # switch back to train mode

    def on_validation_epoch_end(self, trainer, pl_module):
        # your existing code:
        loss = trainer.callback_metrics.get("val_loss")
        acc  = trainer.callback_metrics.get("val_acc")
        if loss is not None:
            self.val_losses.append(
                loss.cpu().item() if hasattr(loss, 'cpu') else float(loss)
            )
        if acc is not None:
            self.val_accuracies.append(
                acc.cpu().item() if hasattr(acc, 'cpu') else float(acc)
            )


# ---------------------------
# Low Level Features Extraction - BiLSTM Model
# ---------------------------
# Squeeze-and-Excitation (SE) module
class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# Enhanced Low-Level BiLSTM
class LowLevelBiLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Multi-scale convolutional layers
        self.conv3 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(48)  # 16 channels * 3 conv layers

        self.conv_final = nn.Conv1d(48, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        # SE Attention
        self.se = SEModule(channels=32)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )

        # Attention
        self.attention_conv = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.attention_linear = nn.Linear(64, 1)

        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 5)
        )

        # Loss function
        if config.BiLSTM_LOSS_FUNCTION == "eamcl":
            self.loss_fn = AdaptiveMarginContextLoss(y_train, max_m=0.5, lambda_coef=0.1)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, return_attention=False):
        x = x.unsqueeze(1)  # [batch, 1, 188]

        # Multi-scale convolutions
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))

        # Concatenate multi-scale outputs
        x = torch.cat([x3, x5, x7], dim=1)  # [batch, 48, 188]
        x = F.relu(self.bn1(x))

        # Final convolution
        x = F.relu(self.bn2(self.conv_final(x)))  # [batch, 32, 188]
        x = self.se(x)  # Apply SE attention

        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # [batch, 188, 32]
        lstm_out, _ = self.bilstm(x)  # [batch, 188, 64]

        # Residual Connection
        x_res = torch.cat([x, x], dim=-1)  # [batch, 188, 64]
        lstm_out = lstm_out + x_res

        # Attention mechanism
        attn_weights = torch.tanh(self.attention_conv(lstm_out.permute(0, 2, 1)))
        attn_weights = self.attention_linear(attn_weights.permute(0, 2, 1)).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch, 188]
        attn_applied = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        # Classification
        output = self.fc(attn_applied)
        if return_attention:
            return output, attn_weights
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = {
            'val_loss': loss,
            'val_acc': Accuracy(task='multiclass', num_classes=5).to(self.device)(preds, y),
            'val_precision': Precision(task='multiclass', num_classes=5).to(self.device)(preds, y),
            'val_recall': Recall(task='multiclass', num_classes=5).to(self.device)(preds, y),
            'val_f1': F1Score(task='multiclass', num_classes=5).to(self.device)(preds, y)
        }
        self.log_dict(metrics, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # You can simply reuse the validation logic if it fits your testing scenario.
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.lrBiLSTM, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

# ---------------------------
# 4. 10-Fold Cross Validation & Final Testing
# ---------------------------
if __name__ == '__main__':
    """
    # Create 10 stratified splits (80% train, 20% validation per fold).
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    cv_results = {}
    fold_metrics_list = []
    monitoring_table = []

    for fold, (train_idx, val_idx) in enumerate(sss.split(X_train, y_train)):
        print(f"\n=== CV Fold {fold+1} ===")
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        train_dataset = TensorDataset(torch.tensor(X_train_fold).float(), torch.tensor(y_train_fold).long())
        val_dataset = TensorDataset(torch.tensor(X_val_fold).float(), torch.tensor(y_val_fold).long())
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH, shuffle=True, num_workers=3, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH, shuffle=False, num_workers=3, persistent_workers=True)
        
        # Select loss function based on config.
        if config.BiLSTM_LOSS_FUNCTION == "eamcl":
            # Create an instance of EAMCL using training labels.
            loss_fn = AdaptiveMarginContextLoss(y_train, max_m=0.5, lambda_coef=0.1)
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        model = LowLevelBiLSTM()
        metric_logger = MetricLogger()
        gradnorm_cb   = GradNormCallback()
        trainer = pl.Trainer(
            max_epochs=config.EPOCH,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            callbacks=[metric_logger, gradnorm_cb]
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

        attn_mean, attn_std, attn_min, attn_max, attn_entropy = get_attention_stats(model, val_loader, device="cuda" if torch.cuda.is_available() else "cpu")
        grad_norm = gradnorm_cb.last_grad_norm if gradnorm_cb.last_grad_norm is not None else 0.0

        monitor_row = {
            "fold": fold + 1,
            "val_loss": float(fold_metric["val_loss"]),
            "attn_mean_std": f"{attn_mean:.4f} ± {attn_std:.4f}",
            "attn_min": attn_min,
            "attn_max": attn_max,
            "attn_entropy": attn_entropy,
            "grad_norm": grad_norm
        }
        monitoring_table.append(monitor_row)
    

    print("=== Attention and Stability Monitoring Table (BiLSTM) ===")
    print(f"|{'Fold':^6}|{'Loss':^8}|{'Attn. mean ± std':^18}|{'Attn. min':^12}|{'Attn. max':^12}|{'Attn. entropy':^16}|{'Grad norm':^10}|")
    print("|"+"-"*6+"|"+"-"*8+"|"+"-"*18+"|"+"-"*12+"|"+"-"*12+"|"+"-"*16+"|"+"-"*10+"|")
    for row in monitoring_table:
        print(f"|{row['fold']:^6}|{row['val_loss']:^8.4f}|{row['attn_mean_std']:^18}|{row['attn_min']:^12.4f}|{row['attn_max']:^12.4f}|{row['attn_entropy']:^16.4f}|{row['grad_norm']:^10.4f}|")


    avg_loss = np.mean([m["val_loss"] for m in fold_metrics_list if m["val_loss"] is not None])
    avg_acc = np.mean([m["val_acc"] for m in fold_metrics_list if m["val_acc"] is not None])
    avg_prec = np.mean([m["val_precision"] for m in fold_metrics_list if m["val_precision"] is not None])
    avg_rec = np.mean([m["val_recall"] for m in fold_metrics_list if m["val_recall"] is not None])
    avg_f1 = np.mean([m["val_f1"] for m in fold_metrics_list if m["val_f1"] is not None])
    
    print("\n=== Cross Validation Results (Average over 10 folds) ===")
    print(f"Loss Function: {config.BiLSTM_LOSS_FUNCTION}")
    print(f"  Avg Validation Loss : {avg_loss}")
    print(f"  Avg Accuracy        : {avg_acc}")
    print(f"  Avg Precision       : {avg_prec}")
    print(f"  Avg Recall          : {avg_rec}")
    print(f"  Avg F1 Score        : {avg_f1}")
"""


    
    # Final training on full training data and testing on test dataset.

    # … (all of your code up to final training) …

    # Split X_train/y_train into train and validation sets (10% val)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
    )

    train_dataset = TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr).long())
    val_dataset   = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH, shuffle=True,  num_workers=8, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH, shuffle=False, num_workers=8, persistent_workers=True)
        
    final_model = LowLevelBiLSTM()
    metric_logger = MetricLogger()

    trainer = pl.Trainer(
        max_epochs=config.EPOCH,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[metric_logger],
        num_sanity_val_steps=0
    )

    # Training
    trainer.fit(final_model, train_loader, val_loader)

    # ─────────────────────────────────────────────────────────────────────────────
    # Plotting (Fix 2): Separate “Loss” and “Accuracy” loops so that missing train_acc
    # does not zero‐out everything.
    #
    # 1) Plot Loss over epochs
    # 2) Plot Accuracy over epochs (only plot whatever is available)
    # ─────────────────────────────────────────────────────────────────────────────

    # a) Figure out how many epochs we have recorded for train‐loss vs val‐loss
    epoch_loss_len = min(
        len(metric_logger.train_losses),
        len(metric_logger.val_losses)
    )
    epochs_loss = range(1, epoch_loss_len + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    if epoch_loss_len > 0:
        plt.plot(
            epochs_loss,
            metric_logger.train_losses[:epoch_loss_len],
            label="Train Loss"
        )
        plt.plot(
            epochs_loss,
            metric_logger.val_losses[:epoch_loss_len],
            label="Val Loss"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # b) Plot Accuracy (training and/or validation, whichever exists)
    # We know we at least have val_accuracy logged by on_validation_epoch_end.
    epoch_val_acc_len = len(metric_logger.val_accuracies)
    epochs_acc = range(1, epoch_val_acc_len + 1)

    plt.subplot(1, 2, 2)
    # If train_acc was ever logged, plot it; otherwise skip it.
    if len(metric_logger.train_accuracies) > 0:
        # Only plot up to whichever is shorter (train_acc vs val_acc)
        train_acc_len = min(len(metric_logger.train_accuracies), epoch_val_acc_len)
        plt.plot(
            range(1, train_acc_len + 1),
            metric_logger.train_accuracies[:train_acc_len],
            label="Train Accuracy"
        )

    # Plot whatever validation‐accuracy we have
    if epoch_val_acc_len > 0:
        plt.plot(
            epochs_acc,
            metric_logger.val_accuracies[:epoch_val_acc_len],
            label="Val Accuracy"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────

    # Save the final trained model.
    torch.save(final_model.state_dict(), 'low_level_bilstm_adaptive.pth')

    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH, shuffle=False, num_workers=8, persistent_workers=True)

    # Run test
    trainer.test(final_model, test_loader)
    final_metrics = trainer.callback_metrics
    print("\n=== Final Test Metrics ===")
    print(f"- Accuracy: {final_metrics.get('test_acc')}")
    print(f"- Precision: {final_metrics.get('test_precision')}")
    print(f"- Recall: {final_metrics.get('test_recall')}")
    print(f"- F1 Score: {final_metrics.get('test_f1')}")