import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping, Callback

# ---------------------------
# Import data configuration and global config
# ---------------------------
import config
from data_config import X_train, y_train, X_test, y_test

# =====================================
# Import Pre-trained Models (Updated)
# =====================================
from BiLSTM import LowLevelBiLSTM
from CapsNet import ECGCapsNet

# If using EAMCL for CapsNet, import it.
if config.BiDi_LOSS_FUNCTION == "eamcl":
    from EAMCL import AdaptiveMarginContextLoss

# Load pre-trained weights explicitly
low_model = LowLevelBiLSTM()
low_model.load_state_dict(torch.load('low_level_bilstm_adaptive.pth'))
low_model.eval()

signal_length = X_train.shape[1]
high_model = ECGCapsNet()
checkpoint = torch.load('high_level_capsnet_adaptive.pth')
high_model.eval()

# Freeze pre-trained models initially (recommended)
for param in low_model.parameters():
    param.requires_grad = False

for param in high_model.parameters():
    param.requires_grad = False

# ===================================
# Bi-Directional Attention Module
# ===================================
# Multi-head Bi-Directional Attention with Adaptive Gating
class MultiHeadAdaptiveFusion(nn.Module):
    def __init__(self, low_dim, high_dim, attn_dim, num_heads=4):
        super().__init__()
        self.attn_dim = attn_dim

        self.low_norm = nn.LayerNorm(attn_dim)
        self.high_norm = nn.LayerNorm(attn_dim)

        # Cross attentions (batch_first=True is important)
        self.low_to_high_attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.high_to_low_attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True, dropout=0.1)

        # Project branch features to a common attention dim
        self.low_proj  = nn.Linear(low_dim,  attn_dim)
        self.high_proj = nn.Linear(high_dim, attn_dim)

        # Per-time-step gate: maps [*, 2D] -> [*, D]
        self.adaptive_gate = nn.Sequential(
            nn.Linear(attn_dim * 2, attn_dim),
            nn.Sigmoid()
        )

        # Temporal refinement over fused sequence (light, stable)
        self.fusion_conv = nn.Conv1d(in_channels=attn_dim, out_channels=attn_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)
        )

    def _align_time(self, x, target_T):
        """x: [B, T, D] -> resize time dim to target_T via linear interpolation."""
        if x.size(1) == target_T:
            return x
        # [B, T, D] -> [B, D, T] -> interpolate -> [B, T, D]
        return F.interpolate(x.permute(0, 2, 1), size=target_T, mode='linear', align_corners=False).permute(0, 2, 1)

    def forward(self, low_feat, high_feat):
        # 1) project to common dim
        # In forward, replace the projection lines with:
        low_q  = self.low_norm(self.low_proj(low_feat))    # [B, T_low,  D]
        high_k = self.high_norm(self.high_proj(high_feat)) # [B, T_high, D]
        #low_q  = self.low_proj(low_feat)    # [B, T_low,  D]
        #high_k = self.high_proj(high_feat)  # [B, T_high, D]

        # 2) bi-directional cross attention
        #    output shapes follow the query length
        low_high,  _ = self.low_to_high_attn (query=low_q,  key=high_k, value=high_k)   # [B, T_low,  D]
        high_low, _  = self.high_to_low_attn(query=high_k, key=low_q,  value=low_q)     # [B, T_high, D]

        # 3) align sequences so we can fuse per-time-step
        #T = low_high.size(1)
        #high_low_aligned = self._align_time(high_low, target_T=T)  # [B, T, D]
        T = max(low_q.size(1), high_k.size(1))
        low_high = self._align_time(low_high, target_T=T)  # Align low_high if necessary
        high_low_aligned = self._align_time(high_low, target_T=T)

        # 4) per-time-step gate in sequence space
        combined_seq = torch.cat([low_high, high_low_aligned], dim=-1)  # [B, T, 2D]
        gate_seq = self.adaptive_gate(combined_seq)                     # [B, T, D]
        fused_seq = gate_seq * low_high + (1.0 - gate_seq) * high_low_aligned  # [B, T, D]

        # 5) temporal refinement + late pooling
        z = fused_seq.permute(0, 2, 1)           # [B, D, T]
        z = F.relu(self.fusion_conv(z))          # [B, D, T]
        z = self.pool(z).squeeze(-1)             # [B, D]

        # 6) classify
        return self.classifier(z)

# Full Lightning Module incorporating Bi-directional Fusion
class CombinedECGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.low_model = low_model
        self.high_model = high_model

        self.bi_attn = MultiHeadAdaptiveFusion(low_dim=64, high_dim=32, attn_dim=64, num_heads=4)

        if config.BiDi_LOSS_FUNCTION == "eamcl":
            self.loss_fn = AdaptiveMarginContextLoss(y_train, max_m=0.5, lambda_coef=0.1)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        with torch.no_grad():
            # Enhanced Low-Level feature extraction
            x_low = x.unsqueeze(1)  # [batch, 1, 188]

            # Multi-scale convolutions
            x_low_3 = F.relu(self.low_model.conv3(x_low))
            x_low_5 = F.relu(self.low_model.conv5(x_low))
            x_low_7 = F.relu(self.low_model.conv7(x_low))

            # Concatenate multi-scale outputs
            x_low = torch.cat([x_low_3, x_low_5, x_low_7], dim=1)  # [batch, 48, 188]
            x_low = F.relu(self.low_model.bn1(x_low))

            # Final convolution
            x_low = F.relu(self.low_model.bn2(self.low_model.conv_final(x_low)))

            # SE Attention
            x_low = self.low_model.se(x_low)

            # Prepare for LSTM
            x_low = x_low.permute(0, 2, 1)  # [batch, seq_len, channels]

            # BiLSTM
            low_features, _ = self.low_model.bilstm(x_low)

            # Residual connection
            x_res = torch.cat([x_low, x_low], dim=-1)
            low_features = low_features + x_res

            # High-Level (Capsule) features extraction (Corrected)
            x_high = x.unsqueeze(1)  # [batch, 1, 188]
            x_high = F.relu(self.high_model.bn1(self.high_model.conv1(x_high)))

            primary_caps_output = self.high_model.primary_capsules(x_high)
            secondary_caps_output = self.high_model.secondary_capsules(primary_caps_output)
            capsule_features = self.high_model.class_capsules(secondary_caps_output)
            capsule_features = self.high_model.capsule_attention(capsule_features)

        # Bi-Directional Attention fusion
        output = self.bi_attn(low_features, capsule_features)
        return output


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        # log both
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc',  acc,  prog_bar=True)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

# ---------------------------
# Custom callback to record losses
# ---------------------------
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses   = []
        self.train_accs   = []
        self.val_accs     = []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.train_losses.append(m['train_loss'].cpu().item())
        self.train_accs.append(   m['train_acc'].cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.val_losses.append(m['val_loss'].cpu().item())
        self.val_accs.append(   m['val_acc'].cpu().item())
    
# ============================
# 10-Fold Cross Validation and Final Testing
# ============================
if __name__ == '__main__':
    # 1) Merge & split
    Xc = np.concatenate([X_train, X_test], axis=0)
    yc = np.concatenate([y_train, y_test], axis=0)
    Xtr, Xte, ytr, yte = train_test_split(
        Xc, yc, test_size=0.2, stratify=yc, random_state=42
    )

    train_ds = TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr).long())
    test_ds  = TensorDataset(torch.tensor(Xte).float(), torch.tensor(yte).long())

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=8)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=8)

    # 2) Init model & load branches
    model = CombinedECGModel()
    model.low_model.load_state_dict(torch.load('low_level_bilstm_adaptive.pth'))
    model.high_model.load_state_dict(torch.load('high_level_capsnet_adaptive.pth'))

    # 3) Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, mode='min', verbose=True)
    history       = LossHistory()

    # 4) Trainer & train
    trainer = pl.Trainer(
        max_epochs=config.EPOCH,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[early_stopping, history]
    )
    trainer.fit(model, train_loader, test_loader)

    epochs_ran = len(history.train_losses)
    print(f"Stopped after {epochs_ran} epochs (max {config.EPOCH})")

    # 5) Save
    torch.save(model.state_dict(), 'BiDirectional_Fusion_EarlyStopped.pth')

    # 6) Final test & print
    trainer.test(model, test_loader)
    fm = trainer.callback_metrics
    print("\n=== Final Test Metrics ===")
    print(f"Acc : {fm['val_acc']:.4f}")
    print(f"Prec: {fm['val_precision']:.4f}")
    print(f"Rec : {fm['val_recall']:.4f}")
    print(f"F1  : {fm['val_f1']:.4f}")

    # 7) Plot loss & accuracy
    # 7) Plot loss & accuracy, but first align lengths
    n_epochs = min(len(history.train_losses), len(history.val_losses))
    epochs   = np.arange(1, n_epochs + 1)

    # 1) grab histories
    train_losses = history.train_losses
    val_losses   = history.val_losses
    train_accs   = history.train_accs
    val_accs     = history.val_accs

    # 2) if val has one extra (initial) element, drop it
    if len(val_losses) > len(train_losses):
        val_losses = val_losses[1:]
        val_accs   = val_accs[1:]

    # 3) truncate to the same length (just in case)
    n = min(len(train_losses), len(val_losses))
    train_losses = train_losses[:n]
    val_losses   = val_losses[:n]
    train_accs   = train_accs[:n]
    val_accs     = val_accs[:n]

    # 4) build epoch axis starting at 1
    epochs = np.arange(1, n+1)

    # convert to percent
    train_accs_pct = np.array(history.train_accs[:n]) * 100
    val_accs_pct   = np.array(history.val_accs[:n]) * 100

    plt.figure(figsize=(12,5))

    # Loss subplot (unchanged)
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Loss Curve'); plt.legend()

    # Accuracy subplot: in percent, y-range from 90 to 100
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs_pct, label='Train Acc (%)')
    plt.plot(epochs, val_accs_pct,   label='Val Acc (%)')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.ylim(90, 100)
    plt.yticks(np.arange(90, 101, 2))
    plt.legend()

    plt.tight_layout()
    plt.show()