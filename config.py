# config.py

EPOCH = 60 # Number of Epoch for BiLSTM

EPOCHCAPS = 50 # Number of Epoch for CapsNET

BATCH = 64 # Batch Size

BATCHCAPS = 128 # CapsNet Batch size

lrCapsNET = 0.001 # CapsNET Learning

lrBiLSTM = 0.0007 # BiLSTM Learning Rate

BiLSTM_LOSS_FUNCTION = "eamcl"  # Options: "eamcl" or "crossentropy" for the BiLSTM model

CAPS_LOSS_FUNCTION = "eamcl"    # Options: "eamcl" or "margin" for the CapsNet model

BiDi_LOSS_FUNCTION = "eamcl"    # Options: "eamcl" or "margin" for the CapsNet model