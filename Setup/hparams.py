"""Per-dataset CONAP-FM hyperparameters, exactly as in Table 1 of the paper.

W = window size, lr = learning rate (eta), tau = variance threshold, r = LoRA rank, alpha = LoRA alpha.
W, tau, epsilon and the cooldown C are tuned on the 15% validation split with grid_search.py, the chosen values are in Setup/final.py.
Shared settings for all datasets (code defaults) are in SHARED.
"""

HPARAMS = {
    "IRO5000":                   dict(W=100, lr=2e-3, tau=5e-4, r=8, alpha=16),
    "ORI5000":                   dict(W=100, lr=2e-4, tau=5e-2, r=256, alpha=512),
    "ROI5000":                   dict(W=100, lr=2e-3, tau=5e-4, r=8, alpha=16),
    "OIR5000":                   dict(W=100, lr=2e-4, tau=5e-4, r=256, alpha=512),
    "RIO5000":                   dict(W=100, lr=2e-3, tau=5e-2, r=8, alpha=16),
    "BPIC2015_Recurrent":        dict(W=100, lr=2e-4, tau=5e-4, r=256, alpha=512),
    "RequestForPayment":         dict(W=100, lr=2e-4, tau=5e-4, r=128, alpha=256),
    "InternationalDeclarations": dict(W=100, lr=2e-4, tau=5e-3, r=256, alpha=512),
    "DomesticDeclarations":      dict(W=100, lr=2e-4, tau=5e-4, r=256, alpha=512),
    "BPIC2017":                  dict(W=100, lr=2e-4, tau=5e-3, r=256, alpha=512),
}

# Adam steps on every window after it was tested.
EPOCHS = 10

SHARED = dict(
    gradient_steps_per_window=EPOCHS,   # Adam steps on each window after testing it
    batch_size=32,
    loss_queue_length_N=5,
    cooldown_C=10,
    energy_threshold=0.95,
    epsilon=0.05,
    lora_dropout=0.05,
    svd_batches=30,
)

