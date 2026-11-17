"""Final values of the four searched parameters per log, read by run_experiments.py.

window = W, tau = variance threshold, epsilon = CaGA threshold (1.0 = CaGA off), cooldown = C.
They start at the values of Table 1 (W, tau), epsilon 0.05 and cooldown 10. Change them here once the grid is analysed.
Learning rate, LoRA rank and alpha and everything else stay as in Setup/hparams.py (Table 1).
"""

FINAL = {
    "IRO5000": dict(window=100, tau=0.0005, epsilon=0.05, cooldown=10),
    "ORI5000": dict(window=100, tau=0.05, epsilon=0.05, cooldown=10),
    "ROI5000": dict(window=100, tau=0.0005, epsilon=0.05, cooldown=10),
    "OIR5000": dict(window=100, tau=0.0005, epsilon=0.05, cooldown=10),
    "RIO5000": dict(window=100, tau=0.05, epsilon=0.05, cooldown=10),
    "BPIC2015_Recurrent": dict(window=100, tau=0.0005, epsilon=0.05, cooldown=10),
    "RequestForPayment": dict(window=100, tau=0.0005, epsilon=0.05, cooldown=10),
    "InternationalDeclarations": dict(window=100, tau=0.005, epsilon=0.05, cooldown=10),
    "DomesticDeclarations": dict(window=100, tau=0.0005, epsilon=0.05, cooldown=10),
    "BPIC2017": dict(window=100, tau=0.005, epsilon=0.05, cooldown=10),
}
