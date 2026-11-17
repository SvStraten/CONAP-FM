DRIFT_EVAL = {
    "IRO5000": [2456, 7664, 13164, 18308, 23869, 29031, 34656, 39838, 45354],
    "ORI5000": [2325, 7526, 13276, 18426, 24280, 29448, 35083, 40271, 45941],
    "ROI5000": [2633, 7834, 13019, 18167, 23471, 28633, 33834, 39016, 44266],
    "OIR5000": [3273, 8475, 14405, 19554, 25414, 30581, 36496, 41683, 47562],
    "RIO5000": [2903, 8108, 13643, 18793, 24345, 29513, 35005, 40193, 45658],
    "BPIC2015_Recurrent": [4006, 22338, 38097, 51712, 70044, 85803, 99418],
    "RequestForPayment": [289, 3674],
    "InternationalDeclarations": [2905, 7799],
    "DomesticDeclarations": [1089, 5712],
}


def _find(table, data_name):
    clean = data_name.replace(".csv", "").replace("Data/", "").lower()
    for key in sorted(table, key=len, reverse=True):
        if key.lower() in clean:
            return list(table[key])
    return []


def lookup(data_name):
    """Drift points in evaluated stream coordinates (old API)."""
    return sorted(_find(DRIFT_EVAL, data_name))


def drift_points(data_name, mask=None, source="eval"):
    """Ground truth drift indices in the coordinates of the evaluated stream (empty list if the log has none)."""
    return sorted(_find(DRIFT_EVAL, data_name))
