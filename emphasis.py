import numpy as np

def compute_emphasis(words, pitches, threshold):
    ratios = []
    for i in range(len(words)-1):
        if pitches[i] == 0:
            ratio = 100.0
        else:
            ratio = (pitches[i+1] / pitches[i]) * 100
        ratios.append(ratio)

    candidates = [i for i, r in enumerate(ratios) if r < threshold]

    if candidates:
        best_idx = candidates[np.argmin([ratios[i] for i in candidates])]
        emphasized_words = words.copy()
        emphasized_words[best_idx] = f"*{emphasized_words[best_idx]}*"
    else:
        emphasized_words = words.copy()

    return " ".join(emphasized_words), ratios


def compute_emphasis_all_thresholds(words, pitches, thresholds):
    results = []
    for t in thresholds:
        text, ratios = compute_emphasis(words, pitches, t)
        results.append({
            "threshold": round(t, 3),
            "text": text,
            "ratios": [round(r, 2) for r in ratios]
        })
    return results
