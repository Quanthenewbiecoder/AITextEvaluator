# anti_ai_toolkit.py
# Heuristic AI-content detector + Supervised & Unsupervised pipelines (+ TF-IDF, Embeddings, Blending, Grouped CV)
# Large-dataset friendly: subsampled OCSVM/LOF, parallel IF/LOF, optional float32 downcast
# ---------------------------------------------------------------------------
# Install (for training):
#   pip install scikit-learn joblib numpy
# Optional (for embeddings):
#   pip install sentence-transformers
#
# Supervised (with TF-IDF + blend + grouped CV):
#   python anti_ai_toolkit.py --train human1.csv human2.csv ai.csv --dedupe --approx-dedupe --cv 5 --cv-grouped --blend
#   python anti_ai_toolkit.py --train ... --no-tfidf                # to disable tf-idf model
#   python anti_ai_toolkit.py --train ... --use-embeddings          # to add embeddings to blend
#   python anti_ai_toolkit.py --use-model anti_ai_model.joblib --check "text..." --rewrite
#
# Unsupervised (train on human-only rows: label=0):
#   python anti_ai_toolkit.py --unsup-train-humans human1.csv human2.csv \
#       --unsup-model-type ensemble --unsup-outlier-frac 0.1 --unsup-thr 90
#
# Notes:
#   - Supervised saves either a single feature model OR a blended pack with tf-idf/embeddings and weights.
#   - Predict will auto-detect and use the blended pack if present.

import re, math, csv, argparse, statistics, os, sys, random, hashlib, warnings
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_recall_curve, f1_score, precision_score, recall_score

from sklearn.svm import OneClassSVM, LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator

# Optional embeddings
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

MODEL_DEFAULT_PATH = "anti_ai_model.joblib"
UNSUP_DEFAULT_PATH = "anti_ai_unsup.joblib"

# ---------------- Lexicons ----------------
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before
being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't
down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought
our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was wasn't we we'd we'll we're we've were
weren't what what's when when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

FUNCTION_WORDS = set("""
the a an and or but if then else when where while although though because however therefore moreover
furthermore additionally indeed hence thus meanwhile whereas upon onto into out of from by for with
without within across per via among amongst despite toward towards above below before after during
between around about over under against beyond i you he she it we they me him her us them my your his
her its our their mine yours hers ours theirs is are was were be being been do does did doing done can
could may might must shall should will would this that these those here there
""".split())

CONNECTORS = [
    "however","moreover","furthermore","additionally","therefore","thus","in addition",
    "on the other hand","at the same time","overall","in summary","in conclusion",
    "meanwhile","nevertheless","nonetheless","as a result","consequently","notably"
]

AUX = {"am","is","are","was","were","be","being","been"}

ZERO_WIDTH = {
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e"
}

# ---------------- Tokenizers ----------------
def sent_tokenize(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def word_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+'[A-Za-z]+|[A-Za-z]+|\d+|[^\w\s]", text)

# ---------------- Features ----------------
def char_entropy(text: str) -> float:
    if not text: return 0.0
    counts = Counter(text); total = sum(counts.values())
    return -sum((c/total)*math.log2(c/total) for c in counts.values())

def type_token_ratio(words: List[str]) -> float:
    toks = [w.lower() for w in words if re.match(r"[A-Za-z]+$", w)]
    return len(set(toks))/len(toks) if toks else 0.0

def stopword_ratio(words: List[str]) -> float:
    toks = [w.lower() for w in words if re.match(r"[A-Za-z]+$", w)]
    if not toks: return 0.0
    return sum(1 for w in toks if w in STOPWORDS)/len(toks)

def function_word_ratio(words: List[str]) -> float:
    toks = [w.lower() for w in words if re.match(r"[A-Za-z]+$", w)]
    if not toks: return 0.0
    return sum(1 for w in toks if w in FUNCTION_WORDS)/len(toks)

def repetition_ratios(tokens: List[str]) -> Dict[str,float]:
    bigrams = list(zip(tokens, tokens[1:])); trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    out = {}
    for name, grams in (("bigram", bigrams), ("trigram", trigrams)):
        if not grams:
            out[f"repeat_{name}_ratio"] = 0.0; continue
        counts = Counter(grams); repeats = sum(c for _,c in counts.items() if c > 1)
        out[f"repeat_{name}_ratio"] = repeats/len(grams)
    return out

def punctuation_diversity(text: str) -> int:
    return len(set(re.findall(r"[^\w\s]", text)))

def connector_ratio(words: List[str]) -> float:
    lw = " ".join(w.lower() for w in words)
    hits = sum(lw.count(c) for c in CONNECTORS)
    return hits/max(1, len(words))

def passive_voice_ratio(text: str) -> float:
    w = [t.lower() for t in word_tokenize(text)]
    if not w: return 0.0
    idx = [i for i,t in enumerate(w) if t in AUX]
    sents = sent_tokenize(text); hits = 0
    for i in idx:
        window = w[i+1:i+4]
        if any(re.match(r"[a-z]+ed$", t) for t in window):
            hits += 1
    return hits/max(1, len(sents))

def sentence_length_stats(sents: List[str]) -> Tuple[float, float]:
    lengths = [len(word_tokenize(s)) for s in sents] or [0]
    mean = sum(lengths)/len(lengths)
    std = statistics.pstdev(lengths) if len(lengths)>1 else 0.0
    return mean, std

def numeric_ratio(words: List[str]) -> float:
    toks = [w for w in words if re.match(r"[A-Za-z]+|\d+", w)]
    if not toks: return 0.0
    return sum(1 for w in toks if w.isdigit())/len(toks)

def zero_width_count(text: str) -> int:
    return sum(text.count(ch) for ch in ZERO_WIDTH)

def homoglyph_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    non_basic = sum(1 for c in letters if ord(c) > 0x024F)
    return non_basic/len(letters)

def shingle_overlap_ratio(words: List[str], k: int = 8) -> float:
    toks = [w.lower() for w in words if re.match(r"[A-Za-z]+|\d+", w)]
    if len(toks) < k+1: return 0.0
    shingles = [" ".join(toks[i:i+k]) for i in range(len(toks)-k+1)]
    counts = Counter(shingles)
    repeats = sum(c for c in counts.values() if c > 1)
    return repeats/len(shingles)

# ---------------- Scoring ----------------
FEATURE_ORDER = [
    "num_sentences","num_tokens","type_token_ratio","stopword_ratio","function_word_ratio",
    "connector_ratio","passive_voice_ratio","char_entropy","mean_sentence_len","burstiness_std",
    "punctuation_diversity","numeric_ratio","zero_width_count","homoglyph_ratio",
    "repeat_bigram_ratio","repeat_trigram_ratio","repeat_shingle8_ratio"
]

def score_features(text: str) -> Dict[str, Any]:
    sents = sent_tokenize(text); words = word_tokenize(text)
    mean_len, burst = sentence_length_stats(sents)
    reps = repetition_ratios([w.lower() for w in words])

    feats = {
        "num_sentences": len(sents),
        "num_tokens": len(words),
        "type_token_ratio": round(type_token_ratio(words),4),
        "stopword_ratio": round(stopword_ratio(words),4),
        "function_word_ratio": round(function_word_ratio(words),4),
        "connector_ratio": round(connector_ratio(words),4),
        "passive_voice_ratio": round(passive_voice_ratio(text),4),
        "char_entropy": round(char_entropy(text),4),
        "mean_sentence_len": round(mean_len,2),
        "burstiness_std": round(burst,2),
        "punctuation_diversity": punctuation_diversity(text),
        "numeric_ratio": round(numeric_ratio(words),4),
        "zero_width_count": zero_width_count(text),
        "homoglyph_ratio": round(homoglyph_ratio(text),4),
        "repeat_shingle8_ratio": round(shingle_overlap_ratio(words, k=8),4),
        **{k: round(v,4) for k,v in reps.items()},
    }

    # Interpretable 0–100 heuristic — tune as needed
    score = 0.0
    score += 20.0 * max(0.0, (12.0 - min(12.0, feats["burstiness_std"])) / 12.0)
    score += 18.0 * min(1.0, feats["connector_ratio"] * 10)
    score += 12.0 * max(0.0, (feats["function_word_ratio"] - 0.45) / 0.25)
    score += 12.0 * max(0.0, (0.45 - feats["type_token_ratio"]) / 0.45)
    score += 12.0 * min(1.0, feats.get("repeat_bigram_ratio",0.0)*5)
    score += 4.0  * min(1.0, feats.get("repeat_trigram_ratio",0.0)*5)
    score += 6.0  * min(1.0, feats.get("repeat_shingle8_ratio",0.0)*5)
    if feats["zero_width_count"] > 0: score += 6.0
    score += 8.0 * min(1.0, feats["homoglyph_ratio"] * 5)
    ent = feats["char_entropy"]
    if ent < 3.5: score += 4.0 * (3.5 - ent)/3.5
    elif ent > 5.0: score += 4.0 * (ent - 5.0)/3.0

    feats["ai_likelihood_score_0_100"] = round(max(0.0, min(100.0, score)), 1)
    return feats

def classify(score: float) -> str:
    if score >= 70: return "Highly likely AI-generated"
    if score >= 50: return "Moderately likely AI-generated"
    if score >= 35: return "Uncertain / mixed"
    return "More likely human-written"

# ---------------- Data loading / dedupe ----------------
def normalize_text(s: str) -> str:
    s2 = re.sub(r"\s+", " ", s).strip().lower()
    s2 = re.sub(r"\s+([,.!?;:])", r"\1", s2)
    return s2

def shingle_hash(s: str, k: int = 8) -> str:
    words = [w for w in word_tokenize(s.lower()) if re.match(r"[A-Za-z]+|\d+", w)]
    if len(words) < k: k = max(2, len(words))
    shingles = [" ".join(words[i:i+k]) for i in range(0, max(1, len(words)-k+1))]
    h = hashlib.sha1(("||".join(shingles)).encode("utf-8")).hexdigest()
    return h

def read_csvs(paths: List[str], dedupe: bool = True, approx_dedupe: bool = False):
    texts, labels, total_rows, kept = [], [], 0, 0
    seen = set(); seen_approx = set()
    source_ids = []  # group index per row (for grouped CV / leave-one-source-out)
    for src_id, p in enumerate(paths):
        with open(p, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if "text" not in r.fieldnames or "label" not in r.fieldnames:
                raise ValueError(f"{p} must have columns: text,label")
            for row in r:
                total_rows += 1
                t = (row.get("text") or "").strip()
                lab = row.get("label")
                try:
                    lab = int(lab)
                except:
                    continue
                if not t or lab not in (0,1):
                    continue
                if dedupe:
                    key = normalize_text(t)
                    if key in seen:  # exact dedupe
                        continue
                    seen.add(key)
                if approx_dedupe:
                    h = shingle_hash(t, k=8)
                    if h in seen_approx:
                        continue
                    seen_approx.add(h)
                texts.append(t); labels.append(lab); source_ids.append(src_id)
                kept += 1
    return texts, labels, total_rows, kept, source_ids

# ---------------- Supervised Model (features) ----------------
def _vec(feats: Dict[str,Any]) -> List[float]:
    return [float(feats.get(f, 0.0)) for f in FEATURE_ORDER]

def kfold_eval(X, y, k=5, seed=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs, rocs = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = LogisticRegression(solver="lbfgs", max_iter=5000, C=0.5, class_weight="balanced")
        Xtr = [X[i] for i in tr]; ytr = [y[i] for i in tr]
        Xte = [X[i] for i in te]; yte = [y[i] for i in te]
        clf.fit(Xtr, ytr)
        yproba = clf.predict_proba(Xte)[:,1]
        yhat = (yproba >= 0.5).astype(int)
        acc = accuracy_score(yte, yhat); roc = roc_auc_score(yte, yproba)
        print(f"Fold {fold}: acc={acc:.4f}  roc_auc={roc:.4f}")
        accs.append(acc); rocs.append(roc)
    print(f"{k}-fold mean acc={sum(accs)/k:.4f}, mean roc_auc={np.mean(rocs):.4f}")
    return accs, rocs

def kfold_eval_grouped(X, y, groups, k=5):
    gkf = GroupKFold(n_splits=min(k, len(set(groups))))
    accs, rocs = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        clf = LogisticRegression(solver="lbfgs", max_iter=5000, C=0.5, class_weight="balanced")
        Xtr = [X[i] for i in tr]; ytr = [y[i] for i in tr]
        Xte = [X[i] for i in te]; yte = [y[i] for i in te]
        clf.fit(Xtr, ytr)
        yproba = clf.predict_proba(Xte)[:,1]
        yhat = (yproba >= 0.5).astype(int)
        acc = accuracy_score(yte, yhat); roc = roc_auc_score(yte, yproba)
        print(f"[Grouped] Fold {fold}: acc={acc:.4f}  roc_auc={roc:.4f}")
        accs.append(acc); rocs.append(roc)
    print(f"[Grouped] mean acc={np.mean(accs):.4f}, mean roc_auc={np.mean(rocs):.4f}")
    return accs, rocs

# ---------------- TF-IDF & Embeddings ----------------
def build_tfidf_model():
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3, max_df=0.9, sublinear_tf=True)
    base = LinearSVC(C=1.0, class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    return make_pipeline(tfidf, clf)

def embed_texts(texts: List[str], model_name: str):
    if not _HAS_ST:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
    st = SentenceTransformer(model_name)
    return st.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

def best_threshold(y_true, scores, metric="f1"):
    ps, rs, ts = precision_recall_curve(y_true, scores)
    best, best_thr = -1.0, 0.5
    for t in ts:
        yhat = (scores >= t).astype(int)
        if metric == "f1":
            m = f1_score(y_true, yhat)
        elif metric == "precision":
            m = precision_score(y_true, yhat, zero_division=0)
        else:
            m = recall_score(y_true, yhat)
        if m > best:
            best, best_thr = m, t
    return float(best_thr), float(best)

# ---------------- Train Supervised ----------------
def train_from_csv(csv_paths: List[str], save_path: str = MODEL_DEFAULT_PATH,
                   test_size: float = 0.20, no_holdout: bool = False,
                   dedupe: bool = True, approx_dedupe: bool = False, cv: int = 0,
                   seed: int = 42, cv_grouped: bool = False,
                   use_tfidf: bool = True, use_embeddings: bool = False,
                   embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                   blend: bool = True, blend_weights: str = "feat=0.5,tfidf=0.5,emb=0.0",
                   eval_leave_one: bool = False):
    texts, labels, total_rows, kept, groups = read_csvs(csv_paths, dedupe=dedupe, approx_dedupe=approx_dedupe)
    counts = Counter(labels)
    print(f"Total rows read (raw): {total_rows} | Usable after filters: {kept}")
    print(f"Class counts after filters: {dict(counts)}")
    if not texts:
        raise ValueError("No valid rows after filtering.")

    # Feature vectors
    samples = [score_features(t) for t in texts]
    X = [_vec(s) for s in samples]; y = labels
    y = np.array(y)

    # CV
    if cv and cv > 1:
        print(f"Running {cv}-fold cross-validation...")
        if cv_grouped:
            kfold_eval_grouped(X, y, groups, k=cv)
        else:
            kfold_eval(X, y, k=cv, seed=seed)
        print("CV done.\n")

    # Leave-one-source-out eval (optional)
    if eval_leave_one and len(set(groups)) > 1:
        print("Leave-one-source-out evaluation:")
        for g in sorted(set(groups)):
            tr_idx = [i for i, gg in enumerate(groups) if gg != g]
            te_idx = [i for i, gg in enumerate(groups) if gg == g]
            clf = LogisticRegression(solver="lbfgs", max_iter=5000, C=0.5, class_weight="balanced")
            clf.fit([X[i] for i in tr_idx], y[tr_idx])
            proba = clf.predict_proba([X[i] for i in te_idx])[:,1]
            roc = roc_auc_score(y[te_idx], proba)
            print(f"  Hold-out source {g}: ROC AUC={roc:.4f}")
        print()

    # Train/Test split
    Xtr, Xte, ytr, yte, tr_idx, te_idx = train_test_split(
        X, y, np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )
    print(f"Train size: {len(ytr)} | Test size: {len(yte)}")

    # Feature model
    feat_clf = LogisticRegression(solver="lbfgs", max_iter=5000, C=0.5, class_weight="balanced")
    feat_clf.fit(Xtr, ytr)
    feat_proba_tr = feat_clf.predict_proba(Xtr)[:,1]
    feat_proba_te = feat_clf.predict_proba(Xte)[:,1]

    # TF-IDF model (default ON)
    tfidf_clf = None
    tfidf_proba_tr = tfidf_proba_te = None
    if use_tfidf:
        tfidf_clf = build_tfidf_model()
        tfidf_clf.fit([texts[i] for i in tr_idx], [int(y[i]) for i in tr_idx])
        tfidf_proba_tr = tfidf_clf.predict_proba([texts[i] for i in tr_idx])[:,1]
        tfidf_proba_te = tfidf_clf.predict_proba([texts[i] for i in te_idx])[:,1]

    # Embeddings model (optional)
    emb_clf = None
    emb_proba_tr = emb_proba_te = None
    if use_embeddings:
        if not _HAS_ST:
            raise RuntimeError("Embeddings requested but sentence-transformers not installed.")
        emb_tr = embed_texts([texts[i] for i in tr_idx], embed_model)
        emb_te = embed_texts([texts[i] for i in te_idx], embed_model)
        emb_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        emb_clf.fit(emb_tr, y[tr_idx])
        emb_proba_tr = emb_clf.predict_proba(emb_tr)[:,1]
        emb_proba_te = emb_clf.predict_proba(emb_te)[:,1]

    # Evaluate & optionally blend
    def prf(ytrue, scores, name):
        roc = roc_auc_score(ytrue, scores)
        thr_f1, best_f1 = best_threshold(ytrue, scores, metric="f1")
        thr_p, best_p  = best_threshold(ytrue, scores, metric="precision")
        thr_r, best_r  = best_threshold(ytrue, scores, metric="recall")
        print(f"{name}: ROC AUC={roc:.4f} | best_F1_thr={thr_f1:.3f} F1={best_f1:.4f} | best_P_thr={thr_p:.3f} P={best_p:.4f} | best_R_thr={thr_r:.3f} R={best_r:.4f}")
        return roc

    print("Holdout metrics:")
    roc_feat = prf(yte, feat_proba_te, "Features")

    roc_tfidf = roc_emb = None
    if tfidf_proba_te is not None:
        roc_tfidf = prf(yte, tfidf_proba_te, "TF-IDF")
    if emb_proba_te is not None:
        roc_emb = prf(yte, emb_proba_te, "Embeddings")

    weights = {"feat": 1.0, "tfidf": 0.0, "emb": 0.0}
    if blend:
        # Parse weights string, e.g., "feat=0.5,tfidf=0.5,emb=0.0"
        try:
            for kv in blend_weights.split(","):
                k, v = kv.split("="); weights[k.strip()] = float(v)
        except Exception:
            print("WARNING: Could not parse --blend-weights, using defaults.")
        # zero out unavailable channels
        if not use_tfidf or tfidf_proba_te is None: weights["tfidf"] = 0.0
        if not use_embeddings or emb_proba_te is None: weights["emb"] = 0.0
        s = weights["feat"] + weights["tfidf"] + weights["emb"]
        if s <= 0:  # fallback
            weights = {"feat": 1.0, "tfidf": 0.0, "emb": 0.0}; s = 1.0
        for k in weights: weights[k] = weights[k] / s

        blend_te = weights["feat"]*feat_proba_te
        if tfidf_proba_te is not None: blend_te += weights["tfidf"]*tfidf_proba_te
        if emb_proba_te   is not None: blend_te += weights["emb"]*emb_proba_te
        roc_blend = prf(yte, blend_te, f"BLEND (feat={weights['feat']:.2f}, tfidf={weights['tfidf']:.2f}, emb={weights['emb']:.2f})")

    # Report + write classification report for feature model (baseline)
    yhat = (feat_proba_te >= 0.5).astype(int)
    report = classification_report(yte, yhat, digits=4)
    print(report)
    with open("train_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Train size: {len(ytr)} | Test size: {len(yte)}\n")
        f.write(report)

    # Save pack
    pack = {
        "feature_order": FEATURE_ORDER,
        "feature_clf": feat_clf,
        "has_tfidf": bool(tfidf_clf is not None),
        "tfidf_clf": tfidf_clf,
        "has_embeddings": bool(emb_clf is not None),
        "emb_clf": emb_clf,
        "embed_model": embed_model if emb_clf is not None else None,
        "blend": bool(blend),
        "weights": weights,
        "channels": {
            "feat": True,
            "tfidf": bool(tfidf_clf is not None),
            "emb":   bool(emb_clf is not None)
        }
    }
    joblib.dump(pack, save_path)
    print(f"Saved model to {save_path}")

def load_model(path: str = MODEL_DEFAULT_PATH):
    return joblib.load(path)

def predict(text: str, model_path: str = MODEL_DEFAULT_PATH) -> Dict[str,Any]:
    pack = load_model(model_path)
    feats = score_features(text)
    out = {
        "features": feats,
        "heuristic_score_0_100": feats["ai_likelihood_score_0_100"],
        "heuristic_label": classify(feats["ai_likelihood_score_0_100"])
    }
    try:
        # Feature channel
        feat_vec = np.array([_vec(feats)], dtype=float)
        feat_proba = pack["feature_clf"].predict_proba(feat_vec)[:,1][0]

        # TF-IDF channel
        tfidf_proba = None
        if pack.get("has_tfidf") and pack.get("tfidf_clf") is not None:
            tfidf_proba = float(pack["tfidf_clf"].predict_proba([text])[:,1][0])

        # Embeddings channel
        emb_proba = None
        if pack.get("has_embeddings") and pack.get("emb_clf") is not None:
            if not _HAS_ST:
                raise RuntimeError("Embedded model saved but sentence-transformers not installed.")
            st = SentenceTransformer(pack.get("embed_model"))
            emb = st.encode([text], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
            emb_proba = float(pack["emb_clf"].predict_proba(emb)[:,1][0])

        if pack.get("blend"):
            w = pack.get("weights", {"feat":1.0, "tfidf":0.0, "emb":0.0})
            score = w["feat"]*feat_proba + (w["tfidf"]*(tfidf_proba or 0.0)) + (w["emb"]*(emb_proba or 0.0))
            out["model_probability_ai"] = round(float(score),4)
            out["model_label"] = int(score >= 0.5)
            out["channels"] = {
                "feat": round(float(feat_proba),4),
                "tfidf": round(float(tfidf_proba),4) if tfidf_proba is not None else None,
                "emb": round(float(emb_proba),4) if emb_proba is not None else None,
                "weights": w
            }
        else:
            out["model_probability_ai"] = round(float(feat_proba),4)
            out["model_label"] = int(feat_proba >= 0.5)
    except Exception as e:
        out["model_error"] = str(e)
    return out

# ---------------- Unsupervised (human-only training) ----------------
def _to_vec(feats: dict, feature_order) -> np.ndarray:
    return np.array([float(feats.get(f, 0.0)) for f in feature_order], dtype=float)

def _standardize(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    return (X - mu) / sigma, mu, sigma

def unsup_train_humans(csv_paths: List[str],
                       feature_order=FEATURE_ORDER,
                       save_path: str = UNSUP_DEFAULT_PATH,
                       model_type: str = "ensemble",
                       outlier_frac: float = 0.1,
                       threshold_percentile: float = 90.0,
                       dedupe: bool = True, approx_dedupe: bool = False,
                       max_svm: int = 12000, max_lof: int = 20000, sample_seed: int = 42):
    texts, labels, total_rows, kept, _groups = read_csvs(csv_paths, dedupe=dedupe, approx_dedupe=approx_dedupe)
    humans = [t for t, y in zip(texts, labels) if y == 0]
    if not humans:
        raise ValueError("No human (label=0) rows found after filtering.")

    print(f"[UNSUP] Loaded {len(humans)} human texts (from {kept} usable rows; raw {total_rows}).")

    feats = [score_features(t) for t in humans]
    X = np.vstack([_to_vec(f, feature_order) for f in feats])
    Xz, mu, sigma = _standardize(X)
    Xz = Xz.astype(np.float32, copy=False)

    n_total = Xz.shape[0]
    rng = np.random.RandomState(sample_seed)

    def choose_idx(limit):
        if n_total <= limit:
            return np.arange(n_total)
        return rng.choice(n_total, size=limit, replace=False)

    models = {}
    scores = []

    # One-Class SVM (subsample)
    if model_type in ("svm","ensemble"):
        idx_svm = choose_idx(max_svm)
        ocsvm = OneClassSVM(kernel="rbf", nu=outlier_frac, gamma="scale")
        ocsvm.fit(Xz[idx_svm])
        s = ocsvm.decision_function(Xz).ravel()
        s = (s - s.min())/(s.max()-s.min()+1e-9)
        models["ocsvm"] = ocsvm
        scores.append(s)

    # Isolation Forest (full, parallel)
    if model_type in ("iforest","ensemble"):
        iforest = IsolationForest(contamination=outlier_frac, n_estimators=300, random_state=42, n_jobs=-1)
        iforest.fit(Xz)
        s = iforest.score_samples(Xz)
        s = (s - s.min())/(s.max()-s.min()+1e-9)
        models["iforest"] = iforest
        scores.append(s)

    # LOF (novelty=True on subsample; score full)
    if model_type in ("lof","ensemble"):
        idx_lof = choose_idx(max_lof)
        lof = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1)
        lof.fit(Xz[idx_lof])
        s_full = lof.score_samples(Xz).ravel()
        s_full = (s_full - s_full.min())/(s_full.max()-s_full.min()+1e-9)
        models["lof"] = {"novelty": lof, "idx": idx_lof}
        scores.append(s_full)

    # Mahalanobis (full)
    if model_type in ("mahalanobis","ensemble"):
        cov = EmpiricalCovariance().fit(Xz)
        md2 = cov.mahalanobis(Xz)
        s = -md2
        s = (s - s.min())/(s.max()-s.min()+1e-9)
        models["mahalanobis"] = {"cov": cov}
        scores.append(s)

    if not scores:
        raise ValueError("No unsupervised models selected.")
    S = np.vstack(scores).T
    ensemble_score = S.mean(axis=1)

    thr = np.percentile(ensemble_score, 100 - threshold_percentile)
    dump_obj = {
        "feature_order": feature_order,
        "mu": mu, "sigma": sigma,
        "models": models,
        "model_type": model_type,
        "threshold": float(thr),
        "calibration_percentile": threshold_percentile
    }
    joblib.dump(dump_obj, save_path)
    print(f"[UNSUP] Trained on {len(humans)} human texts. Saved to {save_path}.")
    print(f"[UNSUP] Threshold (anomaly if normality < {thr:.4f}) at {threshold_percentile}th percentile.")

def _unsup_normality_score(xz: np.ndarray, dump_obj: dict) -> float:
    models = dump_obj["models"]
    scores = []
    if "ocsvm" in models:
        s = models["ocsvm"].decision_function(xz).ravel()
        s = (s - s.min())/(s.max()-s.min()+1e-9)
        scores.append(s)
    if "iforest" in models:
        s = models["iforest"].score_samples(xz).ravel()
        s = (s - s.min())/(s.max()-s.min()+1e-9)
        scores.append(s)
    if "lof" in models:
        lof_obj = models["lof"]["novelty"]
        s = lof_obj.score_samples(xz).ravel()
        s = (s - s.min())/(s.max()-s.min()+1e-9)
        scores.append(s)
    if "mahalanobis" in models:
        cov = models["mahalanobis"]["cov"]
        md2 = cov.mahalanobis(xz)
        s = -md2
        s = (s - s.min())/(s.max()-s.min()+1e-9)
        scores.append(s)
    if not scores:
        return 0.0
    S = np.vstack(scores).T
    return float(S.mean(axis=1)[0])

def unsup_predict(text: str, model_path: str = UNSUP_DEFAULT_PATH) -> dict:
    dump_obj = joblib.load(model_path)
    feats = score_features(text)
    vec = _to_vec(feats, dump_obj["feature_order"]).reshape(1, -1)
    mu = dump_obj["mu"]; sigma = dump_obj["sigma"]
    xz = (vec - mu) / sigma
    norm = _unsup_normality_score(xz, dump_obj)
    thr = dump_obj["threshold"]
    is_anomaly = norm < thr
    return {
        "normality_0_1": round(norm, 4),
        "anomaly_label": int(is_anomaly),  # 1 = AI-like, 0 = human-like
        "threshold": round(thr, 4),
        "features": feats
    }

# ---------------- Tips & Light Rewrite ----------------
AIISH_PHRASES = {
    "in summary": "overall",
    "in conclusion": "to wrap up",
    "moreover": "also",
    "furthermore": "also",
    "additionally": "also",
    "therefore": "so",
    "however": "but",
    "thus": "so",
}

def humanization_tips(text: str, feats: Dict[str,Any]) -> List[str]:
    tips = []
    if feats["connector_ratio"] > 0.03:
        tips.append("Reduce heavy connectors (e.g., “however”, “moreover”); use simpler transitions or drop them.")
    if feats["burstiness_std"] < 6:
        tips.append("Vary sentence lengths: mix short and long sentences to create a more natural rhythm.")
    if feats["function_word_ratio"] > 0.55:
        tips.append("Trim filler/function words; prefer concrete nouns and active verbs.")
    if feats["type_token_ratio"] < 0.35:
        tips.append("Increase lexical variety: swap repeated words/phrases with precise alternatives.")
    if feats.get("repeat_bigram_ratio",0) > 0.08 or feats.get("repeat_trigram_ratio",0) > 0.04 or feats.get("repeat_shingle8_ratio",0)>0.03:
        tips.append("Remove repeated phrases; rephrase or merge similar sentences.")
    if feats["passive_voice_ratio"] > 0.25:
        tips.append("Switch passive to active voice where appropriate (e.g., “was completed by” → “completed”).")
    if feats["zero_width_count"] > 0:
        tips.append("Remove hidden/zero-width characters.")
    if feats["homoglyph_ratio"] > 0:
        tips.append("Normalize odd Unicode characters (homoglyphs) to standard Latin letters.")
    tips.append("Add small, truthful specifics (time, place, constraint) to ground the writing — avoid invented details.")
    return tips

def light_rewrite(text: str) -> str:
    if not text or not text.strip():
        return text
    original = text
    out = text
    for k, v in AIISH_PHRASES.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.IGNORECASE)
    sents = sent_tokenize(out)
    rewritten_sents = []
    for s in sents:
        tokens = word_tokenize(s)
        window = 7; keep = []; seen = set(); i = 0
        while i < len(tokens):
            j = min(i + window, len(tokens))
            sh = " ".join(tokens[i:j]).lower()
            if sh in seen and len(keep) > int(0.8 * len(tokens)):
                i += window; continue
            seen.add(sh); keep.append(tokens[i]); i += 1
        s2 = " ".join(keep)
        s2 = re.sub(r"\s+([,.!?;:])", r"\1", s2)
        s2 = re.sub(r"\bwas\b\s+([a-z]+ed)\b\s+by\b", r"\1 by", s2, flags=re.IGNORECASE)
        s2 = re.sub(r"\bwere\b\s+([a-z]+ed)\b\s+by\b", r"\1 by", s2, flags=re.IGNORECASE)
        s2 = s2.strip() or s.strip()
        rewritten_sents.append(s2)
    out = " ".join(rewritten_sents).strip()
    final_sents = []
    for s in sent_tokenize(out):
        if len(word_tokenize(s)) > 34:
            s = s.replace(" — ", ". ").replace("; ", ". ")
        final_sents.append(s.strip())
    out = " ".join(final_sents).strip()
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    if not out.endswith((".", "!", "?")): out += "."
    if len(out) < max(80, int(0.6 * len(original))):
        minimal = original
        for k, v in AIISH_PHRASES.items():
            minimal = re.sub(rf"\b{k}\b", v, minimal, flags=re.IGNORECASE)
        minimal = re.sub(r"\s+([,.!?;:])", r"\1", minimal).strip()
        if not minimal.endswith((".", "!", "?")): minimal += "."
        out = minimal
    return out

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="AI-content detector: heuristic + supervised + unsupervised (+ TF-IDF/Embeddings/Blend)")
    # supervised
    ap.add_argument("--train", nargs="+", help="Train supervised model from 1+ CSVs (columns text,label 0/1)")
    ap.add_argument("--use-model", type=str, default=MODEL_DEFAULT_PATH, help="Path to supervised model .joblib")
    ap.add_argument("--check", type=str, help="Check a raw string (supervised)")
    ap.add_argument("--check-file", type=str, help="Check a UTF-8 text file (supervised)")
    ap.add_argument("--rewrite", action="store_true", help="Also print a light rewritten version")
    ap.add_argument("--test-size", type=float, default=0.20, help="Test fraction for holdout (0.05–0.5)")
    ap.add_argument("--no-holdout", action="store_true", help="Train on all data (no test split)")
    ap.add_argument("--cv", type=int, default=0, help="Run K-fold CV (e.g., --cv 5) before holdout")
    ap.add_argument("--cv-grouped", action="store_true", help="Use GroupKFold by source file for CV")
    ap.add_argument("--eval-leave-one", action="store_true", help="Leave-one-source-out evaluation (by file)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    ap.add_argument("--dedupe", action="store_true", help="Exact/normalized dedupe across merged CSVs")
    ap.add_argument("--approx-dedupe", action="store_true", help="Approximate dedupe via shingle hash")
    # blend/extra models
    ap.add_argument("--no-tfidf", action="store_true", help="Disable TF-IDF model")
    ap.add_argument("--use-embeddings", action="store_true", help="Add sentence-transformer embeddings model")
    ap.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformer model name")
    ap.add_argument("--blend", action="store_true", help="Save blended pack (feature + tfidf [+ emb])")
    ap.add_argument("--blend-weights", type=str, default="feat=0.5,tfidf=0.5,emb=0.0", help="Blend weights, e.g., 'feat=0.4,tfidf=0.4,emb=0.2'")
    # unsupervised
    ap.add_argument("--unsup-train-humans", nargs="+", help="Train unsupervised model on 1+ CSVs (human rows only)")
    ap.add_argument("--unsup-model", type=str, default=UNSUP_DEFAULT_PATH, help="Path to unsupervised model .joblib")
    ap.add_argument("--unsup-model-type", type=str, default="ensemble",
                    choices=["svm","iforest","lof","mahalanobis","ensemble"])
    ap.add_argument("--unsup-outlier-frac", type=float, default=0.1)
    ap.add_argument("--unsup-thr", type=float, default=90.0, help="Calibration percentile (higher=stricter)")
    ap.add_argument("--unsup-check", type=str, help="Check a raw string (unsupervised)")
    ap.add_argument("--unsup-check-file", type=str, help="Check a file (unsupervised)")
    ap.add_argument("--unsup-max-svm", type=int, default=12000, help="Max rows for OCSVM training (subsample if larger)")
    ap.add_argument("--unsup-max-lof", type=int, default=20000, help="Max rows for LOF training (subsample if larger)")
    ap.add_argument("--unsup-sample-seed", type=int, default=42, help="Random seed for subsampling")

    args = ap.parse_args()

    # supervised train
    if args.train:
        train_from_csv(
            args.train,
            save_path=args.use_model,
            test_size=args.test_size,
            no_holdout=args.no_holdout,
            dedupe=args.dedupe,
            approx_dedupe=args.approx_dedupe,
            cv=args.cv,
            seed=args.seed,
            cv_grouped=args.cv_grouped,
            use_tfidf=not args.no_tfidf,
            use_embeddings=args.use_embeddings,
            embed_model=args.embed_model,
            blend=args.blend,
            blend_weights=args.blend_weights,
            eval_leave_one=args.eval_leave_one
        )
        return

    # supervised check
    if args.check is not None or args.check_file:
        text = args.check
        if args.check_file:
            if not os.path.exists(args.check_file):
                print(f"ERROR: file not found: {args.check_file}", file=sys.stderr); sys.exit(1)
            with open(args.check_file, "r", encoding="utf-8") as f:
                text = f.read()
        res = predict(text, model_path=args.use_model)
        feats = res["features"]
        print("Heuristic score:", res["heuristic_score_0_100"], "→", res["heuristic_label"])
        if "model_probability_ai" in res:
            print("Model prob (AI):", res["model_probability_ai"], "label:", res.get("model_label"))
            if "channels" in res:
                ch = res["channels"]; print("Channels:", ch)
        if args.rewrite:
            print("\nTop tips to humanize ethically:")
            for tip in humanization_tips(text, feats):
                print(" -", tip)
            print("\n--- Light rewrite (review before using) ---")
            print(light_rewrite(text))
        return

    # unsupervised train
    if args.unsup_train_humans:
        unsup_train_humans(args.unsup_train_humans,
                           save_path=args.unsup_model,
                           model_type=args.unsup_model_type,
                           outlier_frac=args.unsup_outlier_frac,
                           threshold_percentile=args.unsup_thr,
                           dedupe=args.dedupe,
                           approx_dedupe=args.approx_dedupe,
                           max_svm=args.unsup_max_svm,
                           max_lof=args.unsup_max_lof,
                           sample_seed=args.unsup_sample_seed)
        return

    # unsupervised check
    if args.unsup_check is not None or args.unsup_check_file:
        text = args.unsup_check
        if args.unsup_check_file:
            if not os.path.exists(args.unsup_check_file):
                print(f"ERROR: file not found: {args.unsup_check_file}", file=sys.stderr); sys.exit(1)
            with open(args.unsup_check_file, "r", encoding="utf-8") as f:
                text = f.read()
        res = unsup_predict(text, model_path=args.unsup_model)
        print(f"Normality score (0–1): {res['normality_0_1']}  |  Threshold: {res['threshold']}")
        print("Anomaly label:", res["anomaly_label"], " (1=AI-like, 0=human-like)")
        feats = res["features"]
        print("\nTop tips to humanize ethically:")
        for tip in humanization_tips(text, feats):
            print(" -", tip)
        return

    ap.print_help()

if __name__ == "__main__":
    main()
