"""
Input-Conditioned Cross-Layer Subspace Overlap (OLMo-2-1B)
==========================================================

The natural next step named in the paper's Limitations (Section 5.1): measure
the cross-layer overlap of the *input-conditioned effective transformation*

    M(x) = W_down @ diag(g(x)) @ W_up,   g(x) = SiLU(W_gate @ x)

rather than the static product W_down @ W_up that the main text and
experiment_b are built on. For a SwiGLU MLP with no internal norm or bias,
M(x) is exactly the linear operator the MLP applies to that input
(M(x) @ x == mlp(x)), so its top-k left singular vectors are the
input-conditioned write directions in the residual stream, and its top-k
right singular vectors the read directions. Because those vectors live in the
same d_model ambient space as the static ones, the principal-angle overlap
between layers is directly comparable to the static numbers in Figure 4 /
Table 1.

For each checkpoint this script computes BOTH:

  - static  : top-k subspace overlap of W_down @ W_up           (the reference;
              reproduces the experiment_b / Figure-4 methodology exactly via a
              full SVD), and
  - dynamic : top-k subspace overlap of M(x), averaged over sampled input
              tokens, with the per-token standard deviation retained.

so the decisive contrast (static dissolves, dynamic is maintained or rises =>
coordination has migrated into the gate) lands in a single JSON / figure,
on the same checkpoints, the same k, and the same random baseline.

All four singular-vector block relations are recorded, matching experiment_b
and Appendix D:
    UiUj : left(early)  vs left(late)    ("write vs write"; the main-text curve)
    ViVj : right(early) vs right(late)   ("read vs read")
    UiVj : left(early)  vs right(late)   ("earlier writes -> later reads")
    ViUj : right(early) vs left(late)    (its reverse)

Per token, every layer's top-k subspace is found ONCE (a randomized SVD of the
factored operator, never forming M), and overlaps for ALL layer pairs are then
read off cheaply -- so the distance-resolved view (d = 1, 3, 5, parallel to
Table 1) and the adjacent-pair curve both come from the same work.

The per-token tie is deliberate: for a given token, layer i uses layer i's own
captured MLP input and layer j uses layer j's, and their subspaces are compared
to each other; the token is the shared conditioning, not the input vector.

Checkpoints: 0, 1000, 3000, 5000, 10000, 100000, 1000000  (overlaps with the
existing gate experiments; steps <= 37000 load from the early-training repo).

Runs CPU-only; no GPU required. fp32 throughout, matching the paper's numbers.
The factored randomized SVD is the only nonstandard numerical piece; it is
validated against a full SVD by _selfcheck_factored_svd() (run with --selfcheck).

Usage:
    python experiment_gate_effective_crosslayer_olmo.py [--n_samples 64] [--selfcheck]

PATCH (disk): each checkpoint is deleted from the HF cache once its data has
been gathered (default on; use --keep_checkpoints to retain downloads). The run
is resumable: completed checkpoints are skipped on restart.
"""

import argparse
import json
import math
import os

import numpy as np

CACHE_DIR = os.environ.get("HF_HOME", None)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EARLY_TRAINING_REPO = "allenai/OLMo-2-0425-1B-early-training"
EARLY_TRAINING_MAX_STEP = 37000

MODEL_NAME = "allenai/OLMo-2-0425-1B"
D_MODEL = 2048
D_FF = 8192
N_LAYERS = 16

CHECKPOINTS = [0, 1000, 3000, 5000, 10000, 100000, 1000000]

# Subspace dimension(s). The paper's figures are top-10; widen if you want a
# robustness sweep (cost grows only with the cheap overlap step, not the SVD).
K_VALUES = [10]

# Randomized SVD settings for the factored M(x) operator. n_iter is the number
# of subspace (power) iterations; oversampling widens the sketch. On a slowly
# decaying spectrum oversampling is the cost-effective accuracy knob: oversample
# 10 -> ~0.95 top-10 subspace agreement, 30 -> >=0.999, for ~25% more time.
# Lower n_iter if your matrices' spectra decay fast (check with --selfcheck).
RANDOMIZED_SVD_N_ITER = 4
RANDOMIZED_SVD_OVERSAMPLE = 30
RANDOMIZED_SVD_SEED = 0

# Fixed seed for subsampling token positions, shared across all layers so that
# token t refers to the same source position in every layer.
SAMPLE_SEED = 42

REL_KEYS = ["UiUj", "ViVj", "UiVj", "ViUj"]

# ---------------------------------------------------------------------------
# Sample text (identical corpus to the other gate experiments)
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "The discovery of penicillin by Alexander Fleming in 1928 revolutionized medicine and ushered in the era of antibiotics. Before this breakthrough, even minor infections could prove fatal, and surgical procedures carried enormous risk. Fleming noticed that a mold called Penicillium notatum had contaminated one of his petri dishes and was killing the surrounding bacteria. This accidental observation led to one of the most important medical advances in history.",
    "In quantum mechanics, the uncertainty principle states that certain pairs of physical properties cannot both be known to arbitrary precision simultaneously. The more precisely one property is measured, the less precisely the other can be controlled. This fundamental limit is not a statement about the inadequacy of measurement instruments, but rather a reflection of the intrinsic nature of quantum systems themselves.",
    "The Amazon rainforest spans approximately 5.5 million square kilometers across nine countries in South America. It contains roughly 10 percent of all species on Earth and produces about 20 percent of the world's oxygen. The forest plays a critical role in regulating global climate patterns by absorbing carbon dioxide and releasing water vapor through transpiration.",
    "Machine learning models trained on large datasets can exhibit unexpected emergent behaviors that were not explicitly programmed or anticipated by their designers. These capabilities often appear suddenly as model scale increases, rather than improving gradually. Understanding when and why emergence occurs remains one of the central open questions in artificial intelligence research today.",
    "The construction of the Panama Canal, completed in 1914, required the excavation of over 200 million cubic yards of earth and rock. The project employed tens of thousands of workers and took over a decade to complete. Engineering challenges included managing tropical diseases, designing a lock system to handle elevation changes, and controlling the unpredictable Chagres River during flood season.",
    "Photosynthesis converts light energy into chemical energy through a series of reactions that take place in the chloroplasts of plant cells. The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH. The Calvin cycle then uses these energy carriers to fix carbon dioxide into organic molecules that the plant uses for growth and energy.",
    "The Renaissance period in European history, spanning roughly from the 14th to the 17th century, marked a profound cultural transformation characterized by renewed interest in classical antiquity. Advances in art, science, and philosophy reshaped European intellectual life. Florence, under the patronage of the Medici family, became a particularly important center of artistic and intellectual activity during this era.",
    "Neural networks process information through layers of interconnected nodes, each applying a nonlinear transformation to its inputs. The universal approximation theorem establishes that sufficiently wide networks can represent any continuous function, but says nothing about whether gradient-based training will find such representations. The gap between representational capacity and learnability remains a fundamental tension in deep learning.",
    "The human genome contains approximately 3.2 billion base pairs of DNA, encoding roughly 20,000 protein-coding genes. However, protein-coding sequences account for only about 1.5 percent of the total genome. The remaining noncoding regions, once dismissed as junk DNA, are now understood to play critical roles in gene regulation, chromosome structure, and evolutionary adaptation.",
    "Ocean currents transport vast quantities of heat around the globe, fundamentally shaping regional climates and weather patterns. The Atlantic meridional overturning circulation, for example, carries warm water northward from the tropics, giving Western Europe a milder climate than its latitude would otherwise suggest. Disruptions to these circulation patterns have been linked to abrupt climate shifts in the geological record.",
]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def step_to_revision(step: int) -> str:
    tokens_b = math.ceil(step * 2048 * 1024 / 1_000_000_000)
    return f"stage1-step{step}-tokens{tokens_b}B"


def load_model_at_checkpoint(step: int):
    """Load OLMo-2-1B at a checkpoint. Returns (model, repo_id, revision)."""
    import torch
    from transformers import AutoModelForCausalLM

    repo = EARLY_TRAINING_REPO if step <= EARLY_TRAINING_MAX_STEP else MODEL_NAME
    revision = step_to_revision(step)
    print(f"  Loading {repo} at {revision}...")

    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
    )
    model.eval()
    return model, repo, revision


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)


def _ref_matches(rev, revision):
    if revision == rev.commit_hash:
        return True
    return any(r == revision or r.endswith("/" + revision) for r in rev.refs)


def free_checkpoint(repo_id, revision, cache_dir=None):
    """Delete one downloaded checkpoint from the HF cache. Best-effort, never
    raises into the sweep. Keeps peak disk to roughly a single checkpoint."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(cache_dir=cache_dir)
    except Exception as e:
        print(f"  [cache] skip free for {repo_id}@{revision}: cannot scan cache ({e})")
        return
    commit_hashes = [
        rev.commit_hash
        for repo in info.repos if repo.repo_id == repo_id
        for rev in repo.revisions if _ref_matches(rev, revision)
    ]
    if not commit_hashes:
        print(f"  [cache] nothing to free for {repo_id}@{revision} (not in cache)")
        return
    try:
        strategy = info.delete_revisions(*commit_hashes)
        freed = strategy.expected_freed_size_str
        strategy.execute()
        print(f"  [cache] freed {freed}: deleted {repo_id}@{revision}")
    except Exception as e:
        print(f"  [cache] could not delete {repo_id}@{revision}: {e}")


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def get_sample_input_ids(tokenizer, seq_len=128):
    all_ids = []
    for text in SAMPLE_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)

    sequences = []
    for i in range(0, len(all_ids) - seq_len + 1, seq_len):
        sequences.append(all_ids[i:i + seq_len])
    if not sequences:
        sequences.append(all_ids[:seq_len])

    import torch
    return torch.tensor(sequences, dtype=torch.long)


def collect_mlp_inputs(model, input_ids, n_samples):
    """Run the model once over the sample text and capture the input to each
    layer's MLP module. Returns dict: layer_idx -> np.ndarray (n_samples, d_model),
    subsampled with a single shared index set so token t is the same source
    position in every layer."""
    import torch

    mlp_inputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            # args[0] is the literal tensor fed to mlp.forward, i.e. exactly what
            # gate_proj/up_proj see. Capturing it here makes g(x) and M(x) correct
            # regardless of where the block's norms sit.
            mlp_inputs[layer_idx] = args[0].detach().cpu()
        return hook_fn

    hooks = [model.model.layers[i].mlp.register_forward_hook(make_hook(i))
             for i in range(N_LAYERS)]
    device = next(model.parameters()).device

    accum = {i: [] for i in range(N_LAYERS)}
    with torch.no_grad():
        for b in range(input_ids.shape[0]):
            ids = input_ids[b:b + 1].to(device)
            model(ids)
            for li in range(N_LAYERS):
                accum[li].append(mlp_inputs[li].squeeze(0))  # (seq_len, d_model)

    for h in hooks:
        h.remove()

    # One shared subsample index set across layers (token counts are identical).
    total_tokens = sum(t.shape[0] for t in accum[0])
    if total_tokens > n_samples:
        rng = np.random.RandomState(SAMPLE_SEED)
        idx = rng.choice(total_tokens, n_samples, replace=False)
    else:
        idx = np.arange(total_tokens)

    out = {}
    for li in range(N_LAYERS):
        x = torch.cat(accum[li], dim=0).float().numpy()  # (total_tokens, d_model)
        out[li] = x[idx]                                  # (n_samples, d_model)
    return out


# ---------------------------------------------------------------------------
# Subspace overlap (principal angles) -- identical to experiment_b
# ---------------------------------------------------------------------------

def principal_angles_cosines(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Cosines of principal angles between two subspaces (orthonormal columns)."""
    M = U1.T @ U2
    cosines = np.linalg.svd(M, compute_uv=False)
    return np.clip(cosines, 0.0, 1.0)


def subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> dict:
    cosines = principal_angles_cosines(U1, U2)
    angles = np.arccos(np.clip(cosines, -1, 1))
    return {
        "mean_cosine": float(np.mean(cosines)),
        "max_cosine": float(np.max(cosines)),
        "min_cosine": float(np.min(cosines)),
        "median_cosine": float(np.median(cosines)),
        "grassmann_distance": float(np.sqrt(np.sum(angles ** 2))),
        "overlap_frac_gt_0.5": float(np.mean(cosines > 0.5)),
        "n_angles": int(len(cosines)),
    }


def mean_principal_cosine(U1: np.ndarray, U2: np.ndarray) -> float:
    """Just the headline scalar, for the per-token dynamic accumulation."""
    return float(np.mean(principal_angles_cosines(U1, U2)))


def random_subspace_baseline(d: int, k: int, n_trials: int = 200) -> dict:
    """Expected overlap of two random k-dim subspaces of R^d. Applies to both
    the static and dynamic measurements (both live in R^d_model)."""
    vals = []
    for _ in range(n_trials):
        U1 = np.linalg.qr(np.random.randn(d, k))[0]
        U2 = np.linalg.qr(np.random.randn(d, k))[0]
        vals.append(float(np.mean(principal_angles_cosines(U1, U2))))
    return {"mean_cosine_mean": float(np.mean(vals)),
            "mean_cosine_std": float(np.std(vals))}


# ---------------------------------------------------------------------------
# Top-k SVD: static (explicit) and dynamic (factored, never forms M)
# ---------------------------------------------------------------------------

def _silu_np(z: np.ndarray) -> np.ndarray:
    return z / (1.0 + np.exp(-z))


def static_topk(W_down: np.ndarray, W_up: np.ndarray, k: int):
    """Top-k left (U) and right (V) singular vectors of the static W_down @ W_up,
    via a full SVD -- matching experiment_b exactly."""
    composed = W_down @ W_up                       # (d_model, d_model)
    U, _, Vt = np.linalg.svd(composed, full_matrices=True)
    return U[:, :k], Vt[:k, :].T


def randomized_topk_factored(W_down: np.ndarray, g: np.ndarray, W_up: np.ndarray,
                             k: int, n_iter: int = None,
                             oversample: int = None,
                             seed: int = RANDOMIZED_SVD_SEED):
    """Top-k left (U) and right (V) singular vectors of

        M = W_down @ diag(g) @ W_up

    WITHOUT forming M. Randomized range finding (Halko, Martinsson & Tropp,
    Alg. 4.4 + 5.1) on the factored operator: M and M^T are applied through the
    three thin matrices, so each application is O(d_model * d_ff * l) rather than
    the O(d_model^2 * d_ff) cost of materializing M.

        W_down: (d_model, d_ff)   W_up: (d_ff, d_model)   g: (d_ff,)
    Returns U (d_model, k), V (d_model, k), with orthonormal columns.
    """
    if n_iter is None:
        n_iter = RANDOMIZED_SVD_N_ITER
    if oversample is None:
        oversample = RANDOMIZED_SVD_OVERSAMPLE
    d_model = W_down.shape[0]
    l = min(k + oversample, d_model)
    rng = np.random.RandomState(seed)

    g_col = g[:, None]  # (d_ff, 1), broadcasts over the l columns

    def M_apply(X):       # M @ X : (d_model, l) -> (d_model, l)
        return W_down @ (g_col * (W_up @ X))

    def MT_apply(Y):      # M^T @ Y : (d_model, l) -> (d_model, l)
        return W_up.T @ (g_col * (W_down.T @ Y))

    Q, _ = np.linalg.qr(M_apply(rng.randn(d_model, l)))
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(MT_apply(Q))
        Q, _ = np.linalg.qr(M_apply(Q))

    B = MT_apply(Q).T                              # Q^T M, shape (l, d_model)
    Ub, _, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub[:, :k]                              # (d_model, k)
    V = Vt[:k, :].T                                # (d_model, k)
    return U, V


# ---------------------------------------------------------------------------
# Per-pair relation helpers
# ---------------------------------------------------------------------------

def all_pairs(n_layers):
    return [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)]


def static_relations(U, V, i, j, k):
    """Four full overlap dicts between layers i (earlier) and j (later)."""
    return {
        "UiUj": subspace_overlap(U[i][:, :k], U[j][:, :k]),
        "ViVj": subspace_overlap(V[i][:, :k], V[j][:, :k]),
        "UiVj": subspace_overlap(U[i][:, :k], V[j][:, :k]),
        "ViUj": subspace_overlap(V[i][:, :k], U[j][:, :k]),
    }


def dynamic_relation_cosines(U, V, i, j, k):
    """Four headline mean-cosines between layers i and j for one token."""
    return {
        "UiUj": mean_principal_cosine(U[i][:, :k], U[j][:, :k]),
        "ViVj": mean_principal_cosine(V[i][:, :k], V[j][:, :k]),
        "UiVj": mean_principal_cosine(U[i][:, :k], V[j][:, :k]),
        "ViUj": mean_principal_cosine(V[i][:, :k], U[j][:, :k]),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def compute_checkpoint(model, mlp_inputs):
    """Static + dynamic overlap for every layer pair at one checkpoint."""
    import torch

    # Pull the three MLP weight matrices per layer once, as numpy fp32.
    W = []
    for li in range(N_LAYERS):
        mlp = model.model.layers[li].mlp
        W.append({
            "gate": mlp.gate_proj.weight.detach().float().cpu().numpy(),  # (d_ff, d_model)
            "up": mlp.up_proj.weight.detach().float().cpu().numpy(),      # (d_ff, d_model)
            "down": mlp.down_proj.weight.detach().float().cpu().numpy(),  # (d_model, d_ff)
        })

    k_max = max(K_VALUES)
    pairs = all_pairs(N_LAYERS)

    # ---- static reference ----
    sU = {}
    sV = {}
    for li in range(N_LAYERS):
        sU[li], sV[li] = static_topk(W[li]["down"], W[li]["up"], k_max)

    static_pairs = {}
    for (i, j) in pairs:
        static_pairs[(i, j)] = {f"top{k}": static_relations(sU, sV, i, j, k)
                                for k in K_VALUES}

    # ---- dynamic, averaged over tokens ----
    n_samples = mlp_inputs[0].shape[0]
    # running sums / sums-of-squares of the mean-cosine, per pair / k / relation
    acc_sum = {(i, j): {k: {r: 0.0 for r in REL_KEYS} for k in K_VALUES} for (i, j) in pairs}
    acc_sq = {(i, j): {k: {r: 0.0 for r in REL_KEYS} for k in K_VALUES} for (i, j) in pairs}

    for t in range(n_samples):
        dU = {}
        dV = {}
        for li in range(N_LAYERS):
            x = mlp_inputs[li][t]                          # (d_model,)
            g = _silu_np(W[li]["gate"] @ x)                # (d_ff,)
            dU[li], dV[li] = randomized_topk_factored(
                W[li]["down"], g, W[li]["up"], k_max)
        for (i, j) in pairs:
            for k in K_VALUES:
                rels = dynamic_relation_cosines(dU, dV, i, j, k)
                for r in REL_KEYS:
                    v = rels[r]
                    acc_sum[(i, j)][k][r] += v
                    acc_sq[(i, j)][k][r] += v * v
        if (t + 1) % 16 == 0 or t == n_samples - 1:
            print(f"    dynamic: {t + 1}/{n_samples} tokens")

    dynamic_pairs = {}
    for (i, j) in pairs:
        per_k = {}
        for k in K_VALUES:
            rel_stats = {}
            for r in REL_KEYS:
                m = acc_sum[(i, j)][k][r] / n_samples
                var = max(acc_sq[(i, j)][k][r] / n_samples - m * m, 0.0)
                rel_stats[r] = {"mean": float(m), "std": float(math.sqrt(var))}
            per_k[f"top{k}"] = rel_stats
        dynamic_pairs[(i, j)] = per_k

    # ---- assemble ----
    out = {"pairs": {}}
    for (i, j) in pairs:
        out["pairs"][f"{i}_{j}"] = {
            "distance": j - i,
            "static": static_pairs[(i, j)],
            "dynamic": dynamic_pairs[(i, j)],
        }
    return out


def run_experiment(n_samples, output_dir, checkpoints, free_checkpoints=True):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "gate_effective_crosslayer_olmo_1b.json")

    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Loaded existing results ({len(results['checkpoints'])} checkpoints)")
    else:
        print("Computing random subspace baselines...")
        baselines = {str(k): random_subspace_baseline(D_MODEL, k) for k in K_VALUES}
        for k in K_VALUES:
            b = baselines[str(k)]
            print(f"  k={k}: random mean_cosine = "
                  f"{b['mean_cosine_mean']:.4f} +/- {b['mean_cosine_std']:.4f}")
        results = {
            "model": "1b",
            "model_name": MODEL_NAME,
            "model_family": "olmo",
            "d_model": D_MODEL, "d_ff": D_FF, "n_layers": N_LAYERS,
            "k_values": K_VALUES,
            "n_samples": n_samples,
            "randomized_svd": {"n_iter": RANDOMIZED_SVD_N_ITER,
                               "oversample": RANDOMIZED_SVD_OVERSAMPLE},
            "random_baselines": baselines,
            "checkpoints": {},
        }

    tokenizer = get_tokenizer()
    input_ids = get_sample_input_ids(tokenizer)
    print(f"Input: {input_ids.shape[0]} sequences of length {input_ids.shape[1]}")

    for step in checkpoints:
        if str(step) in results["checkpoints"]:
            print(f"\n  Skipping step {step} (already computed)")
            continue

        print(f"\n{'=' * 60}\n  Step {step}\n{'=' * 60}")
        model, repo, revision = load_model_at_checkpoint(step)
        mlp_inputs = collect_mlp_inputs(model, input_ids, n_samples)
        step_results = compute_checkpoint(model, mlp_inputs)

        # quick console summary: adjacent-pair mean UiUj, static vs dynamic
        k = K_VALUES[0]
        adj = [(p["static"][f"top{k}"]["UiUj"]["mean_cosine"],
                p["dynamic"][f"top{k}"]["UiUj"]["mean"])
               for p in step_results["pairs"].values() if p["distance"] == 1]
        s_mean = float(np.mean([a for a, _ in adj]))
        d_mean = float(np.mean([b for _, b in adj]))
        step_results["summary"] = {
            "adjacent_static_UiUj_mean": s_mean,
            "adjacent_dynamic_UiUj_mean": d_mean,
        }
        print(f"  adjacent top{k} UiUj:  static={s_mean:.4f}   "
              f"dynamic(M(x))={d_mean:.4f}   "
              f"(random={results['random_baselines'][str(k)]['mean_cosine_mean']:.4f})")

        results["checkpoints"][str(step)] = step_results

        del model, mlp_inputs
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if free_checkpoints:
            free_checkpoint(repo, revision, CACHE_DIR)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results to {out_path}")

    print(f"\nDone. Results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _adjacent_means(results, kind, k):
    """Mean over adjacent pairs of UiUj, per checkpoint, for 'static' or 'dynamic'."""
    steps = sorted(int(s) for s in results["checkpoints"].keys())
    means, stds = [], []
    for step in steps:
        pairs = results["checkpoints"][str(step)]["pairs"]
        vals = []
        for p in pairs.values():
            if p["distance"] != 1:
                continue
            if kind == "static":
                vals.append(p["static"][f"top{k}"]["UiUj"]["mean_cosine"])
            else:
                vals.append(p["dynamic"][f"top{k}"]["UiUj"]["mean"])
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return steps, np.array(means), np.array(stds)


def plot_handoff(results, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping handoff plot.")
        return
    k = results["k_values"][0]
    steps, s_mean, _ = _adjacent_means(results, "static", k)
    _, d_mean, d_std = _adjacent_means(results, "dynamic", k)
    bl = results["random_baselines"][str(k)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, s_mean, "o-", color="tab:red",
            label=r"static  $W_{down}W_{up}$")
    ax.plot(steps, d_mean, "o-", color="tab:blue",
            label=r"input-conditioned  $W_{down}\,\mathrm{diag}(g(x))\,W_{up}$")
    ax.fill_between(steps, d_mean - d_std, d_mean + d_std, color="tab:blue", alpha=0.15)
    ax.axhline(bl["mean_cosine_mean"], color="gray", ls="--", alpha=0.7, label="random")
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Mean cosine (top-{k}, adjacent layers)")
    ax.set_title("Static vs input-conditioned cross-layer alignment (OLMo-2-1B)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "gate_effective_handoff_olmo_1b.png")
    plt.savefig(path, dpi=150)
    print(f"Handoff plot saved to {path}")
    plt.close()


def plot_distance_resolved(results, output_dir):
    """At the final checkpoint: overlap vs layer distance, static vs dynamic
    (parallel to the d = 1, 3, 5 columns of Table 1)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping distance plot.")
        return
    k = results["k_values"][0]
    final = str(sorted(int(s) for s in results["checkpoints"].keys())[-1])
    pairs = results["checkpoints"][final]["pairs"]

    by_dist_s, by_dist_d = {}, {}
    for p in pairs.values():
        d = p["distance"]
        by_dist_s.setdefault(d, []).append(p["static"][f"top{k}"]["UiUj"]["mean_cosine"])
        by_dist_d.setdefault(d, []).append(p["dynamic"][f"top{k}"]["UiUj"]["mean"])
    dists = sorted(by_dist_s.keys())
    s = [np.mean(by_dist_s[d]) for d in dists]
    dy = [np.mean(by_dist_d[d]) for d in dists]
    bl = results["random_baselines"][str(k)]["mean_cosine_mean"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dists, s, "o-", color="tab:red", label="static")
    ax.plot(dists, dy, "o-", color="tab:blue", label="input-conditioned")
    ax.axhline(bl, color="gray", ls="--", alpha=0.7, label="random")
    ax.set_xlabel("Layer distance d")
    ax.set_ylabel(f"Mean cosine (top-{k}, UiUj)")
    ax.set_title(f"Overlap vs distance at step {final} (OLMo-2-1B)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "gate_effective_distance_olmo_1b.png")
    plt.savefig(path, dpi=150)
    print(f"Distance plot saved to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Self-check for the factored randomized SVD
# ---------------------------------------------------------------------------

def _selfcheck_factored_svd(seed=1):
    """Validate randomized_topk_factored two ways:

    (1) Operator equivalence: the factored applies must equal an explicit M @ X
        and M^T @ Y to machine precision. This is the real correctness test for
        the gate scaling and the transpose; it is spectrum-independent.
    (2) Subspace recovery: on a matrix with a well-defined (decaying) top-10
        spectrum, the factored top-10 subspace must match a full SVD's to >0.999.
        (A near-flat spectrum makes the top-10 subspace genuinely ambiguous, so
        it is not a fair correctness target and is not used here.)
    """
    rng = np.random.RandomState(seed)
    k = 10

    # (1) operator equivalence, small scale
    d_model, d_ff = 256, 1024
    W_gate = rng.randn(d_ff, d_model) / math.sqrt(d_model)
    W_up = rng.randn(d_ff, d_model) / math.sqrt(d_model)
    W_down = rng.randn(d_model, d_ff) / math.sqrt(d_ff)
    x = rng.randn(d_model)
    g = _silu_np(W_gate @ x)
    M = W_down @ (g[:, None] * W_up)
    X = rng.randn(d_model, k + RANDOMIZED_SVD_OVERSAMPLE)
    gc = g[:, None]
    err_M = np.abs((W_down @ (gc * (W_up @ X))) - M @ X).max()
    err_MT = np.abs((W_up.T @ (gc * (W_down.T @ X))) - M.T @ X).max()
    op_ok = (err_M < 1e-9) and (err_MT < 1e-9)
    print(f"[selfcheck] (1) operator equivalence: ||M_apply-M@X||={err_M:.1e} "
          f"||MT_apply-M^T@Y||={err_MT:.1e}  -> {'ok' if op_ok else 'FAIL'}")

    # (2) subspace recovery on a decaying spectrum, full residual-stream scale
    dm, dff = D_MODEL, D_FF
    Uo, _ = np.linalg.qr(rng.randn(dm, dm))
    Vo, _ = np.linalg.qr(rng.randn(dm, dm))
    A, _ = np.linalg.qr(rng.randn(dff, dm))
    sig = np.exp(-np.arange(dm) / 30.0)
    Wd = (Uo * np.sqrt(sig)) @ A.T
    Wu = A @ (np.sqrt(sig)[:, None] * Vo.T)
    gg = _silu_np(rng.randn(dff))
    Mn = Wd @ (gg[:, None] * Wu)
    Uf, _, Vtf = np.linalg.svd(Mn)
    U_f, V_f = randomized_topk_factored(Wd, gg, Wu, k)
    cu = principal_angles_cosines(Uf[:, :k], U_f).min()
    cv = principal_angles_cosines(Vtf.T[:, :k], V_f).min()
    sub_ok = (cu > 0.999) and (cv > 0.999)
    print(f"[selfcheck] (2) subspace recovery (decaying spectrum, {dm}x{dff}): "
          f"min cos U={cu:.5f} V={cv:.5f}  -> {'ok' if sub_ok else 'FAIL'}  "
          f"(n_iter={RANDOMIZED_SVD_N_ITER}, oversample={RANDOMIZED_SVD_OVERSAMPLE})")

    ok = op_ok and sub_ok
    print(f"[selfcheck] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Input-conditioned cross-layer subspace overlap (OLMo-2-1B)")
    parser.add_argument("--n_samples", type=int, default=32,
                        help="Token positions sampled per checkpoint for M(x). The "
                             "token-averaged mean is usually stable by ~32; raise "
                             "for tighter error bands (cost scales linearly).")
    parser.add_argument("--output_dir", default="results/gate_effective_crosslayer")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=None,
                        help="Override the checkpoint steps to process.")
    parser.add_argument("--svd_n_iter", type=int, default=None,
                        help="Subspace iterations for the factored SVD "
                             f"(default {RANDOMIZED_SVD_N_ITER}).")
    parser.add_argument("--svd_oversample", type=int, default=None,
                        help="Oversampling for the factored SVD "
                             f"(default {RANDOMIZED_SVD_OVERSAMPLE}).")
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Keep downloads in the HF cache (default: delete each "
                             "checkpoint once its data has been gathered).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="Validate the factored SVD against a full SVD and exit.")
    args = parser.parse_args()

    if args.svd_n_iter is not None:
        RANDOMIZED_SVD_N_ITER = args.svd_n_iter
    if args.svd_oversample is not None:
        RANDOMIZED_SVD_OVERSAMPLE = args.svd_oversample

    if args.selfcheck:
        _selfcheck_factored_svd()
        raise SystemExit(0)

    checkpoints = args.checkpoints if args.checkpoints is not None else CHECKPOINTS
    results = run_experiment(args.n_samples, args.output_dir, checkpoints,
                             free_checkpoints=not args.keep_checkpoints)
    plot_handoff(results, args.output_dir)
    plot_distance_resolved(results, args.output_dir)
