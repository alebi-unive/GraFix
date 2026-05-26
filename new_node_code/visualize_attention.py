"""
Visualize saved attention matrices from NodeClassification_CPU.py.

Loads .npz files produced by --save-pre-norm-attention, --save-attention or
--save-gate-modulation and plots:
  • sorted heatmap (always, --out)
  • class-aggregated C×C matrix  (--extra-plots class-agg)
  • symmetry scatter A(i,j) vs A(j,i)  (--extra-plots symmetry-scatter)
  • per-node entropy distribution  (--extra-plots entropy)
  • same-class vs cross-class violin  (--extra-plots same-cross-class)

Use --extra-plots all to generate all four additional plots.

Usage examples
--------------
# Compare symmetric vs asymmetric + all extra diagnostic plots:
python visualize_attention.py \\
    --npz symmetric.npz asymmetric.npz \\
    --dataset Cora \\
    --matrix-key pre_norm_attention_head_0 \\
    --titles "Cora symmetric" "Cora asymmetric" \\
    --out cora_compare.png \\
    --extra-plots all
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from torch_geometric import datasets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_EXTRA = ['class-agg', 'symmetry-scatter', 'entropy', 'same-cross-class']

_RNG = np.random.default_rng(0)


def load_labels(dataset_name: str) -> np.ndarray:
    """Return integer class labels for every node in the dataset."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    if dataset_name in ('Cora', 'CiteSeer', 'PubMed'):
        ds = datasets.Planetoid(root=data_path, name=dataset_name)
    elif dataset_name in ('Cornell', 'Texas', 'Wisconsin'):
        ds = datasets.WebKB(root=data_path, name=dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return ds[0].y.numpy()


def sort_order(labels: np.ndarray) -> np.ndarray:
    """Return index permutation that sorts nodes by class label."""
    return np.argsort(labels, kind='stable')


def load_matrix(npz_path: str, key: str) -> np.ndarray:
    """Load a 2-D matrix from a compressed .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    if key not in data:
        available = [k for k in data.files if not k.startswith('_')]
        raise KeyError(
            f"Key '{key}' not found in {npz_path}.\n"
            f"Available keys: {available}"
        )
    mat = data[key]
    if mat.ndim != 2:
        raise ValueError(
            f"Expected a 2-D matrix under key '{key}', got shape {mat.shape}."
        )
    return mat.astype(np.float32)


def row_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-normalize a matrix so each row sums to 1 (for entropy)."""
    row_sums = mat.sum(axis=1, keepdims=True)
    return mat / (row_sums + 1e-9)


# ---------------------------------------------------------------------------
# Plot: sorted heatmap
# ---------------------------------------------------------------------------

def plot_single(ax, matrix: np.ndarray, order: np.ndarray, title: str, vmax=None):
    sorted_mat = matrix[np.ix_(order, order)]
    if vmax is None:
        vmax = float(np.nanpercentile(sorted_mat, 99))
    im = ax.imshow(sorted_mat, aspect='auto', cmap='viridis',
                   vmin=0.0, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Destination node sorted by true label')
    ax.set_ylabel('Source node sorted by true label')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Extra plot: C×C class-aggregated attention
# ---------------------------------------------------------------------------

def class_aggregated_matrix(mat: np.ndarray, labels: np.ndarray, mode: str = 'mean') -> np.ndarray:
    """
    For each (class_src, class_dst) pair compute the mean attention value.
    Returns a (C, C) matrix.
    """
    classes = np.unique(labels)
    C = len(classes)
    agg = np.zeros((C, C), dtype=np.float64)
    for i, ci in enumerate(classes):
        src_idx = np.where(labels == ci)[0]
        for j, cj in enumerate(classes):
            dst_idx = np.where(labels == cj)[0]
            sel = mat[np.ix_(src_idx, dst_idx)]
            if mode in ('sum', 'row_norm'):
                agg[i, j] = sel.sum()
            else:
                agg[i, j] = sel.mean()
    if mode == 'row_norm':
        row_sums = agg.sum(axis=1, keepdims=True)
        agg = agg / row_sums
    return agg.astype(np.float32)


def plot_class_agg_mean(out_path: str, matrices: list, labels: np.ndarray,
                   titles: list, dpi: int):
    """Save a C×C class-aggregated attention heatmap for each model."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    classes = np.unique(labels)
    class_ticks = [str(c) for c in classes]

    agg_mats = [class_aggregated_matrix(m, labels) for m in matrices]
    use_shared_vmax = False
    shared_vmax = max(float(np.nanpercentile(a, 99)) for a in agg_mats) if use_shared_vmax else None

    for i, (agg, title) in enumerate(zip(agg_mats, titles)):
        ax = axes[0][i]
        max_value = float(agg.max()) if not use_shared_vmax else shared_vmax
        im = ax.imshow(agg, aspect='auto', cmap='viridis',
                       vmin=0.0, vmax=max_value, interpolation='nearest')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(class_ticks)
        ax.set_yticklabels(class_ticks)
        ax.set_xlabel('Destination class')
        ax.set_ylabel('Source class')
        ax.set_title(f'{title}\n(class-aggregated mean attention)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells with values
        for r in range(len(classes)):
            for c in range(len(classes)):
                ax.text(c, r, f'{agg[r, c]:.3g}',
                        ha='center', va='center', fontsize=7,
                        color='white' if agg[r, c] < max_value * 0.6 else 'black')

    fig.suptitle('Class-aggregated attention', y=1.01)
    plt.tight_layout()
    _save(fig, out_path, dpi)


def plot_class_agg_sum(out_path: str, matrices: list, labels: np.ndarray,
                   titles: list, dpi: int):
    """Save a C×C row-normalised attention heatmap for each model.

    Each cell agg[i, j] is the fraction of total attention emitted by
    class-i nodes that lands on class-j nodes.  Rows sum to 1, so the
    diagonal can be read directly as a homophily score.
    """
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    classes = np.unique(labels)
    class_ticks = [str(c) for c in classes]

    agg_mats = [class_aggregated_matrix(m, labels, 'row_norm') for m in matrices]

    for i, (agg, title) in enumerate(zip(agg_mats, titles)):
        ax = axes[0][i]
        max_value = float(agg.max())
        im = ax.imshow(agg, aspect='auto', cmap='viridis',
                       vmin=0.0, vmax=max_value, interpolation='nearest')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(class_ticks)
        ax.set_yticklabels(class_ticks)
        ax.set_xlabel('Destination class')
        ax.set_ylabel('Source class')
        ax.set_title(f'{title}\n(row-normalised: fraction of attention per source class)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells with percentage
        for r in range(len(classes)):
            for c in range(len(classes)):
                ax.text(c, r, f'{agg[r, c]:.3g}',
                        ha='center', va='center', fontsize=7,
                        color='white' if agg[r, c] < max_value * 0.6 else 'black')

    fig.suptitle('Row-normalised class attention\n'
                 '(each row = fraction of total attention sent to each class; diagonal = homophily)',
                 y=1.03)
    plt.tight_layout()
    _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# Extra plot: symmetry scatter A(i,j) vs A(j,i)
# ---------------------------------------------------------------------------

def plot_symmetry_scatter(out_path: str, matrices: list, titles: list,
                          dpi: int, n_sample: int = 50_000):
    """
    Scatter A(i,j) vs A(j,i) for a random sample of off-diagonal pairs.
    A perfectly symmetric gate lies exactly on the diagonal y = x.
    Asymmetric gates spread away from the diagonal.
    """
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for i, (mat, title) in enumerate(zip(matrices, titles)):
        N = mat.shape[0]
        # Sample random off-diagonal (src, dst) pairs with src < dst
        max_pairs = N * (N - 1) // 2
        k = min(n_sample, max_pairs)
        src = _RNG.integers(0, N, size=k * 3)
        dst = _RNG.integers(0, N, size=k * 3)
        mask = src != dst
        src, dst = src[mask][:k], dst[mask][:k]

        a_ij = mat[src, dst]
        a_ji = mat[dst, src]

        # Compute symmetry deviation: mean |A(i,j) - A(j,i)|
        sym_dev = float(np.mean(np.abs(a_ij - a_ji)))

        ax = axes[0][i]
        ax.scatter(a_ij, a_ji, s=1, alpha=0.2, rasterized=True)

        lim = max(float(a_ij.max()), float(a_ji.max())) * 1.05
        ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='y = x (perfect symmetry)')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel('A(i, j)')
        ax.set_ylabel('A(j, i)')
        ax.set_title(f'{title}\nMean |A(i,j)−A(j,i)| = {sym_dev:.4g}')
        ax.legend(fontsize=8)
        ax.set_aspect('equal')

    fig.suptitle('Symmetry scatter: symmetric gate → all points on y = x', y=1.01)
    plt.tight_layout()
    _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# Extra plot: per-node entropy
# ---------------------------------------------------------------------------

def node_entropy(mat: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy for each row (after row-normalization)."""
    p = row_normalize(mat)
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(axis=1)


def plot_entropy(out_path: str, matrices: list, titles: list, dpi: int):
    """
    Histogram of per-node attention entropy.
    Lower entropy = more focused/selective attention.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for mat, title in zip(matrices, titles):
        ent = node_entropy(mat)
        mean_ent = float(ent.mean())
        ax.hist(ent, bins=60, alpha=0.6, density=True,
                label=f'{title}  (mean={mean_ent:.2g})')

    ax.set_xlabel('Per-node entropy H(i)')
    ax.set_ylabel('Density')
    ax.set_title('Attention entropy per node\n(lower = more selective attention)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# Extra plot: same-class vs cross-class attention violin
# ---------------------------------------------------------------------------

def plot_same_cross_class(out_path: str, matrices: list, labels: np.ndarray,
                          titles: list, dpi: int, n_sample: int = 100_000):
    """
    Violin plot comparing attention values for same-class vs cross-class pairs.
    A good gate should assign higher values to same-class pairs (homophily).
    """
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for i, (mat, title) in enumerate(zip(matrices, titles)):
        N = mat.shape[0]
        k = min(n_sample, N * N)
        src = _RNG.integers(0, N, size=k)
        dst = _RNG.integers(0, N, size=k)

        same_mask = labels[src] == labels[dst]
        vals_same = mat[src[same_mask], dst[same_mask]]
        vals_cross = mat[src[~same_mask], dst[~same_mask]]

        ax = axes[0][i]
        parts = ax.violinplot(
            [vals_same, vals_cross],
            positions=[0, 1],
            showmedians=True,
            showextrema=False,
        )
        # Color the two violins differently
        colors = ['#2196F3', '#FF5722']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        mean_same = float(vals_same.mean())
        mean_cross = float(vals_cross.mean())
        ratio = mean_same / (mean_cross + 1e-9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            f'Same class\n(mean={mean_same:.4g})',
            f'Cross class\n(mean={mean_cross:.4g})',
        ])
        ax.set_ylabel('Attention value A(i, j)')
        ax.set_title(f'{title}\nSame/Cross ratio = {ratio:.2g}')

    fig.suptitle('Same-class vs cross-class attention distribution\n'
                 '(higher same/cross ratio = gate suppresses inter-class attention)',
                 y=1.02)
    plt.tight_layout()
    _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save(fig, path: str, dpi: int):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    fig.savefig(f'{path[:-4]}.eps', bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


def _extra_path(base_out: str, suffix: str) -> str:
    """Derive an output path by inserting a suffix before the extension."""
    stem, ext = os.path.splitext(base_out)
    return f"{stem}_{suffix}{ext}"


def _iter_subsets(n: int):
    """Yield (indices, suffix) for each single panel and each pair."""
    for i in range(n):
        yield [i], f'single_{i}'
    for i in range(n):
        for j in range(i + 1, n):
            yield [i, j], f'pair_{i}_{j}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualize GraFix attention matrices sorted by class label.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--npz', nargs='+', required=True,
        help='Path(s) to .npz file(s) produced by NodeClassification_CPU.py.',
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['Cora', 'CiteSeer', 'PubMed', 'Cornell', 'Texas', 'Wisconsin'],
        help='Dataset name (needed to load true labels for sorting).',
    )
    parser.add_argument(
        '--matrix-key', type=str, default='pre_norm_attention_head_0',
        help='Key inside the .npz file to plot. '
             'Typical values: pre_norm_attention_head_0, '
             'attention_head_0, gate_modulation_head_0.',
    )
    parser.add_argument(
        '--title', type=str, default=None,
        help='Title for a single-panel plot (ignored when multiple .npz are given).',
    )
    parser.add_argument(
        '--titles', nargs='+', default=None,
        help='Titles for each panel when multiple .npz files are given.',
    )
    parser.add_argument(
        '--out', type=str, default='attention_heatmap.png',
        help='Output image file path for the sorted heatmap.',
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='DPI for all saved figures.',
    )
    parser.add_argument(
        '--vmax', type=float, default=None,
        help='Shared colour-scale maximum for the heatmap. '
             'Defaults to 99th percentile.',
    )
    parser.add_argument(
        '--extra-plots', nargs='+', default=[],
        metavar='PLOT',
        help=(
            'Additional diagnostic plots to generate alongside the heatmap. '
            'Choices: class-agg, symmetry-scatter, entropy, same-cross-class, all. '
            'Output files are derived from --out by adding a suffix. '
            'Example: --extra-plots all'
        ),
    )
    args = parser.parse_args()

    # Resolve 'all' shorthand
    extra = set(args.extra_plots)
    if 'all' in extra:
        extra = set(_ALL_EXTRA)
    unknown = extra - set(_ALL_EXTRA)
    if unknown:
        parser.error(f"Unknown --extra-plots value(s): {unknown}. "
                     f"Valid choices: {_ALL_EXTRA + ['all']}")

    # Load labels
    print(f"Loading labels for {args.dataset}...")
    labels = load_labels(args.dataset)
    order = sort_order(labels)
    print(f"  {len(labels)} nodes, {len(np.unique(labels))} classes.")

    npz_files = args.npz
    n = len(npz_files)

    # Build titles list
    if n == 1:
        titles = [args.title or os.path.splitext(os.path.basename(npz_files[0]))[0]]
    else:
        if args.titles and len(args.titles) == n:
            titles = args.titles
        else:
            titles = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]

    # Load all matrices once (needed by multiple plots)
    print("Loading matrices...")
    matrices = []
    for f, title in zip(npz_files, titles):
        mat = load_matrix(f, args.matrix_key)
        print(f"  [{title}] shape={mat.shape}, min={mat.min():.4g}, max={mat.max():.4g}")
        matrices.append(mat)

    # ---- Sorted heatmap (always) ----------------------------------------
    print("\nGenerating sorted heatmap...")
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    shared_vmax = args.vmax
    if shared_vmax is None and n > 1:
        stacked = np.concatenate([m.ravel() for m in matrices])
        shared_vmax = float(np.nanpercentile(stacked, 99))
    for i, (mat, title) in enumerate(zip(matrices, titles)):
        plot_single(axes[0][i], mat, order, title, vmax=shared_vmax)
    plt.tight_layout()
    _save(fig, args.out, args.dpi)

    # ---- Individual and paired sorted heatmaps --------------------------
    if n > 1:
        print("\nGenerating individual and paired sorted heatmaps...")
        for idxs, lbl in _iter_subsets(n):
            sub_mats = [matrices[k] for k in idxs]
            sub_titles = [titles[k] for k in idxs]
            m = len(idxs)
            fig_sub, axes_sub = plt.subplots(1, m, figsize=(7 * m, 6), squeeze=False)
            sub_vmax = args.vmax
            if sub_vmax is None and m > 1:
                stacked = np.concatenate([sm.ravel() for sm in sub_mats])
                sub_vmax = float(np.nanpercentile(stacked, 99))
            for k, (mat, title) in enumerate(zip(sub_mats, sub_titles)):
                vmax_k = sub_vmax if sub_vmax is not None else float(np.nanpercentile(mat, 99))
                plot_single(axes_sub[0][k], mat, order, title, vmax=vmax_k)
            plt.tight_layout()
            _save(fig_sub, _extra_path(args.out, lbl), args.dpi)

    # ---- Extra plots --------------------------------------------------------
    if 'class-agg' in extra:
        print("\nGenerating class-aggregated matrix...")
        plot_class_agg_mean(
            _extra_path(args.out, 'class_agg'),
            matrices, labels, titles, args.dpi,
        )
        plot_class_agg_sum(
            _extra_path(args.out, 'class_agg_sum'),
            matrices, labels, titles, args.dpi,
        )
        if n > 1:
            for idxs, lbl in _iter_subsets(n):
                sub_mats = [matrices[k] for k in idxs]
                sub_titles = [titles[k] for k in idxs]
                plot_class_agg_mean(
                    _extra_path(args.out, f'class_agg_{lbl}'),
                    sub_mats, labels, sub_titles, args.dpi,
                )
                plot_class_agg_sum(
                    _extra_path(args.out, f'class_agg_sum_{lbl}'),
                    sub_mats, labels, sub_titles, args.dpi,
                )

    if 'symmetry-scatter' in extra:
        print("\nGenerating symmetry scatter...")
        plot_symmetry_scatter(
            _extra_path(args.out, 'symmetry_scatter'),
            matrices, titles, args.dpi,
        )
        if n > 1:
            for idxs, lbl in _iter_subsets(n):
                sub_mats = [matrices[k] for k in idxs]
                sub_titles = [titles[k] for k in idxs]
                plot_symmetry_scatter(
                    _extra_path(args.out, f'symmetry_scatter_{lbl}'),
                    sub_mats, sub_titles, args.dpi,
                )

    if 'entropy' in extra:
        print("\nGenerating entropy distribution...")
        plot_entropy(
            _extra_path(args.out, 'entropy'),
            matrices, titles, args.dpi,
        )
        if n > 1:
            for idxs, lbl in _iter_subsets(n):
                sub_mats = [matrices[k] for k in idxs]
                sub_titles = [titles[k] for k in idxs]
                plot_entropy(
                    _extra_path(args.out, f'entropy_{lbl}'),
                    sub_mats, sub_titles, args.dpi,
                )

    if 'same-cross-class' in extra:
        print("\nGenerating same-class vs cross-class violin...")
        plot_same_cross_class(
            _extra_path(args.out, 'same_cross_class'),
            matrices, labels, titles, args.dpi,
        )
        if n > 1:
            for idxs, lbl in _iter_subsets(n):
                sub_mats = [matrices[k] for k in idxs]
                sub_titles = [titles[k] for k in idxs]
                plot_same_cross_class(
                    _extra_path(args.out, f'same_cross_class_{lbl}'),
                    sub_mats, labels, sub_titles, args.dpi,
                )

    print("\nDone.")


if __name__ == '__main__':
    main()
