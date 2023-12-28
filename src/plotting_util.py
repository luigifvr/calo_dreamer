import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import gridspec

dup = lambda a: np.append(a, a[-1])

def plot_ui_dists(gen_us, ref_us, xlim=(-0.05, 1.05), ratio_ylim=(0, 2), num_bins=64, color='fuchsia', quantile_bins=False, skip_quantiles=0, documenter=None):

    # setup figure
    for i, (ref, gen) in enumerate(zip(ref_us.T, gen_us.T)):
        fig, ax = plt.subplots(figsize=(5, 5))
        main_cell, ratio_cell = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=ax, height_ratios=[5, 1.5], hspace=0.05
        )
        main_ax = plt.subplot(main_cell)
        ratio_ax = plt.subplot(ratio_cell)
        if quantile_bins:
            total = np.hstack([ref])
            quantiles = np.linspace(0, 1, num_bins+1)
            bins = np.quantile(
                total, quantiles[skip_quantiles:-skip_quantiles] if skip_quantiles else quantiles
            )
        else:
            if xlim == 'auto':
                xlim = min(gen.min(), ref.min()), max(gen.max(), ref.max())
            bins = np.linspace(xlim[0], (2 if i == 0 else 1)*xlim[1], num_bins)

        # plot main dists
        ref_counts, edges = np.histogram(ref, bins=bins, density=False)
        ref_counts_norm = ref_counts/ref_counts.sum()
        ref_error = ref_counts_norm/np.sqrt(ref_counts)
        main_ax.step(edges, dup(ref_counts_norm), label='Geant', linestyle='-',
                     alpha=0.8, linewidth=1.0, color='k', where='post')
        main_ax.fill_between(edges, dup(ref_counts_norm+ref_error), dup(ref_counts_norm-ref_error),
                             step='post', color='k', alpha=0.2)
        
        # plot model dists
        gen_counts, edges = np.histogram(gen, bins=bins, density=False)
        gen_counts_norm = gen_counts/gen_counts.sum()
        gen_error = gen_counts_norm/np.sqrt(gen_counts)
        main_ax.step(edges, dup(gen_counts_norm), label='Model', linestyle='-',
                     alpha=0.8, linewidth=1.0, color=color, where='post')
        main_ax.fill_between(edges, dup(gen_counts_norm+gen_error), dup(gen_counts_norm-gen_error),
                             step='post', color=color, alpha=0.2)
        main_ax.semilogy()
        main_ax.set_ylim(None, min(main_ax.get_ylim()[1], 200))
        main_ax.set_title(f"$u_{{{i}}}$")

        # ratio plots
        ratio = gen_counts / ref_counts
        ratio_ax.step(edges, dup(ratio), linewidth=1.0, alpha=1.0, color=color, where='post')
        ratio_ax.fill_between(edges, dup(ratio-gen_error/ref_counts_norm),
                              dup(ratio+gen_error/ref_counts_norm), step='post',
                              color=color, alpha=0.2)
        ratio_ax.fill_between(edges, dup(1-ref_error/ref_counts_norm), dup(1+ref_error/ref_counts_norm),
                              step='post', color='k', alpha=0.2)

       
        ratio_ax.set_ylim(*ratio_ylim)
        ratio_ax.set_xlim(*main_ax.get_xlim())
        main_ax.set_xticklabels([])

        main_ax.legend(frameon=False)

        if documenter is not None:
            fig.savefig(
                documenter.get_file(f"u{i}_dist.pdf"), dpi=200, bbox_inches='tight'
            )
        else:
            plt.show()
        
        plt.close(fig)

