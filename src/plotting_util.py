import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import gridspec

def plot_ui_dists(gen_us, ref_us, xlim=(-0.05, 1.05), ratio_ylim=(0, 2), num_bins=64, color='fuchsia', quantile_bins=False, skip_quantiles=0, documenter=None):

    # setup figure
    for i, (ref, gen) in enumerate(zip(ref_us.T, gen_us.T)):
        fig, ax = plt.subplots(figsize=(5, 5))
        main_cell, ratio_cell = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=ax, height_ratios=[5, 1.5], hspace=0.1
        )
        main_ax = plt.subplot(main_cell)
        ratio_ax = plt.subplot(ratio_cell)
        if quantile_bins:
            total = np.hstack([ref])
            # total = np.hstack([gen])
            # total = np.hstack([ref, gen])
            quantiles = np.linspace(0, 1, num_bins+1)
            bins = np.quantile(
                total, quantiles[skip_quantiles:-skip_quantiles] if skip_quantiles else quantiles
            )
            # bins = np.quantile(total, quantiles)
        else:
            if xlim == 'auto':
                xlim = min(gen.min(), ref.min()), max(gen.max(), ref.max())
            bins = np.linspace(xlim[0], (2 if i == 0 else 1)*xlim[1], num_bins)

        # plot main dists
        ref_vals, edges, _ = main_ax.hist(
            ref, bins=bins, density=True, color='#646464', label='Dataset')
        gen_vals, _ = np.histogram(gen, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:])/2
        main_ax.plot(centers, gen_vals, color=color,
                     lw=2, alpha=0.8, label='Model')
        main_ax.semilogy()
        # main_ax.set_xlim(*np.quantile(total, quantiles[[4,-4]]))
        # main_ax.autoscale(axis='y')
        main_ax.set_ylim(None, min(main_ax.get_ylim()[1], 200))
        main_ax.set_title(f"$u_{{{i}}}$")

        ratio = gen_vals/ref_vals
        ratio_ax.plot(main_ax.get_xlim(), [1, 1], color='#646464', ls='--')
        ratio_ax.plot(centers, ratio, color=color, lw=2)

        # lo = max(ratio[np.isfinite(ratio)].min(), ratio_ylim[0]) - 0.1
        # hi = min(ratio[np.isfinite(ratio)].max(), ratio_ylim[1]) + 0.1
        # ratio_ax.set_ylim(lo, hi)
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
