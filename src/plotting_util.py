import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

def plot_ui_dists(gen_us, ref_us, xlim=(-0.05, 1.05), ratio_ylim=(0.5, 1.5), num_bins=64, color='fuchsia', quantile_bins=False, skip_quantiles=0, documenter=None):

    # iterate layers
    for i, (ref, gen) in enumerate(zip(ref_us.T, gen_us.T)):

        # create figure and subaxes
        fig, ax = plt.subplots(figsize=(5, 5))
        main_cell, ratio_cell = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=ax, height_ratios=[5, 1.5], hspace=0.05
        )
        main_ax = plt.subplot(main_cell)
        ratio_ax = plt.subplot(ratio_cell)
        ax.set_axis_off()

        # set binning
        if quantile_bins:
            total = np.hstack([ref])
            quantiles = np.linspace(0, 1, num_bins+1)
            bins = np.quantile(
                total, quantiles[skip_quantiles:-skip_quantiles] if skip_quantiles else quantiles
            )
        else:
            if xlim == 'auto':
                xlim = ref.min(), ref.max()
            bins = np.linspace(xlim[0], (2 if i == 0 else 1)*xlim[1], num_bins)
        bin_centers = (bins[1:] + bins[:-1])/2
        bin_widths = bins[1:] - bins[:-1]

        ## MAIN AXIS ##
        # plot central values 
        ref_vals, _, _ = main_ax.hist(
            ref, bins=bins, density=True, histtype='step', linestyle='-',
            alpha=0.8, linewidth=1.0, color='k', label='Geant'
        )
        gen_vals, _, _ = main_ax.hist(
            gen, bins=bins, density=True, histtype='step', linestyle='-',
            alpha=0.8, linewidth=1.0, color=color, label='CaloDREAM'
        )
        # plot error bars
        ref_stds = np.sqrt(ref_vals*len(ref)/bin_widths)/len(ref)
        gen_stds = np.sqrt(gen_vals*len(gen)/bin_widths)/len(gen)
        main_ax.fill_between(
            bin_centers, ref_vals-ref_stds, ref_vals+ref_stds, alpha=0.2,
            step='mid', color='k'
        )
        main_ax.fill_between(
            bin_centers, gen_vals-gen_stds, gen_vals+gen_stds, alpha=0.2,
            step='mid', color=color
        )
        main_ax.semilogy()
        # main_ax.set_ylim(None, min(main_ax.get_ylim()[1], 200))
        main_ax.set_ylim(max(main_ax.get_ylim()[0], 1.1e-4), None)
        main_ax.set_ylabel(f"Prob. density")
        
        ## RATIO AXIS ##
        norm = ref_vals
        # plot central values 
        ratio_ax.step(
            bin_centers, ref_vals/norm, where='mid', linestyle='-', alpha=0.8,
            linewidth=1.0, color='k'
        )
        ratio_ax.step(
            bin_centers, gen_vals/norm, where='mid', linestyle='-', alpha=0.8,
            linewidth=1.0, color=color
        )
        ratio_ax.fill_between(
            bin_centers, (ref_vals-ref_stds)/norm, (ref_vals+ref_stds)/norm,
            alpha=0.2, step='mid', color='k'
        )
        ratio_ax.fill_between(
            bin_centers, (gen_vals-gen_stds)/norm, (gen_vals+gen_stds)/norm,
            alpha=0.2, step='mid', color=color
        )
        ratio_ax.set_ylim(*ratio_ylim)
        ratio_ax.set_xlim(*main_ax.get_xlim())
        ratio_ax.set_xlabel(f"$u_{{{i}}}$")
        main_ax.set_xticklabels([])        
        leg = main_ax.legend(frameon=False)

        for p in leg.get_patches():
            p.set(linewidth=1.5)
        
        if documenter is not None:
            fig.savefig(
                documenter.get_file(f"u{i}_dist.pdf"), dpi=200, bbox_inches='tight'
            )
        else:
            plt.show()
        
        plt.close(fig)