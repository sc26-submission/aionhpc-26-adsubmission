import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Plotting defaults
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "DejaVu Serif",
        ],

        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,

        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        "hatch.linewidth": 0.5,

        # Keep text/fonts embedded cleanly in vector PDF
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


# ============================================================
# Data
# ============================================================

models = [
    "Llama-8B",
    "Qwen-8B",
    "Qwen-14B",
]

data = {
    "Conversation": {
        "Persistent-Only": [3.030, 2.927, 1.913],
        "Static P/D":      [3.379, 3.379, 2.545],
        "OpServe":         [3.379, 3.379, 3.324],
    },

    "Synthetic": {
        "Persistent-Only": [2.835, 2.691, 1.740],
        "Static P/D":      [3.202, 3.135, 1.938],
        "OpServe":         [3.879, 3.864, 2.750],
    },

    "Tool & Agent": {
        "Persistent-Only": [5.703, 5.472, 3.041],
        "Static P/D":      [6.633, 6.635, 4.050],
        "OpServe":         [6.633, 6.635, 5.684],
    },
}


# ============================================================
# Figure layout
# Full-width figure for a two-column paper
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.1, 2.55),
    sharey=True,
)

x = np.arange(len(models))
width = 0.24


# ============================================================
# Bar styles
#
# Persistent = neutral baseline
# Static P/D = orange baseline
# OpServe = strongest visual emphasis
# ============================================================

styles = {
    "Persistent-Only": {
        "color": "#B8B8B8",
        "hatch": "///",
    },

    "Static P/D": {
        "color": "#E69F00",
        "hatch": "\\\\",
    },

    "OpServe": {
        "color": "#0072B2",
        "hatch": "xx",
    },
}

offsets = {
    "Persistent-Only": -width,
    "Static P/D": 0,
    "OpServe": width,
}


# ============================================================
# Plot each workload
# ============================================================

for ax, (workload, workload_data) in zip(
    axes,
    data.items(),
):

    for policy in [
        "Persistent-Only",
        "Static P/D",
        "OpServe",
    ]:

        ax.bar(
            x + offsets[policy],
            workload_data[policy],
            width=width,
            label=policy,
            color=styles[policy]["color"],
            hatch=styles[policy]["hatch"],
            edgecolor="black",
            linewidth=0.65,
        )

    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    ax.set_title(
        workload,
        fontsize=10.0,
        fontweight="bold",
        pad=5,
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        models,
        fontsize=9.0,
        rotation=0,
    )

    ax.tick_params(
        axis="x",
        pad=3,
        width=0.7,
        length=3.0,
    )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    ax.set_ylim(
        0,
        7.1,
    )

    ax.set_yticks(
        np.arange(
            0,
            8,
            1,
        )
    )

    ax.tick_params(
        axis="y",
        labelsize=9.0,
        pad=2,
        width=0.7,
        length=3.0,
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.20,
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Horizontal padding
    # --------------------------------------------------------

    ax.set_xlim(
        -0.55,
        len(models) - 0.45,
    )


# ============================================================
# Shared Y-axis label
# ============================================================

axes[0].set_ylabel(
    "Throughput (req/s)",
    fontsize=9.5,
)


# ============================================================
# Shared legend
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.015),
    ncol=3,
    frameon=False,
    fontsize=9.0,
    handlelength=1.5,
    handletextpad=0.45,
    columnspacing=1.4,
    borderaxespad=0.0,
)


# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.065,
    right=0.995,
    bottom=0.18,
    top=0.79,
    wspace=0.16,
)


# ============================================================
# Save
#
# IMPORTANT:
# Use the PDF in the paper, not the PNG.
# ============================================================

fig.savefig(
    "throughput_three_workloads.pdf",
)

# PNG only for quick viewing/debugging
fig.savefig(
    "throughput_three_workloads.png",
    dpi=600,
)

plt.show()