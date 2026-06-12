import os
import matplotlib.pyplot as plt

# Create folder to store figures
save_dir = "figures"
os.makedirs(save_dir, exist_ok=True)

# Methods to plot
methods_to_plot = [
    "Improved_Robust",
    "DDC_like+Glasso",
    "Mask+CV",
    "Mask+Mean",
    "Winsor+Glasso",
    "Mean+Glasso"
]

# Plot labels
label_map = {
    "Improved_Robust": "Proposed",
    "DDC_like+Glasso": "DDC-like+Glasso",
    "Mask+CV": "Mask+CV",
    "Mask+Mean": "Mask+Mean",
    "Winsor+Glasso": "Winsor+Glasso",
    "Mean+Glasso": "Mean+Glasso"
}

# Colors and line styles
color_map = {
    "Improved_Robust": "black",
    "DDC_like+Glasso": "purple",
    "Mask+CV": "blue",
    "Mask+Mean": "green",
    "Winsor+Glasso": "orange",
    "Mean+Glasso": "red"
}

linestyle_map = {
    "Improved_Robust": "-",
    "DDC_like+Glasso": "--",
    "Mask+CV": "-.",
    "Mask+Mean": ":",
    "Winsor+Glasso": "--",
    "Mean+Glasso": ":"
}

# Generate and save plots
for miss in [0.1, 0.2, 0.3]:
    plt.figure(figsize=(4, 3))

    df_sub = final_table[final_table["missing"] == miss].copy()

    for method in methods_to_plot:
        df_m = df_sub[df_sub["method"] == method].sort_values("contam")

        if df_m.empty:
            continue

        plt.errorbar(
            df_m["contam"],
            df_m["F1_mean"],
            yerr=df_m["F1_std"],
            marker="o",
            color=color_map[method],
            linestyle=linestyle_map[method],
            linewidth=2.5 if method == "Improved_Robust" else 1.8,
            markersize=6,
            capsize=3,
            alpha=1.0 if method == "Improved_Robust" else 0.75,
            label=label_map[method]
        )

    plt.xlabel("Contamination", fontsize=13)
    plt.ylabel("F1 Score", fontsize=13)

    # Automatic y-axis limit
    y_max = df_sub["F1_mean"].max() + df_sub["F1_std"].max() + 0.05
    plt.ylim(0.05, min(0.75, max(0.5, y_max)))

    plt.tick_params(axis="both", labelsize=13)

    # Only show legend once
    if miss == 0.1:
        plt.legend(frameon=False, fontsize=7)

    plt.tight_layout()

    miss_str = str(miss).replace(".", "")
    filename = os.path.join(save_dir, f"figure_f1_miss_{miss_str}_with_ddc.pdf")

    plt.savefig(filename, bbox_inches="tight")
    plt.show()
