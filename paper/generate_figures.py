"""
Generate publication-quality figures for the epistemic blinding preprint.

Figures:
  fig1_method.png       - Method overview (two-row workflow comparison)
  fig3_slope_chart.png  - GBM rank-shift slope chart (biology case study)
  fig4_sp500.png        - S&P 500 consistency across seeds (finance case study)

Figure 2 (smoking gun justification comparison) is a LaTeX table.

Usage:
  python paper/generate_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

BLUE = "#2171B5"
BLUE_LIGHT = "#6BAED6"
BLUE_PALE = "#C6DBEF"
ORANGE = "#E6550D"
ORANGE_LIGHT = "#FD8D3C"
ORANGE_PALE = "#FDBE85"
GRAY = "#999999"
GRAY_LIGHT = "#CCCCCC"
GRAY_DARK = "#525252"
GREEN = "#2ca02c"


# ===================================================================
# FIGURE 1: Method Workflow Comparison
# ===================================================================

def _draw_box(ax, x, y, w, h, text, subtext, facecolor, edgecolor, textcolor,
              fontsize=9, subsize=7.5, lw=1.5):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.015",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=3)
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.02, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)
        ax.text(x, y - 0.028, subtext, ha="center", va="center",
                fontsize=subsize, color=textcolor, alpha=0.8, zorder=4,
                linespacing=1.3)
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)


def _draw_arrow(ax, x1, x2, y, color=GRAY):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->,head_width=0.06,head_length=0.03",
                                color=color, lw=1.2),
                zorder=2)


def make_fig1():
    """Two-row workflow comparison: traditional vs. epistemic blinding."""

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect(0.45)  # wider than tall
    ax.axis("off")

    bh = 0.14   # box height
    gap = 0.012  # arrow gap from box edge

    # == Row 1: Traditional (Unblinded) — 4 boxes ==
    y_top = 0.80

    ax.text(0.0, y_top + 0.12, "TRADITIONAL", ha="left", va="bottom",
            fontsize=10, fontweight="bold", color=ORANGE, fontstyle="italic")

    positions_top = [0.10, 0.33, 0.58, 0.82]
    widths_top = [0.13, 0.15, 0.16, 0.11]

    _draw_box(ax, positions_top[0], y_top, widths_top[0], bh,
              "Data In", "Features + real names\n(KRAS, TP53, EGFR...)",
              facecolor=GRAY_LIGHT, edgecolor=GRAY_DARK, textcolor=GRAY_DARK,
              fontsize=8.5, subsize=7)

    _draw_arrow(ax, positions_top[0] + widths_top[0]/2 + gap,
                positions_top[1] - widths_top[1]/2 - gap, y_top)
    _draw_box(ax, positions_top[1], y_top, widths_top[1], bh,
              "LLM Agent", "Reasons from data\n+ training memory",
              facecolor=ORANGE_PALE, edgecolor=ORANGE, textcolor="#8B3A00",
              lw=2, fontsize=8.5, subsize=7)

    _draw_arrow(ax, positions_top[1] + widths_top[1]/2 + gap,
                positions_top[2] - widths_top[2]/2 - gap, y_top)
    _draw_box(ax, positions_top[2], y_top, widths_top[2], bh,
              "Top-20 Targets", "Knowledge-contaminated\nrankings",
              facecolor="#FFF3E0", edgecolor=ORANGE, textcolor="#8B3A00",
              fontsize=8.5, subsize=7)

    _draw_arrow(ax, positions_top[2] + widths_top[2]/2 + gap,
                positions_top[3] - widths_top[3]/2 - gap, y_top)
    _draw_box(ax, positions_top[3], y_top, widths_top[3], bh,
              "2.75 / 20", "approved targets\nrecovered",
              facecolor="#E8F5E9", edgecolor=GREEN, textcolor="#1B5E20",
              lw=2, fontsize=8.5, subsize=7)

    # == Divider ==
    y_mid = 0.52
    ax.plot([0.02, 0.98], [y_mid, y_mid], color=GRAY_LIGHT, linewidth=0.8,
            linestyle="--", zorder=1)
    ax.text(0.50, y_mid, " same recall \u2014 different novel candidates ",
            ha="center", va="center", fontsize=8, color=GRAY,
            fontstyle="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=2))

    # == Row 2: Epistemic Blinding — 6 boxes ==
    y_bot = 0.24

    ax.text(0.0, y_bot + 0.12, "EPISTEMIC BLINDING", ha="left", va="bottom",
            fontsize=10, fontweight="bold", color=BLUE, fontstyle="italic")

    # Even spacing for 6 boxes across full width
    positions_bot = [0.06, 0.21, 0.38, 0.55, 0.73, 0.91]
    widths_bot =    [0.08, 0.11, 0.12, 0.12, 0.12, 0.08]

    _draw_box(ax, positions_bot[0], y_bot, widths_bot[0], bh,
              "Data In", "Features +\nreal names",
              facecolor=GRAY_LIGHT, edgecolor=GRAY_DARK, textcolor=GRAY_DARK,
              fontsize=7.5, subsize=6.5)

    _draw_arrow(ax, positions_bot[0] + widths_bot[0]/2 + gap,
                positions_bot[1] - widths_bot[1]/2 - gap, y_bot)
    _draw_box(ax, positions_bot[1], y_bot, widths_bot[1], bh,
              "Anonymize", "KRAS\u2192Gene_088\nTP53\u2192Gene_041",
              facecolor=BLUE_PALE, edgecolor=BLUE, textcolor="#0D4F8B",
              fontsize=7.5, subsize=6.5)

    _draw_arrow(ax, positions_bot[1] + widths_bot[1]/2 + gap,
                positions_bot[2] - widths_bot[2]/2 - gap, y_bot)
    _draw_box(ax, positions_bot[2], y_bot, widths_bot[2], bh,
              "LLM Agent", "Reasons from\ndata only",
              facecolor=BLUE_PALE, edgecolor=BLUE, textcolor="#0D4F8B",
              lw=2, fontsize=8, subsize=6.5)

    _draw_arrow(ax, positions_bot[2] + widths_bot[2]/2 + gap,
                positions_bot[3] - widths_bot[3]/2 - gap, y_bot)
    _draw_box(ax, positions_bot[3], y_bot, widths_bot[3], bh,
              "De-anonymize", "Gene_088\u2192KRAS\nGene_041\u2192TP53",
              facecolor=BLUE_PALE, edgecolor=BLUE, textcolor="#0D4F8B",
              fontsize=7.5, subsize=6.5)

    _draw_arrow(ax, positions_bot[3] + widths_bot[3]/2 + gap,
                positions_bot[4] - widths_bot[4]/2 - gap, y_bot)
    _draw_box(ax, positions_bot[4], y_bot, widths_bot[4], bh,
              "Top-20 Targets", "Data-driven\nrankings",
              facecolor="#E3F2FD", edgecolor=BLUE, textcolor="#0D4F8B",
              fontsize=7.5, subsize=6.5)

    _draw_arrow(ax, positions_bot[4] + widths_bot[4]/2 + gap,
                positions_bot[5] - widths_bot[5]/2 - gap, y_bot)
    _draw_box(ax, positions_bot[5], y_bot, widths_bot[5], bh,
              "2.75/20", "",
              facecolor="#E8F5E9", edgecolor=GREEN, textcolor="#1B5E20",
              lw=2, fontsize=7.5)

    # == Bottom annotation ==
    ann_y = 0.02
    ann_box = FancyBboxPatch((0.05, ann_y - 0.045), 0.90, 0.075,
                              boxstyle="round,pad=0.008",
                              facecolor="white", edgecolor=GRAY_LIGHT,
                              linewidth=0.8, zorder=2)
    ax.add_patch(ann_box)
    ax.text(0.50, ann_y,
            "Both workflows recover the same validated targets (2.75/20 per indication). "
            "The difference: unblinded rankings replace\n"
            "data-driven novel candidates with literature-familiar genes "
            "\u2014 16% of the top-20 changes on average.",
            ha="center", va="center", fontsize=7, color=GRAY_DARK,
            linespacing=1.5, zorder=3)

    fig.tight_layout(pad=0.5)
    fig.savefig("paper/figures/fig1_method.png")
    print("Saved: paper/figures/fig1_method.png")
    plt.close(fig)


# ===================================================================
# FIGURE 3: GBM Slope Chart
# ===================================================================

def make_fig3():
    """Slope chart showing blinded -> unblinded rank transitions for GBM."""

    blinded = [
        "TP53", "PIK3CA", "DPP8", "COL1A1", "COL3A1",
        "NF1", "EGFR", "TRPV6", "DPP10", "SLIT3",
        "ARHGEF5", "TRPV5", "CACNA1S", "TENM3", "PTEN",
        "COL28A1", "CACNA2D1", "SCN9A", "PIK3CG", "DSG3",
    ]

    unblinded = [
        "TP53", "EGFR", "PTEN", "PIK3CA", "NF1",
        "RB1", "PIK3R1", "PDGFRA", "DPP8", "PIK3CG",
        "TRPV6", "CACNA1S", "SLIT3", "PIK3CB", "SETD2",
        "DPP10", "ARHGEF5", "TRPV5", "CACNA2D1", "TENM3",
    ]

    b_rank = {g: i + 1 for i, g in enumerate(blinded)}
    u_rank = {g: i + 1 for i, g in enumerate(unblinded)}
    all_genes = list(dict.fromkeys(blinded + unblinded))

    famous = {"TP53", "EGFR", "PTEN", "NF1", "RB1", "PIK3CA", "PIK3R1",
              "PDGFRA", "PIK3CG", "PIK3CB"}
    only_blinded = set(blinded) - set(unblinded)
    only_unblinded = set(unblinded) - set(blinded)

    fig, ax = plt.subplots(figsize=(7, 8.5))
    x_left, x_right = 0.0, 1.0
    OFF_RANK = 21.5

    for gene in all_genes:
        br = b_rank.get(gene, OFF_RANK)
        ur = u_rank.get(gene, OFF_RANK)
        is_famous = gene in famous
        is_only_b = gene in only_blinded
        is_only_u = gene in only_unblinded

        if is_only_b:
            color, alpha, lw = BLUE, 0.7, 1.5
        elif is_only_u:
            color, alpha, lw = ORANGE, 0.7, 1.5
        elif is_famous and ur < br:
            color, alpha, lw = ORANGE, 0.85, 2.2
        elif not is_famous and ur > br:
            color, alpha, lw = BLUE, 0.85, 2.2
        else:
            color, alpha, lw = GRAY_LIGHT, 0.5, 1.0

        ax.plot([x_left, x_right], [br, ur], color=color, alpha=alpha,
                linewidth=lw, solid_capstyle="round", zorder=2)

        if br <= 20:
            fw = "bold" if is_famous else "normal"
            ax.text(x_left - 0.03, br, gene, ha="right", va="center",
                    fontsize=7.5, color=color, fontweight=fw, alpha=max(alpha, 0.7))
        if ur <= 20:
            fw = "bold" if is_famous else "normal"
            ax.text(x_right + 0.03, ur, gene, ha="left", va="center",
                    fontsize=7.5, color=color, fontweight=fw, alpha=max(alpha, 0.7))

    # Dots
    for gene in all_genes:
        br = b_rank.get(gene, OFF_RANK)
        ur = u_rank.get(gene, OFF_RANK)
        is_famous = gene in famous
        if br <= 20:
            c = ORANGE if is_famous else BLUE if gene in only_blinded else GRAY
            ax.plot(x_left, br, "o", color=c, markersize=5, zorder=3)
        if ur <= 20:
            c = ORANGE if is_famous else ORANGE if gene in only_unblinded else GRAY
            ax.plot(x_right, ur, "o", color=c, markersize=5, zorder=3)

    # Headers
    ax.text(x_left, -0.5, "BLINDED", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=BLUE)
    ax.text(x_left, 0.2, "(data-driven)", ha="center", va="bottom",
            fontsize=8, color=BLUE, style="italic")
    ax.text(x_right, -0.5, "UNBLINDED", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=ORANGE)
    ax.text(x_right, 0.2, "(names visible)", ha="center", va="bottom",
            fontsize=8, color=ORANGE, style="italic")

    # Key shift annotations — positioned to avoid line overlap
    ax.annotate("PTEN: #15 \u2192 #3\n(fame effect: \u221212)",
                xy=(0.58, 9.0), fontsize=7.5, color=ORANGE,
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                          edgecolor=ORANGE, alpha=0.85))

    ax.annotate("DPP8: #3 \u2192 #9\n(obscurity penalty: +6)",
                xy=(0.30, 5.5), fontsize=7.5, color=BLUE,
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD",
                          edgecolor=BLUE, alpha=0.85))

    # Off-list zone
    ax.axhline(y=20.5, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.5, 21.2, "\u2190 off top 20 \u2192", ha="center", va="center",
            fontsize=7, color=GRAY, style="italic")

    ax.set_ylim(22.5, -1)
    ax.set_xlim(-0.35, 1.35)
    ax.set_yticks(range(1, 21))
    ax.set_ylabel("Rank", fontsize=11)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.tick_params(axis="y", labelsize=8)

    # Legend
    ax.legend(
        handles=[
            mpatches.Patch(color=ORANGE, alpha=0.8, label="Famous gene (promoted when named)"),
            mpatches.Patch(color=BLUE, alpha=0.8, label="Obscure gene (demoted when named)"),
            mpatches.Patch(color=GRAY_LIGHT, alpha=0.6, label="Stable (small shift)"),
        ],
        loc="lower center", fontsize=7.5, framealpha=0.9,
        ncol=1, bbox_to_anchor=(0.5, -0.06))

    ax.set_title("IDH-Wildtype Glioblastoma: How Rankings Shift\nWhen the LLM Knows Gene Names",
                 fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig("paper/figures/fig3_slope_chart.png")
    print("Saved: paper/figures/fig3_slope_chart.png")
    plt.close(fig)


# ===================================================================
# FIGURE 4: S&P 500 Consistency Across Seeds
# ===================================================================

def make_fig4():
    """Strip chart showing which tickers are consistently promoted/demoted."""

    from collections import Counter

    seeds = [42, 137, 256, 389, 501]
    promoted = {
        42:  ["HIG", "ELV", "QCOM", "WDAY", "CRM", "CI"],
        137: ["CINF", "FSLR", "ELV", "NVDA", "HCA", "CRM", "INTU"],
        256: ["BMY", "FIS", "AMP", "CPAY", "CI"],
        389: ["DELL", "MKC", "ELV", "CI", "CINF"],
        501: ["FSLR", "CINF", "CI", "BMY", "ELV", "DELL", "CTSH"],
    }
    demoted = {
        42:  ["PRU", "FSLR", "CTRA", "LULU", "AIZ", "EXE"],
        137: ["TTD", "NTAP", "ACN", "PTC", "GPN", "CTRA", "AOS"],
        256: ["SYF", "TROW", "NEM", "CTRA"],
        389: ["NTAP", "DAL", "TRV", "TXT", "PRU"],
        501: ["MU", "SYF", "TRV", "GPN", "DECK", "DAL", "BBY"],
    }

    promo_counts = Counter()
    demo_counts = Counter()
    for s in seeds:
        for t in promoted[s]:
            promo_counts[t] += 1
        for t in demoted[s]:
            demo_counts[t] += 1

    promo_tickers = sorted([t for t, c in promo_counts.items() if c >= 2],
                           key=lambda t: -promo_counts[t])
    demo_tickers = sorted([t for t, c in demo_counts.items() if c >= 2],
                          key=lambda t: -demo_counts[t])
    all_tickers = promo_tickers + demo_tickers

    n_tickers = len(all_tickers)
    n_seeds = len(seeds)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for i, ticker in enumerate(all_tickers):
        for j, seed in enumerate(seeds):
            if ticker in promoted[seed]:
                ax.plot(j, i, marker="^", color=ORANGE, markersize=10, zorder=3)
            elif ticker in demoted[seed]:
                ax.plot(j, i, marker="v", color=BLUE, markersize=10, zorder=3)
            else:
                ax.plot(j, i, marker="o", color=GRAY_LIGHT, markersize=4,
                        zorder=2, alpha=0.5)

    divider_y = len(promo_tickers) - 0.5
    ax.axhline(y=divider_y, color=GRAY, linestyle="-", linewidth=0.8, alpha=0.5)

    # Section labels inside the chart area
    ax.text(2.0, 0.0 - 0.6, "\u2191 Promoted when named",
            ha="center", va="center", fontsize=8, color=ORANGE, fontstyle="italic")
    ax.text(2.0, divider_y + 0.6, "\u2193 Demoted when named",
            ha="center", va="center", fontsize=8, color=BLUE, fontstyle="italic")

    for i, ticker in enumerate(all_tickers):
        if ticker in promo_tickers:
            count = promo_counts[ticker]
            c = ORANGE if count >= 3 else "#333"
            fw = "bold" if count >= 4 else "normal"
            ax.text(-0.3, i, ticker, ha="right", va="center", fontsize=9,
                    fontweight=fw, color=c)
            ax.text(n_seeds + 0.15, i, f"{count}/5", ha="left", va="center",
                    fontsize=8, color=ORANGE, fontweight=fw)
        else:
            count = demo_counts[ticker]
            c = BLUE if count >= 3 else "#333"
            fw = "bold" if count >= 4 else "normal"
            ax.text(-0.3, i, ticker, ha="right", va="center", fontsize=9,
                    fontweight=fw, color=c)
            ax.text(n_seeds + 0.15, i, f"{count}/5", ha="left", va="center",
                    fontsize=8, color=BLUE, fontweight=fw)

    for j, seed in enumerate(seeds):
        ax.text(j, -1.5, f"Seed {seed}", ha="center", va="bottom",
                fontsize=8, color="#555")

    ax.text(n_seeds + 0.15, -1.5, "Freq", ha="left", va="bottom",
            fontsize=8, fontweight="bold", color="#555")

    for i in range(n_tickers):
        ax.axhline(y=i, color="#f0f0f0", linewidth=0.5, zorder=1)

    ax.set_xlim(-0.8, n_seeds + 0.6)
    ax.set_ylim(n_tickers - 0.5, -2.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        handles=[
            ax.scatter([], [], marker="^", color=ORANGE, s=60,
                       label="Entered top-20 when named"),
            ax.scatter([], [], marker="v", color=BLUE, s=60,
                       label="Dropped from top-20 when named"),
        ],
        loc="lower center", fontsize=8, framealpha=0.9,
        ncol=2, bbox_to_anchor=(0.45, -0.06))

    ax.set_title("S&P 500 Value Screen: Systematic Bias Across 5 Random Seeds\n"
                 "Each seed shuffles entity codes differently \u2014 consistent arrows = systematic bias, not noise",
                 fontsize=11, fontweight="bold", pad=12,
                 linespacing=1.5)
    # Make subtitle lighter
    ax.title.set_fontsize(11)

    fig.tight_layout()
    fig.savefig("paper/figures/fig4_sp500.png")
    print("Saved: paper/figures/fig4_sp500.png")
    plt.close(fig)


# ===================================================================
if __name__ == "__main__":
    make_fig1()
    make_fig3()
    make_fig4()
    print("Done.")
