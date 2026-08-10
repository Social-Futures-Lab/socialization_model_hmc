"""
Restyled versions of the two figures. Swap the demo `records` for your own.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from poster_style import use_poster_style, DATA_COLORS, DATA_PALETTE, MUTED, save


# ===========================================================================
# FIGURE 1 -- in-group language by subreddit
# ===========================================================================
def plot_ingroup(df, output_file="fig_ingroup"):
    use_poster_style(base_size=15)

    # Order groups explicitly so colors are stable, and sort subreddits by
    # value *within* group -- a reader can then scan top-to-bottom.
    group_order = ["High", "Medium", "Low"]          # <- your "Expected" levels
    means = df.groupby(["Expected", "Subreddit"], observed=True)[
        "% In-Group Language"].mean().reset_index()
    means["Expected"] = pd.Categorical(means["Expected"], group_order, ordered=True)
    sub_order = means.sort_values(
        ["Expected", "% In-Group Language"], ascending=[True, False])["Subreddit"].tolist()

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.pointplot(
        ax=ax, data=df,
        x="% In-Group Language", y="Subreddit",
        hue="Expected", hue_order=group_order, order=sub_order,
        linestyle="none",
        errorbar=("pi", 95), capsize=0.3,
        palette=DATA_PALETTE[:len(group_order)],
        markers=["o", "s", "D"],          # redundant encoding: shape + color
        markersize=11, markeredgecolor="white", markeredgewidth=1.2,
        err_kws={"linewidth": 2.6},
        dodge=False, legend=True,
    )

    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f"{t:.0%}" for t in np.arange(0, 1.01, 0.2)])
    ax.set_xlabel("In-group language")
    ax.set_ylabel("")                      # "r/" prefix makes it self-evident
    ax.set_yticks(range(len(sub_order)), labels=[f"r/{s}" for s in sub_order])

    # Grid only along the measured axis; horizontal rules add nothing here.
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    # Faint band separating the expectation groups.
    boundaries = np.cumsum(
        means.set_index("Subreddit").loc[sub_order, "Expected"]
        .value_counts(sort=False).reindex(group_order).values)[:-1]
    for b in boundaries:
        ax.axhline(b - 0.5, color="#C9CDD4", linewidth=1.2, zorder=0)

    leg = ax.legend(title="Expected in-group\nlanguage", loc="lower right",
                    handletextpad=0.4, borderaxespad=0.8, labelspacing=0.5)
    leg.get_title().set_fontsize(14)
    leg.get_title().set_color(MUTED)

    fig.tight_layout()
    save(fig, output_file)


# ===========================================================================
# FIGURE 2 -- COVID subreddits over time
# ===========================================================================
def plot_covid(df, output_file="fig_covid"):
    use_poster_style(base_size=15)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Time"], format="mixed")

    sub_order = ["Coronavirus", "China_Flu"]
    colors = {"Coronavirus": DATA_COLORS["violet"], "China_Flu": DATA_COLORS["rose"]}

    fig, ax = plt.subplots(figsize=(36, 10))

    sns.lineplot(
        ax=ax, data=df, x="Date", y="% In-Group Language",
        hue="Subreddit", hue_order=sub_order, palette=colors,
        style="Subreddit", dashes={"Coronavirus": "", "China_Flu": (5, 2)},
        errorbar=("pi", 95), linewidth=3.4,
        err_kws={"alpha": 0.18, "linewidth": 0},
        legend=False,                      # replaced by direct labels below
    )

    # Direct labels beat a legend: no back-and-forth eye movement from 2 m away.
    for sub in sub_order:
        last = df[df["Subreddit"] == sub].sort_values("Date").iloc[-1]
        ax.annotate(f"r/{sub}",
                    xy=(last["Date"], last["% In-Group Language"]),
                    xytext=(10, 0), textcoords="offset points",
                    color=colors[sub], fontweight="semibold",
                    va="center", fontsize=16)

    # Dates: one tick per quarter, year shown once.
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax.xaxis.get_major_locator()))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlabel("")
    ax.set_ylabel("In-group language")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", visible=False)

    # Leave room on the right for the direct labels.
    ax.margins(x=0.02)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.16)

    fig.tight_layout()
    save(fig, output_file)

