"""Figures for Lab 2."""

import matplotlib.pyplot as plt
import numpy as np
from textwrap import fill
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


answer_labels = {
    "I use lightning bug and firefly interchangeably": "Uses both",
    "Other responses": "Other\nresponses",
    "No word": "No\nword",
    "Don't recognize": "Don't\nrecognize",
}

missing_color = "#E3E5E7"
title_style = {"fontsize": 13, "fontweight": "bold", "fontfamily": "DejaVu Sans"}


def set_figure_title(fig, title):
    """Set the shared centered title style on an existing figure.

    Args:
        fig (matplotlib.figure.Figure): Figure to modify in place.
        title (str): Explanatory figure title.

    Returns:
        None.
    """
    fig.suptitle(title, x=0.5, y=0.98, ha="center", **title_style)


def save_figure(fig, filename):
    """Save, display, and close a figure in the relative figures directory.

    Args:
        fig (matplotlib.figure.Figure): Completed figure to save and close.
        filename (str): File stem without an extension, such as
            q66_response_maps. Existing files with this stem are overwritten.

    Returns:
        None. Writes a PDF and a 180-dpi PNG preview to ../figs relative to
        the working directory. Run from lab2/code with ../figs already present.

    Notes:
        Save before displaying. Preserve the canvas dimensions so figure
        titles, shared legends, and inset positions remain aligned.
    """
    fig.savefig("../figs/" + filename + ".pdf")
    fig.savefig("../figs/" + filename + ".png", dpi=180)
    plt.show()
    plt.close(fig)


def plot_response_maps(state_summary, state_shapes, question, responses,
                       title, filename, min_respondents=30, colorbar_label=None):
    """Map state-level response or cluster shares with a shared 0-100 scale.

    Args:
        state_summary (pandas.DataFrame): One row per question/answer/state,
            with question, answer, state, responses, respondents, and percent
            columns. For clustering, question identifies the method and
            answer identifies the cluster. Denominators must include every
            valid answer or cluster assignment, including categories not
            displayed. Include zero counts for absent categories in a state
            with available data.
        state_shapes (geopandas.GeoDataFrame): US state/DC polygons with
            postal abbreviations and a defined coordinate system.
        question (str): Question or method identifier, such as Q066 or GMM.
        responses (sequence[str]): Nonempty ordered list of answers to map.
        title (str): Figure title.
        filename (str): Output file stem passed to save_figure.
        min_respondents (int): Minimum answering respondents per state;
            lower counts and missing values are masked in gray. Default 30.
        colorbar_label (str or None): Optional label describing the mapped
            percentages. None or an empty string uses the survey-answer label.

    Returns:
        None. Saves, displays, and closes the figure via save_figure.

    Notes:
        Each panel reports its cohort-wide response count and percentage,
        including low-count states masked on the map. Alaska and Hawaii use
        separate insets. The map describes survey respondents' reported
        states, not a population-representative estimate of state usage.
    """
    rows = (len(responses) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.3 * rows + 0.8), squeeze=False)
    top = 0.79 if rows == 1 else 0.87
    fig.subplots_adjust(left=0.02, right=0.98, top=top, bottom=0.18,
                        wspace=0.06, hspace=0.40)
    norm = Normalize(vmin=0, vmax=100)
    cmap = plt.get_cmap("YlGnBu")
    mainland = state_shapes.loc[~state_shapes["postal"].isin(["AK", "HI"])]
    mainland = mainland.to_crs("EPSG:5070")
    bounds = mainland.total_bounds
    has_missing = False

    for panel, response in enumerate(responses):
        ax = axes.flat[panel]
        selected = state_summary.loc[
            (state_summary["question"] == question) &
            (state_summary["answer"] == response)
        ]
        map_data = state_shapes.merge(selected, left_on="postal", right_on="state", how="left")
        too_few = map_data["respondents"].fillna(0) < min_respondents
        map_data.loc[too_few, "percent"] = np.nan
        has_missing = has_missing or map_data["percent"].isna().any()
        main_data = map_data.loc[~map_data["postal"].isin(["AK", "HI"])]
        main_data = main_data.to_crs("EPSG:5070")
        main_data.plot(ax=ax, color=missing_color, edgecolor="white", linewidth=0.6)
        available = main_data.loc[main_data["percent"].notna()]
        if len(available) > 0:
            available.plot(column="percent", ax=ax, cmap=cmap, vmin=0, vmax=100,
                           edgecolor="white", linewidth=0.6)
        ax.set_xlim(bounds[0] - 80000, bounds[2] + 80000)
        ax.set_ylim(bounds[1] - 350000, bounds[3] + 80000)
        ax.set_axis_off()

        for state, position, crs in [
            ("AK", [0.01, 0.00, 0.23, 0.28], "EPSG:3338"),
            ("HI", [0.27, 0.02, 0.18, 0.20], "EPSG:4326"),
        ]:
            inset = ax.inset_axes(position)
            state_data = map_data.loc[map_data["postal"] == state].to_crs(crs)
            state_data.plot(ax=inset, color=missing_color, edgecolor="white", linewidth=0.5)
            available = state_data.loc[state_data["percent"].notna()]
            if len(available) > 0:
                available.plot(column="percent", ax=inset, cmap=cmap, vmin=0, vmax=100,
                               edgecolor="white", linewidth=0.5)
            inset.set_axis_off()
            inset.text(0.5, -0.05, state, transform=inset.transAxes,
                       ha="center", fontsize=9, color="#555555")

        label = answer_labels.get(response, response.capitalize())
        ax.set_title(label, fontsize=11, fontweight="semibold", pad=22)
        response_count = selected["responses"].sum()
        total = selected["respondents"].sum()
        percent = 100 * response_count / total
        ax.text(0.5, 1.025, f"{response_count:,} of {total:,} respondents ({percent:.1f}%)",
                transform=ax.transAxes, ha="center", fontsize=10)

    for ax in list(axes.flat)[len(responses):]:
        ax.set_axis_off()
    colorbar_y = 0.15 if has_missing else 0.095
    colorbar_ax = fig.add_axes([0.30, colorbar_y, 0.40, 0.025])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label(colorbar_label or "Respondents choosing this answer within each state (%)", fontsize=11)
    if has_missing:
        fig.legend(handles=[Patch(facecolor=missing_color,
                                  label=f"No data or fewer than {min_respondents} respondents")],
                   loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=10)
    set_figure_title(fig, title)
    save_figure(fig, filename)


def plot_question_association(percentages, baseline, title, filename):
    """Show conditional Q74 percentages minus their overall baselines.

    Args:
        percentages (pandas.DataFrame): Q66 answers on rows and Q74 answers
            on columns. Values are conditional percentages on a 0-100 scale;
            empty respondent groups should contain NaN, not zero percent.
        baseline (pandas.Series): Overall Q74 percentages for the same cohort,
            indexed by column answer labels and weighted by respondent count.
        title (str): Figure title.
        filename (str): Output file stem passed to save_figure.

    Returns:
        None. Saves, displays, and closes the figure via save_figure.

    Notes:
        Cell numbers and colors both represent percentage-point differences,
        not raw conditional percentages, correlations, or prediction accuracy.
        The diverging color scale is symmetric around zero; NaN cells are gray.
    """
    differences = percentages.sub(baseline.reindex(percentages.columns), axis=1)
    values = differences.to_numpy(dtype=float)
    color_limit = max(5, 5 * np.ceil(np.nanmax(np.abs(values)) / 5))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(missing_color)

    fig = plt.figure(figsize=(6.5, 3.7))
    ax = fig.add_axes([0.20, 0.25, 0.78, 0.54])
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
                      vmin=-color_limit, vmax=color_limit, aspect="auto")
    for row_number, column_number in np.ndindex(values.shape):
        difference = values[row_number, column_number]
        label = f"{difference:+.1f}" if np.isfinite(difference) else "-"
        if np.isfinite(difference) and abs(difference) < 0.05:
            label = "0.0"
        color = "white" if abs(difference) > 0.65 * color_limit else "#222222"
        ax.text(column_number, row_number, label, ha="center", va="center",
                fontsize=9, color=color)

    row_labels = []
    for response in percentages.index:
        label = answer_labels.get(response, response.capitalize()).replace("\n", " ")
        row_labels.append(label)
    column_labels = []
    for response in percentages.columns:
        label = answer_labels.get(response, response.capitalize().replace(" ", "\n"))
        column_labels.append(label)
    ax.set_yticks(range(len(values)), labels=row_labels)
    ax.set_xticks(range(len(column_labels)), labels=column_labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=6, labelsize=9)
    ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(values), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.axvline(4.5, color="white", linewidth=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar_ax = fig.add_axes([0.30, 0.12, 0.40, 0.028])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal",
                           ticks=np.linspace(-color_limit, color_limit, 5))
    colorbar.set_label("Difference from baseline (percentage points)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8, length=2)
    set_figure_title(fig, title)
    save_figure(fig, filename)


def plot_three_question_association(group_tables, title, filename):
    """Compare roly-poly usage conditional on Q65 and common Q66 answers.

    Args:
        group_tables (dict): Ordered mapping from Q65 answer text to tables.
            Each entry has counts, row_counts, and percentages for Q66 rows
            and Q74 columns. percentages must use all Q74 answers in each
            denominator and be on a 0-100 scale. Expected Q66 rows include
            crawfish, crayfish, and crawdad; Q74 includes roly poly.
        title (str): Figure title.
        filename (str): Output file stem passed to save_figure.

    Returns:
        None. Saves, displays, and closes the figure via save_figure.

    Notes:
        Each point is P(Q74 = roly poly | Q65 group, Q66 answer), expressed
        as a percentage. Point denominators are stored in
        group_tables[group]['row_counts']; they are not annotated. Other
        Q65 groups, Q66 terms, and Q74 outcomes are not plotted. This is a
        descriptive comparison, not a test of independence or generalization.
    """
    responses = ["crawfish", "crayfish", "crawdad"]
    colors = ["#337D8A", "#9A4C21", "#715A91"]
    markers = ["o", "s", "^"]
    offsets = [-0.19, 0, 0.19]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.subplots_adjust(left=0.25, right=0.96, top=0.72, bottom=0.15)
    for response, color, marker, offset in zip(responses, colors, markers, offsets):
        percentages = [tables["percentages"].loc[response, "roly poly"]
                       for tables in group_tables.values()]
        positions = np.arange(len(group_tables)) + offset
        ax.scatter(percentages, positions, color=color, marker=marker, s=45,
                   label=response.capitalize(), zorder=3)
        for percent, position in zip(percentages, positions):
            if np.isfinite(percent):
                ax.text(percent + 1.1, position, f"{percent:.1f}%", va="center",
                        fontsize=9, color=color)

    group_labels = [answer_labels.get(group, group.capitalize()) for group in group_tables]
    ax.set_yticks(range(len(group_labels)), labels=group_labels)
    ax.set_ylim(len(group_labels) - 0.5, -0.5)
    ax.set_xlim(0, 65)
    ax.set_xticks(range(0, 61, 10), labels=[f"{value}%" for value in range(0, 61, 10)])
    ax.set_ylabel("Glowing insect name", fontsize=9, labelpad=10)
    ax.grid(axis="x", color="#E8EAEC", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.legend(title="Freshwater crustacean name", ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.89), borderaxespad=0,
               fontsize=9, title_fontsize=9)
    set_figure_title(fig, title)
    fig.text(0.5, 0.04, "Respondents choosing 'roly poly'", fontsize=9, ha="center", va="center")
    save_figure(fig, filename)


region_colors = {
    "Northeast": "#0072B2",
    "Midwest": "#E69F00",
    "South": "#009E73",
    "West": "#CC79A7",
}


def plot_pca_variance(pca_models, title, filename):
    """Compare individual and cumulative explained variance for fitted PCA models.

    Args:
        pca_models (dict): Nonempty ordered mapping from display names to
            fitted PCA estimators with explained_variance_ratio_. Use equal
            component counts; only the first two entries are plotted.
        title (str): Figure title.
        filename (str): Output file stem without an extension.

    Returns:
        None. Saves a PDF and PNG, displays, and closes the figure through
        save_figure. Run from lab2/code with the figures directory present.

    Notes:
        Variance fractions are converted to percentages. Each curve refers
        to its model's input representation, which may differ through scaling.
        The x-axis tick positions are configured for the 50-component analysis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.77, bottom=0.15, wspace=0.28)
    colors = ["#337D8A", "#9A4C21"]
    for (name, model), color in zip(pca_models.items(), colors):
        components = np.arange(1, len(model.explained_variance_ratio_) + 1)
        variance = 100 * model.explained_variance_ratio_
        axes[0].plot(components, variance, color=color, label=name)
        axes[1].plot(components, variance.cumsum(), color=color, label=name)

    axes[0].set_title("Variation explained by each component", fontsize=11)
    axes[1].set_title("Cumulative variation explained", fontsize=11)
    axes[0].set_ylabel("Explained variance (%)")
    axes[1].set_ylabel("Cumulative explained variance (%)")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xlabel("Principal component")
        ax.set_xlim(1, len(components))
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=2, frameon=False)
    set_figure_title(fig, title)
    save_figure(fig, filename)


def plot_geographic_projections(views, regions, title, filename):
    """Plot dimension reduction views with consistent geographic region colors.

    Args:
        views (dict): Nonempty ordered mapping from panel names to dictionaries
            containing scores, x, y, x_label, and y_label. scores is a DataFrame
            indexed by respondent; x and y name its plotted columns, and
            x_label and y_label supply the axis labels.
        regions (pandas.Series): Region names indexed by the respondents to
            display. Values must be keys in region_colors, and every index
            must occur in each view's scores. The legend reports these counts.
        title (str): Figure title.
        filename (str): Output file stem without an extension.

    Returns:
        None. Saves a PDF and PNG, displays, and closes the figure through
        save_figure. Run from lab2/code with the figures directory present.

    Notes:
        Every panel selects the same respondents. Drawing order is shuffled
        with seed 215 to reduce systematic overlap; scatter points are
        rasterized in the saved PDF. Geography supplies colors only, and the
        supplied projection scores and region labels are not modified.
    """
    rows = (len(views) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4.6 * rows + 1.0), squeeze=False)
    top = 0.81 if rows == 1 else 0.89
    bottom = 0.23 if rows == 1 else 0.17
    fig.subplots_adjust(left=0.07, right=0.98, top=top, bottom=bottom,
                        wspace=0.25, hspace=0.40)

    for panel, (name, view) in enumerate(views.items()):
        ax = axes.flat[panel]
        scores = view["scores"].loc[regions.index].sample(frac=1, random_state=215)
        colors = regions.loc[scores.index].map(region_colors)
        ax.scatter(scores[view["x"]], scores[view["y"]], c=colors,
                   s=3, alpha=0.25, edgecolors="none", rasterized=True)
        ax.set_title(name, fontsize=11, fontweight="semibold")
        ax.set_xlabel(view["x_label"], fontsize=10)
        ax.set_ylabel(view["y_label"], fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in list(axes.flat)[len(views):]:
        ax.set_axis_off()
    counts = regions.value_counts()
    handles = [Patch(facecolor=color, label=f"{region} (n={counts.get(region, 0):,})")
               for region, color in region_colors.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=4, frameon=False, fontsize=12)
    set_figure_title(fig, title)
    save_figure(fig, filename)


def plot_component_maps(state_summary, state_shapes, components, variance,
                        title, filename, min_respondents=30):
    """Map supplied state-average component scores on separate diverging scales.

    Args:
        state_summary (pandas.DataFrame): One row per state, with state postal
            abbreviations, respondents counts, and a mean-score column for
            each requested component. Aggregation is performed by the caller.
        state_shapes (geopandas.GeoDataFrame): US state/DC polygons with a
            postal abbreviation column and a defined coordinate system.
        components (sequence[str]): Nonempty ordered list of mean-score
            column names to map, such as PC1 and PC2.
        variance (sequence[float]): Explained variance fractions on a 0-1
            scale, in the same order as components, used in panel titles.
        title (str): Figure title.
        filename (str): Output file stem without an extension.
        min_respondents (int): Minimum respondents per state; smaller counts
            and missing scores are masked in gray. Default 30.

    Returns:
        None. Saves a PDF and PNG, displays, and closes the figure through
        save_figure. Run from lab2/code with the figures directory present.

    Notes:
        Each component has its own color scale symmetric around zero, based
        on its largest unmasked absolute state mean with a minimum limit of
        0.01. Read each colorbar when comparing panels. Alaska and Hawaii use
        separate insets; the figure describes the supplied survey cohort.
    """
    fig, axes = plt.subplots(1, len(components), figsize=(7 * len(components), 5.6), squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.30, wspace=0.12)
    map_data = state_shapes.merge(state_summary, left_on="postal", right_on="state", how="left")
    too_few = map_data["respondents"].fillna(0) < min_respondents
    mainland = state_shapes.loc[~state_shapes["postal"].isin(["AK", "HI"])]
    bounds = mainland.to_crs("EPSG:5070").total_bounds
    has_missing = False

    for panel, component in enumerate(components):
        ax = axes.flat[panel]
        selected = map_data.copy()
        selected.loc[too_few, component] = np.nan
        has_missing = has_missing or selected[component].isna().any()
        limit = max(0.01, selected[component].abs().max())
        main_data = selected.loc[~selected["postal"].isin(["AK", "HI"])].to_crs("EPSG:5070")
        main_data.plot(ax=ax, color=missing_color, edgecolor="white", linewidth=0.6)
        available = main_data.loc[main_data[component].notna()]
        if len(available) > 0:
            available.plot(column=component, ax=ax, cmap="RdBu", vmin=-limit, vmax=limit,
                           edgecolor="white", linewidth=0.6)
        ax.set_xlim(bounds[0] - 80000, bounds[2] + 80000)
        ax.set_ylim(bounds[1] - 350000, bounds[3] + 80000)
        ax.set_axis_off()

        for state, position, crs in [
            ("AK", [0.01, 0.00, 0.23, 0.28], "EPSG:3338"),
            ("HI", [0.27, 0.02, 0.18, 0.20], "EPSG:4326"),
        ]:
            inset = ax.inset_axes(position)
            state_data = selected.loc[selected["postal"] == state].to_crs(crs)
            state_data.plot(ax=inset, color=missing_color, edgecolor="white", linewidth=0.5)
            available = state_data.loc[state_data[component].notna()]
            if len(available) > 0:
                available.plot(column=component, ax=inset, cmap="RdBu", vmin=-limit, vmax=limit,
                               edgecolor="white", linewidth=0.5)
            inset.set_axis_off()
            inset.text(0.5, -0.05, state, transform=inset.transAxes,
                       ha="center", fontsize=9, color="#555555")

        ax.set_title(f"{component}: {100 * variance[panel]:.1f}% of response variation",
                     fontsize=11, pad=20)
        colorbar_ax = ax.inset_axes([0.18, -0.17, 0.64, 0.045])
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu"),
                               cax=colorbar_ax, orientation="horizontal")
        colorbar.set_label("Mean component score within each state", fontsize=10)
        colorbar.ax.tick_params(labelsize=9)

    if has_missing:
        fig.legend(handles=[Patch(facecolor=missing_color,
                                  label=f"No data or fewer than {min_respondents} respondents")],
                   loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=9)
    set_figure_title(fig, title)
    save_figure(fig, filename)


cluster_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                  "#D55E00", "#56B4E9", "#8C564B", "#555555"]


def plot_cluster_selection(summary, selected_k, title, filename):
    """Display K-means inertia and converged GMM BIC across candidate counts.

    Args:
        summary (pandas.DataFrame): Candidate fits ordered by K, with Method,
            K, Inertia, BIC, and boolean Converged columns. Method labels must
            be "K-means" or "GMM"; include at least one converged GMM fit.
        selected_k (dict): Cluster counts to mark with dashed lines, keyed
            by "K-means" and "GMM". These choices are supplied by the caller.
        title (str): Figure title.
        filename (str): Output file stem without an extension.

    Returns:
        None. Saves a PDF and PNG, displays, and closes the figure through
        save_figure. Run from lab2/code with the figures directory present.

    Notes:
        Inertia is divided by 1,000. GMM values show BIC minus the minimum
        among converged candidates, divided by 1,000. The objectives are
        different and cannot be compared numerically across panels. The GMM
        axis spans K=0 to 100. This function does not choose an optimal K.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.77, bottom=0.22, wspace=0.30)
    kmeans = summary.loc[summary["Method"] == "K-means"]
    gmm = summary.loc[(summary["Method"] == "GMM") & summary["Converged"]]
    axes[0].plot(kmeans["K"], kmeans["Inertia"] / 1000, "o-", color="#0072B2")
    axes[0].axvline(selected_k["K-means"], color="#0072B2", linestyle="--", alpha=0.5)
    axes[0].set_title("K-means inertia analysis", fontsize=11)
    axes[0].set_ylabel("Within-cluster squared distance (thousands)")
    axes[0].set_xticks(kmeans["K"])

    axes[1].plot(gmm["K"], (gmm["BIC"] - gmm["BIC"].min()) / 1000,
                 "o-", color="#D55E00", markersize=3)
    axes[1].axvline(selected_k["GMM"], color="#D55E00", linestyle="--", alpha=0.5)
    axes[1].set_title("GMM BIC analysis", fontsize=11)
    axes[1].set_ylabel("BIC above best candidate (thousands)")
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[1].set_xlim(0, 100)
    axes[1].text(0.97, 0.85, f"Comparison uses K={selected_k['GMM']}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Number of clusters (K)")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.04, "Dashed lines mark the selected candidates.",
             ha="center", fontsize=9)
    set_figure_title(fig, title)
    save_figure(fig, filename)


def plot_cluster_projections(scores, labels, variance, components, title, filename):
    """Display fitted cluster assignments in a chosen pair of PCA coordinates.

    Args:
        scores (pandas.DataFrame): Respondent scores indexed by row ID, with
            columns named PC1, PC2, and so on.
        labels (pandas.DataFrame): One column per clustering method, with the
            same respondents as scores. Labels must be one-based integers
            supported by cluster_colors, currently 1 through 8.
        variance (sequence[float]): PCA explained variance fractions in
            component order, used to label the axes with percentages.
        components (tuple[int, int]): One-based numbers of the two PCs to show.
        title (str): Figure title.
        filename (str): Output file stem without an extension.

    Returns:
        None. Saves a PDF and PNG, displays, and closes the figure through
        save_figure. Run from lab2/code with the figures directory present.

    Notes:
        Drawing order is shuffled with seed 215, and scatter points are
        rasterized in the PDF. Colors identify method-specific cluster labels;
        equal numbers across methods do not establish corresponding groups.
        Overlap on two plotted PCs does not establish overlap in all fitted PCs.
    """
    fig, axes = plt.subplots(1, len(labels.columns), figsize=(13, 5.6), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.30, wspace=0.25)
    first, second = components
    shown = scores.sample(frac=1, random_state=215)
    for panel, name in enumerate(labels.columns):
        ax = axes.flat[panel]
        assignments = labels.loc[shown.index, name]
        colors = [cluster_colors[cluster - 1] for cluster in assignments]
        ax.scatter(shown[f"PC{first}"], shown[f"PC{second}"], c=colors,
                   s=3, alpha=0.30, edgecolors="none", rasterized=True)
        counts = labels[name].value_counts().sort_index()
        ax.set_title(f"{name}: K={len(counts)}, n={len(labels):,}", fontsize=11, fontweight="semibold")
        ax.set_xlabel(f"PC{first} ({100 * variance[first - 1]:.1f}%)")
        ax.set_ylabel(f"PC{second} ({100 * variance[second - 1]:.1f}%)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles = [Patch(facecolor=cluster_colors[cluster - 1],
                         label=f"C{cluster}: {count:,}") for cluster, count in counts.items()]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.23),
                  ncol=min(4, len(counts)), frameon=False, fontsize=9,
                  title="Cluster: respondents", title_fontsize=9)
    set_figure_title(fig, title)
    save_figure(fig, filename)


def plot_cluster_profiles(differences, counts, title, filename):
    """Show answer-frequency differences between fitted clusters and a baseline.

    Args:
        differences (pandas.DataFrame): Nonempty finite numeric table with
            answer features named QNNN__answer on rows and cluster IDs on
            columns. Values are within-cluster percentages minus the overall
            baseline, expressed in percentage points and computed by the caller.
        counts (pandas.Series): Respondent counts indexed by every cluster ID
            in differences.columns, used in the column labels.
        title (str): Figure title.
        filename (str): Output file stem without an extension.

    Returns:
        None. Saves a PDF and PNG, displays, and closes the figure through
        save_figure. Run from lab2/code with the figures directory present.

    Notes:
        Colors use a symmetric scale around zero; annotations round to whole
        percentage points. The caller selects which answer features to show.
        These describe fitted groups, rather than significance or causal effects.
    """
    values = differences.to_numpy(dtype=float)
    limit = max(5, 5 * np.ceil(np.abs(values).max() / 5))
    row_labels = []
    for feature in differences.index:
        question, answer = feature.split("__", 1)
        row_labels.append(fill(f"Q{int(question[1:])}: {answer}", width=48))
    longest_label = max(label.count("\n") + 1 for label in row_labels)
    height = max(0.60 * len(differences) + 2.7, 0.17 * longest_label * len(differences) / 0.66)
    fig, ax = plt.subplots(figsize=(12, height))
    fig.subplots_adjust(left=0.28, right=0.72, top=0.86, bottom=0.16)
    image = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    column_labels = [f"C{cluster}\nn={counts.loc[cluster]:,}" for cluster in differences.columns]
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_xticks(range(len(column_labels)), labels=column_labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=8, labelsize=9)
    for row, column in np.ndindex(values.shape):
        value = values[row, column]
        label = f"{value:+.0f}" if abs(value) >= 0.5 else "0"
        ax.text(column, row, label, ha="center", va="center", fontsize=9,
                color="white" if abs(value) > 0.65 * limit else "#222222")
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar_ax = fig.add_axes([0.30, 0.07, 0.40, 0.025])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label("Answer percentage minus all-respondent baseline (percentage points)", fontsize=9)
    set_figure_title(fig, title)
    save_figure(fig, filename)


def plot_cluster_agreement(counts, title, filename):
    """Compare K-means and GMM assignments using counts and row percentages.

    Args:
        counts (pandas.DataFrame): Nonempty contingency table for the same
            respondents, with K-means cluster IDs on rows and GMM cluster IDs
            on columns. Entries are nonnegative counts; row totals are positive.
        title (str): Figure title.
        filename (str): Output file stem without an extension.

    Returns:
        None. Saves a PDF and PNG, displays, and closes the figure through
        save_figure. Run from lab2/code with the figures directory present.

    Notes:
        Each cell gives its count and the percentage of that K-means cluster
        assigned to the GMM cluster. Colors share a 0-100 scale. Cluster IDs
        are arbitrary; matching numbers do not imply matching groups. This
        plot does not compute a Rand index or determine which method is better.
    """
    percentages = 100 * counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * counts.shape[1] + 2),
                                     max(4.5, 0.65 * counts.shape[0] + 2)))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.76, bottom=0.26)
    image = ax.imshow(percentages, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for row, column in np.ndindex(counts.shape):
        percent = percentages.iloc[row, column]
        ax.text(column, row, f"{counts.iloc[row, column]:,}\n({percent:.1f}%)",
                ha="center", va="center", fontsize=9,
                color="white" if percent > 65 else "#222222")
    ax.set_yticks(range(len(counts)), labels=[f"C{cluster}\nn={total:,}"
                 for cluster, total in counts.sum(axis=1).items()])
    ax.set_xticks(range(counts.shape[1]), labels=[f"C{cluster}" for cluster in counts.columns])
    ax.xaxis.tick_top()
    ax.set_ylabel("K-means cluster")
    fig.text(0.5, 0.85, "GMM cluster", ha="center", fontsize=11)
    ax.tick_params(length=0, pad=7)
    colorbar_ax = fig.add_axes([0.30, 0.12, 0.40, 0.025])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label("Percent of each K-means cluster assigned to a GMM cluster", fontsize=9)
    set_figure_title(fig, title)
    save_figure(fig, filename)


def plot_cluster_stability(summary, reference_counts, title, filename):
    """Save per-cluster Jaccard distributions across initialization and bootstrap runs.

    Args:
        summary (pandas.DataFrame): One row per experiment, run, and reference
            cluster, with Experiment, Cluster, and Jaccard columns. Supported
            experiment names are "Different starts" and "Bootstrap". Each
            experiment must contain finite overlaps for every plotted cluster.
        reference_counts (pandas.Series): Respondent counts indexed by the
            reference cluster labels. Index order determines the plotted
            cluster order, and counts are displayed below each cluster.
        title (str): Overall figure title.
        filename (str): Output file stem passed to save_figure.

    Returns:
        None. Saves a PDF and PNG, displays the figure, and closes it through
        save_figure.

    Notes:
        Each experiment receives a panel with boxplots and all individual run
        values. Seeded horizontal jitter separates overlapping points; numbers
        above the clusters show medians. These distributions describe stability
        relative to the matched reference groups, not confidence intervals or
        evidence that the groups are valid dialect regions.
    """
    experiments = list(summary["Experiment"].drop_duplicates())
    experiment_titles = {
        "Different starts": "Clusters across different random seeds",
        "Bootstrap": "Clusters across bootstrap samples",
    }
    colors = ["#0072B2", "#E69F00", "#009E73"]
    clusters = list(reference_counts.index)
    fig, axes = plt.subplots(1, len(experiments), figsize=(5.5 * len(experiments), 5.2),
                             squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.77, bottom=0.18, wspace=0.25)
    rng = np.random.default_rng(215)

    for ax, name, color in zip(axes[0], experiments, colors):
        selected = summary.loc[summary["Experiment"] == name]
        values = [selected.loc[selected["Cluster"] == cluster, "Jaccard"].to_numpy()
                  for cluster in clusters]
        positions = np.arange(1, len(clusters) + 1)
        boxes = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True,
                           showfliers=False, medianprops={"color": "#222222"})
        for position, scores, box in zip(positions, values, boxes["boxes"]):
            box.set_facecolor(color)
            box.set_alpha(0.25)
            ax.scatter(position + rng.uniform(-0.12, 0.12, len(scores)), scores,
                       color=color, s=20, alpha=0.7, zorder=3)
            ax.text(position, 1.025, f"{np.median(scores):.3f}", ha="center", fontsize=8)
        ax.set_xticks(positions, labels=[f"C{cluster}\nn={reference_counts.loc[cluster]:,}"
                                        for cluster in clusters])
        ax.set_ylim(-0.05, 1.07)
        ax.set_title(experiment_titles[name], fontsize=10, pad=12)
        ax.set_ylabel("Cluster overlap")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    set_figure_title(fig, title)
    save_figure(fig, filename)
