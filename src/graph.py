# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module providing a class for rendering graphs"""

from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter
from src.aux import cumsum, drop_duplicates, lag


class Graph:
    """
    This class generates plots (mainly bar and line plots) and configure their settings to achieve a better aesthetic.
    With this class, it is easier to have all plots in the same standard and similar appereance
    """

    def __init__(
        self,
        zero_indexed: bool = True,
        figure_size_inches=(20, 10),
        legend_out: bool = False,
        xlim: List[float] = None,
        ylim: List[float] = None,
        baseline: float = None,
        y_axis_upper_margin: float = 0.05,
    ) -> None:
        """
        - zero_indexed: whether the plot should be zero indexed. Hint: it should. Always. Yes, always
        - figure_size_inches: size of the figure
        - xlabel: text to label the X axis
        - ylabel: text to label the Y axis
        - legend_out: whether the legend of the colors should be 'inside' or outside the plot
        - xlim: limits of the y axis. if not defined (i.e. None) then will calculate based on the data
        - ylim: limits of the y axis. if not defined (i.e. None) then will calculate based on the data
        - title: title of the plot
        - baseline: baseline value to add as a horizonatal line
        - y_axis_upper_margin: % of max y value that the y axis should be. Should be a positive float. Ignored if ylim is not None
        """

        self.zero_indexed = zero_indexed
        self.figure_size_inches = figure_size_inches
        self.legend_out = legend_out
        self.xlim = xlim
        self.ylim = ylim
        self.baseline = baseline
        self.y_axis_upper_margin = y_axis_upper_margin

        # TODO: set them as inputs with defaults
        self.baseline_transparency = 0.5
        self.baseline_color = "black"
        self.baseline_linetype = "dashed"
        self.baseline_text_x_nudge = 1.02
        self.baseline_text_y_nudge = 1.01
        self.baseline_decimal_precision = 3
        self.baseline_text_size = 14

        self.floor_line_color = "gray"
        self.floor_line_transparency = 0.1

        self.font_family = "Avenir"
        self.legend_text_size = 14
        self.legend_weight = "normal"
        self.plot_style = "whitegrid"

        self.title_text_size = 24
        self.title_aligment = "left"

        self.bar_transparency = 0.6
        self.axis_tick_text_size = 14

    @staticmethod
    def _get_axis_limits(data: pd.Series):
        """
        Gets the lower and upper limits of the data
        """
        return np.min(data), np.max(data)

    def _get_facets_config(self, data: pd.DataFrame, x_axis: str, y_axis: str) -> Dict:
        """
        Get the dictionary to configure the facets based on the data and how the class was configured
        """

        # Get limits for the axis
        lower_x, upper_x = self._get_axis_limits(data[x_axis])
        lower_y, upper_y = self._get_axis_limits(data[y_axis])

        # corrects ylimts to be zero indexed if this was set up
        if lower_y > 0 and self.zero_indexed:
            lower_y = 0
        elif upper_y < 0 and self.zero_indexed:
            upper_y = 0

        # adds some margin from end of the plot to where the title begins
        upper_y = upper_y + np.abs(upper_y) * self.y_axis_upper_margin
        return {
            "legend_out": self.legend_out,
            "xlim": [lower_x, upper_x],
            "ylim": [lower_y, upper_y],
        }

    def set_baseline_value(self, new_baseline: float) -> None:
        """
        Defines new baseline to be shown in the plots, when a baseline is requested
        """
        self.baseline = new_baseline

    def _add_baseline(
        self, grid: sns.axisgrid.FacetGrid, x_axis: str, data: pd.DataFrame
    ) -> sns.axisgrid.FacetGrid:
        """
        Sets a horinzontal line on the indicated baseline and add a text over it indicating the value
        """
        grid.ax.axhline(
            self.baseline,
            ls=self.baseline_linetype,
            color=self.baseline_color,
            alpha=self.baseline_transparency,
        )
        min_x = np.min(data[x_axis])
        text_x_position = (
            0 if isinstance(min_x, str) else min_x * self.baseline_text_x_nudge
        )
        grid.ax.text(
            text_x_position,
            self.baseline * self.baseline_text_y_nudge,
            f"Baseline: {np.round(self.baseline, self.baseline_decimal_precision)}",
            fontsize=self.baseline_text_size,
            color=self.baseline_color,
            font=self.font_family,
        )
        return grid

    def _add_floor_line(self, grid: sns.axisgrid.FacetGrid) -> sns.axisgrid.FacetGrid:
        """
        Sets a horinzontal line on the the X axis, so guarantees that the plot is zero indexed
        """
        grid.ax.axhline(
            0.0, color=self.floor_line_color, alpha=self.floor_line_transparency
        )

    def _format_legends(self, grid: sns.axisgrid.FacetGrid):
        font = {
            "family": self.font_family,
            "weight": self.legend_weight,
            "size": self.legend_text_size,
        }
        legend = grid.ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                plt.setp(text, **font)

    def set_ax_standard(
        self,
        ax: sns.axisgrid.FacetGrid,
        xlabel: str,
        ylabel: str,
        title: str,
    ):
        """
        Applies formatting to the image size, font of the axis and the family font, title position etc
        Assumes the object being passed is a matplotlib.Axes
        """
        ax.set(xlabel=xlabel, ylabel=ylabel)

        ax.figure.set_size_inches(
            self.figure_size_inches[0], self.figure_size_inches[1]
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            fontsize=self.axis_tick_text_size,
            family=self.font_family,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=self.axis_tick_text_size,
            family=self.font_family,
        )
        plt.title(
            title,
            font=self.font_family,
            fontsize=self.title_text_size,
            loc=self.title_aligment,
        )
        return ax

    def _set_standard(
        self,
        data: pd.DataFrame,
        grid: sns.axisgrid.FacetGrid,
        x_axis: str,
        xlabel: str,
        ylabel: str,
        title: str,
    ):
        """
        Applies formatting to the image size, font of the axis and the family font, title position etc
        """
        grid = grid.set(xlabel=xlabel, ylabel=ylabel)

        if self.baseline is not None:
            self._add_baseline(grid, x_axis, data)

        if self.zero_indexed:
            self._add_floor_line(grid)

        self._format_legends(grid)
        grid.figure.set_size_inches(
            self.figure_size_inches[0], self.figure_size_inches[1]
        )
        grid.ax.set_xticks(grid.ax.get_xticks())
        grid.ax.set_yticks(grid.ax.get_yticks())
        grid.ax.set_xticklabels(
            grid.ax.get_xticklabels(),
            fontsize=self.axis_tick_text_size,
            family=self.font_family,
        )
        grid.ax.set_yticklabels(
            grid.ax.get_yticklabels(),
            fontsize=self.axis_tick_text_size,
            family=self.font_family,
        )

        plt.title(
            title,
            font=self.font_family,
            fontsize=self.title_text_size,
            loc=self.title_aligment,
        )
        return grid

    def line_plot(
        self,
        data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        hue: str = None,
        row=None,
        col=None,
        col_wrap=None,
        row_order=None,
        col_order=None,
        palette=None,
        xlabel="",
        ylabel="",
        x_format=None,
        y_format=None,
        legend=True,
        title: str = "",
        data_filter: str = None,
    ) -> sns.axisgrid.FacetGrid:
        """
        This method outputs a line plot using the data initially provided when defining the class instance
        The input parameters are almost all a subset of the possible inputs possible for the seaborn.relplot method
        The advantage of this method is just the fine tweaking on the plot to make it subjectively neater

        In addition to calling the seaborn.relplot method, this method adds the following 'features'
        - sets the style to the class style (default: whitegrid)
        - defines the image size (default: (20, 10))
        - adds the baseline horizontal line and text, if defined
        - adds the floor line, if class parameter is true
        - makes all the text use the same font (default: Avenir)
        - configure the legends parameters
        - adds the title for the plot
        - filter a subset of the provided data if data_filter is specified. If it is, uses the data_filter as the command for the 'query' in the data
        """

        # selects subset of the data if specifed
        data = data.copy() if data_filter is None else data.query(data_filter).copy()

        with sns.axes_style(self.plot_style):

            facets_config = self._get_facets_config(data, x_axis, y_axis)

            grid = sns.relplot(
                data=data,
                kind="line",
                x=x_axis,
                y=y_axis,
                hue=hue,
                facet_kws=facets_config,
                row=row,
                col=col,
                col_wrap=col_wrap,
                row_order=row_order,
                col_order=col_order,
                legend=legend,
                palette=palette,
            )

            grid = self._set_standard(data, grid, x_axis, xlabel, ylabel, title)
            if y_format == "%":
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            elif y_format is not None:
                plt.gca().yaxis.set_major_formatter(
                    FuncFormatter(lambda x, pos: y_format)
                )
            if x_format == "%":
                plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
            elif x_format is not None:
                plt.gca().xaxis.set_major_formatter(
                    FuncFormatter(lambda x, pos: x_format)
                )

        return grid

    def bar_plot(
        self,
        data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        hue: str = None,
        color_dodge=True,
        row=None,
        col=None,
        col_wrap=None,
        row_order=None,
        col_order=None,
        palette=None,
        xlabel="",
        ylabel="",
        x_format=None,
        y_format=None,
        legend=True,
        title: str = "",
        data_filter: str = None,
    ) -> sns.axisgrid.FacetGrid:
        """
        This method outputs a line plot using the data initially provided when defining the class instance
        The input parameters are almost all a subset of the possible inputs possible for the seaborn.relplot method
        The advantage of this method is just the fine tweaking on the plot to make it subjectively neater

        In addition to calling the seaborn.relplot method, this method adds the following 'features'
        - sets the style to the class style (default: whitegrid)
        - defines the image size (default: (20, 10))
        - adds the baseline horizontal line and text, if defined
        - adds the floor line, if class parameter is true
        - makes all the text use the same font (default: Avenir)
        - configure the legends parameters
        - adds the title for the plot
        - filter a subset of the provided data if data_filter is specified. If it is, uses the data_filter as the command for the 'query' in the data
        """

        # selects subset of the data if specifed
        data = data.copy() if data_filter is None else data.query(data_filter).copy()

        with sns.axes_style(self.plot_style):

            grid = sns.catplot(
                data=data,
                kind="bar",
                x=x_axis,
                y=y_axis,
                hue=hue,
                dodge=color_dodge,
                row=row,
                col=col,
                col_wrap=col_wrap,
                row_order=row_order,
                col_order=col_order,
                palette=palette,
                legend=legend,
                legend_out=self.legend_out,
                alpha=self.bar_transparency,
            )

            grid = self._set_standard(data, grid, x_axis, xlabel, ylabel, title)
            if y_format == "%":
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            elif y_format is not None:
                plt.gca().yaxis.set_major_formatter(
                    FuncFormatter(lambda x, pos: x_format)
                )
            if x_format == "%":
                plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
            elif x_format is not None:
                plt.gca().xaxis.set_major_formatter(
                    FuncFormatter(lambda x, pos: x_format)
                )

        return grid

    def grid_plot(
        self,
        data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        annot_values: str,
        fmt: str = ".1%",
        cmap: str = "Blues",
        cbar: bool = False,
        annotate: bool = True,
        annot_font_size: int = 20,
        xlabel="",
        ylabel="",
        title: str = "",
        data_filter: str = None,
    ) -> matplotlib.figure.Figure:
        """
        This method is based on the sns.heatmap method, and is meant to plot the values of interest as a 2D matrix

        In addition to calling the seaborn.relplot method, this method adds the following 'features'
        - sets the style to the class style (default: whitegrid)
        - defines the image size (default: (20, 10))
        - adds the baseline horizontal line and text, if defined
        - adds the floor line, if class parameter is true
        - makes all the text use the same font (default: Avenir)
        - configure the legends parameters
        - adds the title for the plot
        - filter a subset of the provided data if data_filter is specified. If it is, uses the data_filter as the command for the 'query' in the data

        Inputs
            data: Data to be plotted
            x_axis: the column that should be in the X-axis
            y_axis: the column that should be in the Y-axis
            annot_values: the column whose values should be shown in the cells
            fmt: formatting to be applied to the annotations
            cmap: color map for the values in the annotations
            cbar: whether to show the color bar that indicates what each color mean numerically
            annotate: whether we want to annotate
            annot_font_size: the annotation font size
            xlabel: label to be shown in the X-axis
            ylabel: label to be shown in the Y-axis
            title: str = "",
            data_filter: str = None,
        """

        # selects subset of the data if specifed
        data = data.copy() if data_filter is None else data.query(data_filter).copy()
        data = data.pivot(index=y_axis, columns=x_axis, values=annot_values)

        annot_fmt = {"fontsize": annot_font_size}

        with sns.axes_style(self.plot_style):

            ax = sns.heatmap(
                data=data,
                annot=annotate,
                fmt=fmt,
                cmap=cmap,
                cbar=cbar,
                linewidths=0.5,
                linecolor="gray",
                annot_kws=annot_fmt,
            )

            ax = self.set_ax_standard(ax, xlabel, ylabel, title)

        return ax.get_figure()

    def heatmap_plot(
        self,
        pivoted_data: pd.DataFrame,
        fmt: str = ".1%",
        cmap: str = "crest",
        cbar: bool = False,
        annotate: bool = True,
        annot_font_size: int = None,
        xlabel="",
        ylabel="",
        title: str = "",
    ) -> matplotlib.figure.Figure:
        """
        This method is based on the sns.heatmap method, and is meant to plot the values of interest as a 2D matrix

        In addition to calling the seaborn.relplot method, this method adds the following 'features'
        - sets the style to the class style (default: whitegrid)
        - defines the image size (default: (20, 10))
        - adds the baseline horizontal line and text, if defined
        - adds the floor line, if class parameter is true
        - makes all the text use the same font (default: Avenir)
        - configure the legends parameters
        - adds the title for the plot
        - filter a subset of the provided data if data_filter is specified. If it is, uses the data_filter as the command for the 'query' in the data

        Inputs
            pivoted_data: Data to be plotted. Should already be pivoted
            fmt: formatting to be applied to the annotations
            cmap: color map for the values in the annotations
            cbar: whether to show the color bar that indicates what each color mean numerically
            annotate: whether we want to annotate
            annot_font_size: the annotation font size
            xlabel: label to be shown in the X-axis
            ylabel: label to be shown in the Y-axis
            title: str = "",
            data_filter: str = None,
        """

        # add a mask to make the matrix (upper) triangular, cleaning it up
        mask = np.ones_like(pivoted_data, dtype=bool)
        mask[np.triu_indices_from(mask)] = False

        # adapt annot_font_size depending on size of the data if None, else use the one defined
        annot_font_size = (
            200 / len(pivoted_data) if annot_font_size is None else annot_font_size
        )
        annot_fmt = {"fontsize": annot_font_size}

        with sns.axes_style(self.plot_style):

            ax = sns.heatmap(
                data=pivoted_data,
                annot=annotate,
                mask=mask,
                fmt=fmt,
                cmap=cmap,
                cbar=cbar,
                linewidths=0.5,
                linecolor="white",
                annot_kws=annot_fmt,
            )

            ax = self.set_ax_standard(ax, xlabel, ylabel, title)

        return ax.get_figure()


class InteractiveChart:
    """
    This class is mostly an interface to plotly, but with some standards for the visualizations (e.g. font, font size, image shape) pre-defined
    With this class, it is easier to have all plots in the same standard and similar appereance
    """

    def __init__(
        self,
        legend_out: bool = False,
        font: str = "Avenir",
        txt_size: int = 14,
        title_size: int = 18,
        img_shape=(1200, 600),
        bargap: float = 0.1,
        bar_opacity: float = 0.8,
        pad_percentage: float = 0.01,
    ):
        """
        - legend_out: whether the legend of the colors should be 'inside' or outside the plot
        -
        """

        self.legend_out = legend_out
        self.font = font
        self.title_size = title_size
        self.txt_size = txt_size
        self.img_shape = img_shape
        self.bargap = bargap
        self.bar_opacity = bar_opacity
        self.pad_percentage = pad_percentage

    def _apply_fonts_standards(self, fig):
        """
        Apply font standards to all text parts of the
        """
        fig.update_layout(font=dict(family=self.font, size=self.txt_size))
        fig.update_layout(title_font=dict(family=self.font, size=self.title_size))
        fig.update_layout(xaxis_title="", yaxis_title="")
        fig.update_layout(autosize=True)

    def _apply_figure_standards(self, fig):
        """
        Apply standard regarding
            - background color of the chart: lightest gray
            - background color around chart: white
            - color of the X- and Y-axis: light gray
            - image size and shape
        """
        fig.update_layout(plot_bgcolor="#F1F1F1", paper_bgcolor="white")
        fig.update_layout(
            xaxis=dict(linecolor="lightgray"), yaxis=dict(linecolor="lightgray")
        )
        fig.update_layout(width=self.img_shape[0], height=self.img_shape[1])

    def _apply_lines_standards(self, fig):
        fig.update_layout(bargap=self.bargap)

    def _apply_standards(self, fig) -> None:
        self._apply_fonts_standards(fig)
        self._apply_figure_standards(fig)
        self._apply_lines_standards(fig)

    @staticmethod
    def _transform_yaxis_in_perc(fig, precision: int = 0):
        fig.update_layout(yaxis_tickformat=f".{precision}%")

    @staticmethod
    def _transform_yaxis_in_dollars(fig, precision: int = 0):
        fig.update_layout(yaxis=dict(tickformat=f"$.{precision}"))

    def _transform_yaxis_tickformat(self, fig, tickformat: str, precision: int) -> None:
        if tickformat == "":
            fig.update_layout(yaxis_tickformat=f".{precision}")
        elif "$" in tickformat:
            self._transform_yaxis_in_dollars(fig, precision)
        elif "%" in tickformat:
            self._transform_yaxis_in_perc(fig, precision)
        else:
            raise ValueError(
                f"Tick formart [{tickformat}] was not recognized. Only possible values are an empty string for numbers, $ for dollars, and % for percentages"
            )

    @staticmethod
    def _append_txt_to_yaxis_labels(fig, precision: int = 0):
        fig.update_layout(yaxis=dict(tickformat=f"$.{precision}"))

    @staticmethod
    def _add_title(fig, title: str) -> None:
        fig.update_layout(title=title)

    def line_chart(
        self,
        data: pd.DataFrame,
        xaxis: str,
        yaxis: str,
        title: str = "",
        precision: int = 0,
        tickformat: str = "",
    ):
        fig = px.line(data, x=xaxis, y=yaxis, title=title)
        self._transform_yaxis_tickformat(fig, tickformat, precision)
        self._apply_standards(fig)
        self._add_title(fig, title)
        fig.update_traces(line=dict(width=1))
        return fig

    def bar_chart(
        self,
        data: pd.DataFrame,
        xaxis: str,
        yaxis: str,
        color=None,
        title: str = "",
        precision: int = 0,
        tickformat: str = "",
    ):
        fig = px.bar(data, x=xaxis, y=yaxis, color=color, title=title)
        self._transform_yaxis_tickformat(fig, tickformat, precision)
        self._apply_standards(fig)
        self._add_title(fig, title)
        return fig

    def histogram_chart(
        self,
        data: pd.DataFrame,
        xaxis: str,
        yaxis: str,
        title: str = "",
        precision: int = 0,
    ):
        fig = px.histogram(
            data, x=xaxis, y=yaxis, nbins=data.shape[0], opacity=self.bar_opacity
        )
        self._apply_standards(fig)
        self._transform_yaxis_in_perc(fig, precision)
        self._add_title(fig, title)
        return fig

    def flow_chart(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        values: str,
        title: str = "",
    ):
        """
        This method creates a dataframe of flow from 'source' to 'target' based on a given quantity
        It is expected that the data is already orders in a way that it wants to be displayed. If not,
        the quality of the chart will be compromised, as the classes in 'source' to the equivalent 'targets'
        won't be in the same order
        """
        # Define the nodes and links for the Sankey diagram
        sources = data[source].to_list()
        targets = data[target].to_list()
        quantity = data[values].to_list()

        nodes = drop_duplicates(
            sources + targets
        )  # find the unique nodes in both source and targets
        n_nodes = len(nodes)

        # get the number of colors we need based unique classes. Need to duplicate to make equivalent
        # classes in [sources] and [targets] have the same colors
        palette = px.colors.qualitative.Plotly[0:n_nodes] * 2

        # we reference the index of the nodes, not the 'names'
        sources_idxs = [nodes.index(x) for x in sources]
        targets_idxs = [(nodes.index(x) + n_nodes) for x in targets]

        def get_horizontal_positions(n_nodes: int) -> List[float]:
            """
            Return the horizontal positions for the nodes. First values refers to [sources] and last to [targets]
            """
            return [0.001] * n_nodes + [
                0.999
            ] * n_nodes  # they cannot be 0 and 1 because it overlaps with title

        def get_vertical_positions(
            sources: list, targets: list, quantity: list, classes_gap: float = 0.05
        ) -> List[float]:
            """
            Return the vertical positions for the nodes. First values refers to [sources] and last to [targets]
            """
            y_positions_source = {}
            y_positions_target = {}
            # calculate the vertical size of each node
            for i, source in enumerate(sources):
                y_positions_source[source] = (
                    y_positions_source.get(source, 0) + quantity[i]
                )
                y_positions_target[targets[i]] = (
                    y_positions_target.get(targets[i], 0) + quantity[i]
                )

            # Calculate where the node's vertical position should end
            y_positions_source = cumsum(
                list(y_positions_source.values()), constant_delta=classes_gap
            )
            y_positions_target = cumsum(
                list(y_positions_target.values()), constant_delta=classes_gap
            )

            # Lag to get next position, because plotly receives where nodes begin
            y_positions_source = lag(y_positions_source, 1, coalesce=classes_gap)
            y_positions_target = lag(y_positions_target, 1, coalesce=classes_gap)

            return y_positions_source + y_positions_target

        def get_nodes_positions(
            sources: list,
            targets: list,
            quantity: list,
            n_nodes: int,
            classes_gap: float = 0.05,
        ) -> (List[float], List[float]):
            return get_horizontal_positions(n_nodes), get_vertical_positions(
                sources, targets, quantity, classes_gap
            )

        x_positions, y_positions = get_nodes_positions(
            sources, targets, quantity, n_nodes
        )
        # pad between classes needs to consider #classes and size of image
        pad_size = self.img_shape[1] * (n_nodes - 1) * self.pad_percentage

        # replicate nodes with the same values. Necessary because each
        # value in nodes represent a node. So we have to create 2 nodes with the same names
        nodes = nodes * 2

        node = dict(
            pad=pad_size,
            thickness=20,
            line=dict(color="grey", width=0.5),
            color=palette,
            x=x_positions,
            y=y_positions,
            label=nodes,
        )
        link = dict(source=sources_idxs, target=targets_idxs, value=quantity)

        # replicate nodes so that the firsts represent the sources and the others the targets
        nodes = nodes * 2
        # # Create the Sankey diagram
        fig = go.Figure(go.Sankey(arrangement="snap", node=node, link=link))
        # Update the layout of the figure
        self._apply_standards(fig)
        self._add_title(fig, "User Flow Between Classes")
        return fig


def save_plot(fig, file_path: str, dpi: int = 200) -> None:
    """
    Save figure in the defined location
    Inputs
        fig: Either a [plotly.graph_objs._figure.Figure] or [seaborn.axisgrid.FacetGrid],
        file_path: string containing path and name of the file it should be save as. Ex: images/my_image.png or /Users/Documents/my_image.jpeg
        dpi: dots per inches. Only for static images
    """
    if isinstance(fig, sns.axisgrid.FacetGrid):
        fig.savefig(file_path, dpi=dpi)
    elif isinstance(fig, plotly.graph_objs._figure.Figure):
        # assumes image display of 4k (3840 x 2160) pixels and (24.5 x 14.6) inches.
        current_dpi = 2160 / 14.6
        rescaled_height = fig.layout.height * dpi / current_dpi
        rescaled_width = fig.layout.width * dpi / current_dpi
        fig.write_image(file_path, height=rescaled_height, width=rescaled_width)
    elif isinstance(fig, matplotlib.figure.Figure):
        fig.savefig(file_path, bbox_inches="tight", dpi=dpi)
    else:
        raise TypeError(
            f"Input [fig] is of type {type(fig)} is not a valid type. It must be either seaborn.axisgrid.FacetGrid or plotly.graph_objs._figure.Figure"
        )
