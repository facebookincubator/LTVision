# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module providing a class for rendering graphs"""

from typing import List, Dict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
# TODO: remove hard-coded constants inside the code


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

        for text in grid.ax.legend().get_texts():
            plt.setp(text, **font)

    def set_ax_standard(self,
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
        fmt: str='.1%',
        cmap: str='Blues',
        cbar: bool=False,
        annotate: bool=True,
        annot_font_size: int=20,
        xlabel="",
        ylabel="",
        title: str = "",
        data_filter: str = None,
    ) -> sns.axisgrid.FacetGrid:
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

        annot_fmt = {'fontsize': annot_font_size}

        with sns.axes_style(self.plot_style):

            ax = sns.heatmap(
                data=data,
                annot=annotate,
                fmt=fmt,
                cmap=cmap,
                cbar=cbar,
                linewidths=0.5,
                linecolor='gray',
                annot_kws=annot_fmt
                )
            
            ax = self.set_ax_standard(ax, xlabel, ylabel, title)
            
        return ax
