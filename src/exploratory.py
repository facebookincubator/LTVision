# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module providing a class for initial analysis"""
import math
import os
from itertools import product
from typing import Dict, List

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import (
    is_any_real_numeric_dtype,
    is_datetime64_any_dtype,
    is_dtype_equal,
    is_object_dtype,
)
from src.graph import Graph, InteractiveChart


sns.set_style("whitegrid")


class LTVexploratory:
    """This class helps to perform som initial analysis"""

    def __init__(
        self,
        data_customers: pd.DataFrame,
        data_events: pd.DataFrame,
        uuid_col: str = "UUID",
        registration_time_col: str = "timestamp_registration",
        event_time_col: str = "timestamp_event",
        event_name_col: str = "event_name",
        value_col: str = "purchase_value",
        segment_feature_cols: List[str] = None,
        rounding_precision: int = 5,
    ):
        self.data_customers = data_customers
        self.data_events = data_events
        self._period = 7
        self._period_for_ltv = 7 * 10
        self.graph = Graph()
        self.interactive_chart = InteractiveChart()
        self.rounding_precision = rounding_precision

        # store information about the columns of the dataframes
        self.uuid_col = uuid_col
        self.registration_time_col = registration_time_col
        self.event_time_col = event_time_col
        self.event_name_col = event_name_col
        self.value_col = value_col
        self.segment_feature_cols = (
            [] if segment_feature_cols is None else segment_feature_cols
        )
        # run auxiliar methods
        self._validate_datasets()
        self._prep_df()

    def _validate_datasets(self) -> None:
        """
        This method perform the following checks for the input datasets:
        customers dataset:
            -
        customers dataset:
            - customer-id column is string or object type
            - time of event column is datetime

        events dataset:
            - customer-id column is string or object type
            - event name column is string or object type
            - time of event column is datetime
            - purchase value is int or float type

        consistency checks
            - customer-id columns of both datasets are of the same type
            - time of event columns of both datasets are of the same type
        """

        # customers dataset checks
        assert isinstance(
            self.data_customers[self.uuid_col].dtype, pd.StringDtype
        ) or is_object_dtype(
            self.data_customers[self.uuid_col]
        ), f"The column [{self.uuid_col}] referencing to the customer-id in the customers dataset was expected to be of data pd.StringDtype or object. But it is of type {self.data_customers[self.uuid_col].dtype}"
        assert is_datetime64_any_dtype(
            self.data_customers[self.registration_time_col]
        ), f"The column [{self.registration_time_col}] referencing to the registrationtime in the customers dataset was expected to be of type [datetime]. But it is of type {self.data_customers[self.registration_time_col].dtype}"

        # events dataset checks
        assert isinstance(
            self.data_events[self.uuid_col].dtype, pd.StringDtype
        ) or is_object_dtype(
            self.data_events[self.uuid_col]
        ), f"The column [{self.uuid_col}] referencing to the customer-id in the events dataset was expected to be of data pd.StringDtype or object. But it is of type {self.data_events[self.uuid_col].dtype}"
        assert is_datetime64_any_dtype(
            self.data_events[self.event_time_col]
        ), f"The column [{self.event_time_col}] referencing to the time in the events dataset was expected to be of type [datetime]. But it is of type {self.data_events[self.event_time_col].dtype}"
        assert is_any_real_numeric_dtype(
            self.data_events[self.value_col]
        ), f"The column [{self.value_col}] referencing value of a transaction in the events dataset was expected to be of numeric. But it is of type {self.data_events[self.value_col].dtype}"

        # consistency checks
        assert is_dtype_equal(
            self.data_customers[self.uuid_col], self.data_events[self.uuid_col]
        ), f"The customer-id columns of the two input datasets are not the same. In the customers dataset it is of type [{self.data_customers[self.uuid_col].dtype}], while in the events dataset it is of type [{self.data_customers[self.uuid_col].dtype}]"
        assert is_dtype_equal(
            self.data_customers[self.registration_time_col],
            self.data_events[self.event_time_col],
        ), f"The timestamp columns of the two input datasets are not the same. In the customers dataset it is of type [{self.data_customers[self.registration_time_col].dtype}], while in the events dataset it is of type [{self.data_customers[self.event_time_col].dtype}]"

        # check the date range of the data, warn if it is too short
        if (
            self.data_customers[self.registration_time_col].max()
            - self.data_customers[self.registration_time_col].min()
        ).days < 90:
            print(
                "Warning: The date range of the customers data is too short. The analysis may not be accurate and some plots may not be generated."
            )
        if (
            self.data_events[self.event_time_col].max()
            - self.data_events[self.event_time_col].min()
        ).days < 90:
            print(
                "Warning: The date range of the events data is too short. The analysis may not be accurate and some plots may not be generated."
            )

    def _prep_df(self) -> None:
        # Left join customers and events data, so that you have a dataframe with all customers and their events (if no event, then timestamp is null)
        # just select some columns to make it easier to understand what is the
        # information that is used and avoid join complications caused by the
        # name of other columns
        self.joined_df = pd.merge(
            self.data_customers,
            self.data_events,
            on=self.uuid_col,
            how="left",
            suffixes=["_registration", "_event"],
        )[
            [
                self.uuid_col,
                self.registration_time_col,
                self.event_time_col,
                self.event_name_col,
                self.value_col,
            ]
            + self.segment_feature_cols
        ]

        # Calculate how many days since install an event happend
        self.joined_df["days_since_registration"] = self.joined_df.apply(
            lambda row: (
                row[self.event_time_col] - row[self.registration_time_col]
            ).days,
            axis=1,
        )

    # Analysis Plots
    def summary(self):
        data_customers = self.joined_df[
            [self.uuid_col, self.registration_time_col]
        ].drop_duplicates()
        print(
            f"""
    **Customer Data Table**
    - Date start:    {data_customers[self.registration_time_col].min()}
    - Date end:      {data_customers[self.registration_time_col].max()}
    - Period:        {(data_customers[self.registration_time_col].max()-data_customers[self.registration_time_col].min()).days/365:.2f} years
    - Total Customers: {data_customers[self.uuid_col].nunique():,d} customers.
    - Total Events: {data_customers.shape[0]:,d} events.
    """
        )
        data_events = self.joined_df[~self.joined_df[self.event_time_col].isnull()]
        unique_event_types = data_events[self.event_name_col].nunique()
        event_list = list(data_events[self.event_name_col].unique())

        print(
            f"""
    **Event Data Table**
    - Date start:    {data_events[self.event_time_col].min()}
    - Date end:      {data_events[self.event_time_col].max()}
    - Period:        {(data_events[self.event_time_col].max()-data_events[self.event_time_col].min()).days/365:.2f} years
    - Total Customers: {data_events[self.uuid_col].nunique():,d} customers.
    - Total Events: {data_events.shape[0]:,d} events.
    - Unique Event Types: {unique_event_types}.
    - Event list: {event_list}
    - Average Events per Customer: {(data_events.shape[0] / data_events[self.uuid_col].nunique()):.2f} events/customer.
    """
        )

    # All plot methods
    def plot_customers_intersection(self):
        """
        Plot the interection between customers in the two input data
        We expect that all customers in events data are also in customers data.
        The inverse can be true, as there may be customers who never sent an event
        """

        # Get uuids from each input data
        customers_uuids = pd.DataFrame(
            {self.uuid_col: self.data_customers[self.uuid_col].unique()}
        )
        customers_uuids["customers"] = "Present in Customers"
        events_uuids = pd.DataFrame(
            {self.uuid_col: self.data_events[self.uuid_col].unique()}
        )
        events_uuids["events"] = "Present in Events"

        # Calculate how many customers are in each category
        cross_uuid = pd.merge(
            customers_uuids, events_uuids, on=self.uuid_col, how="outer"
        )
        cross_uuid["customers"] = cross_uuid["customers"].fillna(
            "Not in customers")
        cross_uuid["events"] = cross_uuid["events"].fillna("Not in Events")
        cross_uuid = (
            cross_uuid.groupby(["customers", "events"])[self.uuid_col]
            .count()
            .reset_index()
        )

        # Create a dataframe containing all combinations for the visualization
        complete_data = pd.DataFrame(
            {
                "customers": [
                    "Present in Customers",
                    "Not in Customers",
                    "Present in Customers",
                    "Not in Customers",
                ],
                "events": [
                    "Present in Events",
                    "Present in Events",
                    "Not in Events",
                    "Not in Events",
                ],
            }
        )
        complete_data = pd.merge(
            complete_data, cross_uuid, on=["customers", "events"], how="left"
        )
        complete_data = complete_data.fillna(0)
        complete_data[self.uuid_col] = complete_data[self.uuid_col] / np.sum(
            complete_data[self.uuid_col]
        )
        fig = self.graph.grid_plot(
            complete_data, "customers", "events", self.uuid_col)
        return fig, complete_data

    def plot_purchases_distribution(
        self, days_limit: int, truncate_share: float = 0.99
    ):
        """
        Plots an histogram of the number of people by how many purchases they had until [days_limit] after their registration
        We need [days_limit] to ensure that we only select customers with the same opportunity window (i.e. all customers are at least
        [days_limit] days old at the time the data was collected).
        )
        Input
            days_limit: number of days of the event since registration.
            truncate_share: share of total customers/revenue until where the plot shows values
        """
        # Select only customers that are at least [days_limit] days old
        end_events_date = self.joined_df[self.event_time_col].max()
        data = self.joined_df[
            (end_events_date -
             self.joined_df[self.registration_time_col]).dt.days
            >= days_limit
        ].copy()

        # Remove customers who never had a purchase and ensure all customers
        # have the same opportunity window
        data = data[
            (data[self.event_time_col] -
             data[self.registration_time_col]).dt.days
            <= days_limit
        ]

        # Count how many purchase customers had (defined by value > 0) and then
        # how many customers are in each place
        data = data[data[self.value_col] > 0]
        data = (
            data.groupby(self.uuid_col)[self.value_col]
            .agg(["sum", "count"])
            .reset_index()
        )
        data = data.drop(self.uuid_col, axis=1)
        data = data.rename(columns={"count": "purchases"})
        data = data.groupby("purchases")["sum"].agg(
            ["sum", "count"]).reset_index()
        # Calculate the share
        data["sum"] = data["sum"] / data["sum"].sum()
        data["count"] = data["count"] / data["count"].sum()
        # Find treshold truncation
        customers_truncation = data["count"].cumsum() <= truncate_share
        # plot distribution by customers
        grid = self.graph.bar_plot(
            data[customers_truncation],
            x_axis="purchases",
            y_axis="count",
            xlabel="Number of purchases",
            ylabel="Share of customers",
            y_format="%",
            title=f"Customer Purchase Frequency (based on the first {days_limit} days after registration)",
        )
        ax = grid.ax
        # Add percentage labels on top of each bar
        for bar in ax.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height * 100:.1f}%",  # Convert to percentage
                ha="center",
                va="bottom",
            )
        return grid, data

    def plot_revenue_pareto(self, days_limit: int, granularity: int = 1000):
        """
        Plots the - cumulative - share of revenue (Y) versus the share of customers (X), with customers ordered by revenue in descending order
        This plots how concentrated the revenue is. Base of customers is only of spending customers (so customers who never spent anything are ignored)
        )
        Input
            days_limit: number of days of the event since registration.
            granularity: number of steps in the plot
        """
        # Select only customers that are at least [days_limit] days old
        end_events_date = self.joined_df[self.event_time_col].max()
        data = self.joined_df[
            (end_events_date -
             self.joined_df[self.registration_time_col]).dt.days
            >= days_limit
        ].copy()

        # Remove customers who never had a purchase and ensure all customers
        # have the same opportunity window
        data = data[
            (data[self.event_time_col] -
             data[self.registration_time_col]).dt.days
            <= days_limit
        ]

        # Count how many purchase customers had (defined by value > 0) and then
        # how many customers are in each place
        data = data[data[self.value_col] > 0]
        data = (
            data.groupby(self.uuid_col)[self.value_col]
            .sum()
            .reset_index()
            .sort_values(self.value_col, ascending=False)
        )

        # Calculate total revenue and group customers together based on the
        # granilarity
        total_revenue = data[self.value_col].sum()
        total_customers = data.shape[0]
        data["cshare_customers"] = [
            (i + 1) / total_customers for i in range(total_customers)
        ]
        data["cshare_revenue"] = data[self.value_col].cumsum() / total_revenue
        data["group"] = data["cshare_customers"].apply(
            lambda x: np.ceil(x * granularity)
        )
        data = (
            data.groupby("group")[["cshare_customers", "cshare_revenue"]]
            .max()
            .reset_index()
        )
        # Create the plot using the existing graphing method
        fig = self.graph.line_plot(
            data,
            x_axis="cshare_customers",
            y_axis="cshare_revenue",
            xlabel="Share of paying customers",
            ylabel="Share of revenue",
            x_format="%",
            y_format="%",
            title=f"Top spenders' cumulative contribution to total revenue (based on the first {days_limit} days since registration)",
        )
        # Highlight % revenue contribution of top 5%, 10%, 20% of spenders
        top_spenders = [0.05, 0.1, 0.2]
        for spender in top_spenders:
            revenue_contribution = data.loc[
                data["cshare_customers"] <= spender, "cshare_revenue"
            ].max()
            for ax in fig.axes.flat:  # Iterate over all axes in the FacetGrid
                # Find the intersection point of the vertical line and the
                # curve
                intersection_x = spender
                intersection_y = revenue_contribution
                # Plot the vertical line up to the intersection point
                ax.plot(
                    [intersection_x, intersection_x],
                    [0, intersection_y],
                    color="gray",
                    linestyle="--",
                )
                ax.text(
                    spender,
                    revenue_contribution + 0.03,
                    f"{revenue_contribution*100:.1f}%",
                    ha="center",
                    fontsize=18,
                    color="navy",
                )
        # Adjust the x-axis ticks to every 10%
        for ax in fig.axes.flat:  # Iterate over all axes in the FacetGrid
            ax.set_xticks(np.arange(0, 1.1, 0.1))
            ax.set_xticks(list(ax.get_xticks()) + [0.05])  # Add a tick at 5%
        return fig, data
        """
        fig = self.graph.line_plot(
            data,
            x_axis="cshare_customers",
            y_axis="cshare_revenue",
            xlabel="Share of paying customers",
            ylabel="Share of revenue",
            x_format="%",
            y_format="%",
            title=f"Share of all revenue of the first {days_limit} days after customer registration (Y) versus share of paying customers",
        )
        return fig, data
        """

    def plot_customers_histogram_per_conversion_day(
        self, days_limit: int = 60, optimization_window: int = 7, truncate_share=1.0
    ) -> None:
        """
        Plots the distribution of all customers that converted until (days_limit) days after registration per conversion day
        Inputs:
            days_limit: number of days of the event since registration.
            optimization_window: the number of days since registration of a customer that matters for the optimization of campaigns
            truncate_share: the total share of purchasing customers that the histogram includes
        """

        end_events_date = self.joined_df[self.event_time_col].max()
        data = self.joined_df[
            (end_events_date -
             self.joined_df[self.registration_time_col]).dt.days
            >= days_limit
        ].copy()

        # hard cap the histogram to 60 days
        if days_limit > 60:
            print("Warning: the histogram is based on the first 60 days of the data.")
            days_limit = 60

        # Remove customers who never had a purchase and ensure all customers
        # have the same opportunity window
        data = data[data[self.value_col] > 0]
        data["dsi"] = (
            (data[self.event_time_col] -
             data[self.registration_time_col]).dt.days
        ).fillna(0)
        data = data[data["dsi"] <= days_limit]

        # calculate data for the histogram
        data = (
            data.groupby(self.uuid_col)["dsi"]
            .min()
            .reset_index()
            .groupby("dsi")[self.uuid_col]
            .count()
            .reset_index()
        )

        # calculate the share of customers instead of absolute numbers and
        # numbers for the title
        data[self.uuid_col] = data[self.uuid_col] / data[self.uuid_col].sum()
        data = data[data[self.uuid_col].cumsum() < truncate_share]

        share_customers_within_window = data[data["dsi"] <= optimization_window][
            self.uuid_col
        ].sum()
        title = f"% of customers purchased on different days after registration\n{100*share_customers_within_window:.1f}% of first purchases happened within the first {optimization_window} days after registration\n"

        # Create the bar plot using the existing graphing method
        fig = self.graph.bar_plot(
            data,
            x_axis="dsi",
            y_axis=self.uuid_col,
            xlabel="Days Since Registration",
            ylabel="% Purchased Customers",
            y_format="%",
            title=title,
        )

        if days_limit >= 7:
            max_y = math.ceil(data[self.uuid_col].max() * 100) / 100
            plt.fill_between(
                [-0.5, 7.5], [0, 0], [max_y, max_y], alpha=0.3, color="lightblue"
            )
            # Set x-axis limits to prevent shifting
            plt.xlim(data["dsi"].min() - 0.5, data["dsi"].max())

        # Add percentage labels on top of each bar
        for _, row in data.iterrows():
            plt.text(
                row["dsi"],
                row[self.uuid_col],
                f"{row[self.uuid_col]*100:.1f}%",
                ha="center",
                va="bottom",
            )
        return fig, data
        """
        fig = self.graph.bar_plot(
            data,
            x_axis="dsi",
            y_axis=self.uuid_col,
            xlabel="",
            ylabel="",
            y_format="%",
            title=title,
        )
        return fig, data
        """

    def plot_early_late_revenue_correlation(
        self, days_limit: int, optimization_window: int = 7, interval_size: int = None
    ) -> None:
        """
        Calculates and plots correlation between customer-level revenue
        The correlation is between the revenue in the first [optimization_window] days after registration and the following [days_limit - optimization_window] days
        This can be useful to decide what number of days should define the 'Lifetime Value Window' of a customer

        Inputs
             - days_limit: max number of days after registration to be considered in the analysis
             - optimization_window: number of days from registration that the optimization of the marketing campaigns are operated
             - interval_size: number of days between two values shown in the correlation matrix. If None, the method finds the best interval based in the data size
        """
        # Filters customers to ensure that all have the same opportunity to
        # generate revenue until [days_limits] after registration
        end_events_date = self.joined_df[self.event_time_col].max()
        cohort_filter = (
            end_events_date - self.joined_df[self.registration_time_col]
        ).dt.days >= days_limit  # ensure to only gets cohorts that are 'days_limit' old
        opportunity_filter = (
            self.joined_df[self.event_time_col]
            - self.joined_df[self.registration_time_col]
        ).dt.days <= days_limit  # only consider the first 'days_limit' days of a customer
        customer_revenue_data = self.joined_df[
            cohort_filter & opportunity_filter
        ].copy()

        # Create a new dataframe for cross join, so we can ensure that all
        # customers have (days_limit + 1) days to calculate correlation
        days_data = pd.DataFrame(
            {
                "days_since_install": np.linspace(
                    optimization_window,
                    days_limit,
                    days_limit - optimization_window + 1,
                ).astype(np.int32)
            }
        )
        days_data["key"] = 1
        customer_revenue_data["key"] = 1
        customer_revenue_data = pd.merge(
            customer_revenue_data, days_data, on="key"
        ).drop("key", axis=1)

        # Calculates the revenue of each customer until N days after
        # registration (as the customer doesn't necessarily spend on all days)
        customer_revenue_data[self.value_col] = (
            customer_revenue_data["days_since_registration"]
            <= customer_revenue_data["days_since_install"]
        ) * customer_revenue_data[self.value_col]
        customer_revenue_data = (
            customer_revenue_data.groupby([self.uuid_col, "days_since_install"])[
                self.value_col
            ]
            .sum()
            .reset_index()
        )
        # Calculate correlation, extract the correlation only for the 'early
        # revenue' and plot it
        customer_revenue_data = customer_revenue_data.pivot(
            index=self.uuid_col, columns="days_since_install", values=self.value_col
        ).reset_index()
        customer_revenue_data = customer_revenue_data.drop(
            self.uuid_col, axis=1).corr()

        # Filter out only some of the days, otherwise there will have too much
        # granularity for visualization
        interval_size = (
            interval_size
            if interval_size is not None
            else np.round((days_limit - optimization_window) / 20)
        )
        # doesn't work if applied directly on the output of np.round for some
        # reason
        interval_size = interval_size.astype(int)
        days_of_interest = list(
            range(optimization_window, days_limit, interval_size))

        customer_revenue_data = customer_revenue_data[days_of_interest][
            customer_revenue_data.index.isin(days_of_interest)
        ]
        mask = np.zeros_like(customer_revenue_data, dtype=bool)
        mask[np.tril_indices_from(mask)] = True

        fig = self.graph.heatmap_plot(
            customer_revenue_data,
            title="Correlation matrix of revenue per user by different days since registration\n",
            xlabel="Days Since Registration",
            ylabel="Days Since Registration",
        )

        # Highlight the day when correlation drops below 50%
        correlation_threshold = 0.5
        day_below_threshold = 0
        for i in range(len(days_of_interest)):
            if customer_revenue_data.iloc[i, 0] < correlation_threshold:
                day_below_threshold = days_of_interest[i]
                break

        # Calculate the x and y coordinate of the vertical line
        x_coord = (day_below_threshold - optimization_window) / \
            interval_size + 0.5
        # only 1 block down from the diagonal
        y_coord = (optimization_window - 0.4) / optimization_window
        plt.axvline(x=x_coord, ymin=0, ymax=y_coord,
                    color="red", linestyle="--")
        return fig, customer_revenue_data

    @staticmethod
    def _classify_spend(x: float, spending_breaks: Dict[str, float]):
        key = np.argmax(x <= np.array(list(spending_breaks.values())))
        return list(spending_breaks.keys())[key]

    def _group_users_by_spend(
        self,
        days_limit: int,
        early_limit: int,
        spending_breaks: Dict[str, float],
        end_spending_breaks: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Group users based  on their early (early_limit) and late (days_limit) revenue
        Inputs:
            days_limit: number of days of the event since registration used to define the late spending class
            early_limit: number of days of the event since registration that is considered 'early'. Usually refers to optimization window of marketing platforms
            spending_breaks: dictionary, in which the keys defines the name of the class and the values the upper limit of the spending associated with the class. Lower limit is considered to be the lower limit of the previous class, else 0
            end_spending_breaks: dictionary, in which the keys defines the name of the class and the values the upper limit of the spending associated with the class. Lower limit is considered to be the lower limit of the previous class, else 0
        """
        # Select only customers that are at least [days_limit] days old
        end_events_date = self.joined_df[self.event_time_col].max()
        data = self.joined_df[
            (end_events_date -
             self.joined_df[self.registration_time_col]).dt.days
            >= days_limit
        ].copy()

        # Remove customers who never had a purchase and ensure all customers
        # have the same opportunity window
        data = data[
            (data[self.event_time_col] -
             data[self.registration_time_col]).dt.days
            <= days_limit
        ]

        # Count how many purchase customers had (defined by value > 0) and then
        # how many customers are in each place
        data = data[data[self.value_col] > 0]
        data["dsi"] = (
            (data[self.event_time_col] -
             data[self.registration_time_col]).dt.days
        ).fillna(0)
        data["early_revenue"] = data.apply(
            lambda x: (x["dsi"] <= early_limit) * x[self.value_col], axis=1
        )
        data["late_revenue"] = data.apply(
            lambda x: (x["dsi"] <= days_limit) * x[self.value_col], axis=1
        )

        data = (
            data.groupby(self.uuid_col)[["early_revenue", "late_revenue"]]
            .sum()
            .reset_index()
        )

        # Adding default spending breaks if there was none.
        if len(spending_breaks) == 0:
            zero_mask = data["early_revenue"] != 0
            non_zero_data = data[zero_mask]
            spending_breaks["No spend"] = 0
            spending_breaks["Low spend"] = np.percentile(
                non_zero_data["early_revenue"], 33.33
            ).round(2)
            spending_breaks["Medium spend"] = np.percentile(
                non_zero_data["early_revenue"], 66.67
            ).round(2)
            spending_breaks["High spend"] = np.ceil(
                data["early_revenue"].max())
            print("Starting spending breaks:", spending_breaks)

        # Adding default end spending breaks if there was none.
        if len(end_spending_breaks) == 0:
            zero_mask = data["late_revenue"] != 0
            non_zero_data = data[zero_mask]
            end_spending_breaks["No spend"] = 0
            end_spending_breaks["Low spend"] = np.percentile(
                non_zero_data["late_revenue"], 33.33
            ).round(2)
            end_spending_breaks["Medium spend"] = np.percentile(
                non_zero_data["late_revenue"], 66.67
            ).round(2)
            end_spending_breaks["High spend"] = np.ceil(
                data["late_revenue"].max())
            print("Ending spending breaks:", end_spending_breaks)

        # Spending breaks needs to be sorted in ascending order
        sorted_spending_breaks = dict(
            sorted(spending_breaks.items(), key=lambda x: x[1])
        )
        sorted_end_spending_breaks = dict(
            sorted(end_spending_breaks.items(), key=lambda x: x[1])
        )

        data["early_class"] = data["early_revenue"].apply(
            lambda x: self._classify_spend(x, sorted_spending_breaks)
        )
        data["late_class"] = data["late_revenue"].apply(
            lambda x: self._classify_spend(x, sorted_end_spending_breaks)
        )

        def summary(data: pd.DataFrame):
            output = {}
            output["customers"] = len(data)
            output["cumulative_early_revenue"] = data["early_revenue"].sum()
            output["average_cumulative_early_revenue"] = data["early_revenue"].mean()
            output["cumulative_late_revenue"] = data["late_revenue"].sum()
            output["average_cumulative_late_revenue"] = data["late_revenue"].mean()
            return pd.Series(output)

        return (
            data.groupby(["early_class", "late_class"])[
                [self.uuid_col, "early_revenue", "late_revenue"]
            ]
            .apply(summary)
            .sort_values(
                ["average_cumulative_early_revenue",
                    "average_cumulative_late_revenue"]
            )
            .reset_index()
        )

    def plot_paying_customers_flow(
        self,
        days_limit: int,
        early_limit: int,
        spending_breaks: Dict[str, float],
        end_spending_breaks: Dict[str, float],
    ):
        """
        Plots the flow of customers from early spending class to late spending class
        Inputs:
            days_limit: number of days of the event since registration used to define the late spending class
            early_limit: number of days of the event since registration that is considered 'early'. Usually refers to optimization window of marketing platforms
            spending_breaks: dictionary, in which the keys defines the name of the class and the values the upper limit of the spending associated with the class. Lower limit is considered to be the lower limit of the previous class, else 0
            end_spending_breaks: dictionary, in which the keys defines the name of the class and the values the upper limit of the spending associated with the class. Lower limit is considered to be the lower limit of the previous class, else 0
        """
        data = self._group_users_by_spend(
            days_limit, early_limit, spending_breaks, end_spending_breaks
        )

        # add combination of early and late classes that have no customers
        unique_classes = data["early_class"].unique()
        skeleton_data = pd.DataFrame(
            data=list(product(unique_classes, unique_classes)),
            columns=["early_class", "late_class"],
        )
        visualization_data = pd.merge(
            skeleton_data, data, how="left", on=["early_class", "late_class"]
        ).fillna(0.0)
        # Categorize classes, so that they are able to be ordered
        visualization_data["early_class"] = pd.Categorical(
            visualization_data["early_class"],
            ordered=True,
            categories=["No spend", "Low spend", "Medium spend", "High spend"],
        )
        visualization_data["late_class"] = pd.Categorical(
            visualization_data["late_class"],
            ordered=True,
            categories=["No spend", "Low spend", "Medium spend", "High spend"],
        )
        visualization_data.sort_values(
            ["early_class", "late_class"], inplace=True)
        # Add new columns
        visualization_data["perc_total_customers"] = (
            visualization_data["customers"] /
            visualization_data["customers"].sum()
        ) * 100

        # add a percentage sign in front of the values
        visualization_data["perc_total_customers"] = visualization_data[
            "perc_total_customers"
        ].apply(lambda x: f"{x:.1f}%")

        visualization_data = visualization_data[
            [
                "early_class",
                "late_class",
                "customers",
                "perc_total_customers",
                "average_cumulative_early_revenue",
                "cumulative_early_revenue",
                "average_cumulative_late_revenue",
                "cumulative_late_revenue",
            ]
        ]

        # Create a new column for the actual customer count because the
        # modified customer column is somehow used in the flow chart UI
        # rendering.
        data["customer_count"] = data["customers"]
        data["customers"] = data["customers"] / data["customers"].sum()
        self.interactive_chart.legend_out = True
        fig = self.interactive_chart.flow_chart(
            data,
            "early_class",
            "late_class",
            "customers",
            title=f"Purchaser Flow Between an early point in time ({early_limit} days) and a future point in time ({days_limit} days)",
            source_day_limit=early_limit,
            target_day_limit=days_limit,
        )

        return fig, visualization_data.round(self.rounding_precision)

    def _get_upper_limit_ltv(
        self, users_flow_df: pd.DataFrame, is_mobile: bool
    ) -> pd.Series:
        """
        This function estimates the new LTV that should be attributed to users as a result of using a LTV optimization
        This method depends whether the business is from eCommerce or Gaming/Mobile Apps
        Inputs
            users_flow_df: the output dataframe from  plot_paying_customers_flow
            is_mobile: whether it relates to gaming/mobile app business. If not, assumes it is eCommerce
        """
        if is_mobile:
            return self._get_mobile_ltv(users_flow_df)
        else:
            return self._get_ecomm_ltv(users_flow_df)

    def _get_mobile_ltv(self, users_flow_df) -> pd.Series:
        """
        For each combination of (early class, late class), find the largest LTV in each (early_class). This is the best case scenario LTV
        """
        upper_ltv_df = (
            users_flow_df.groupby(["early_class"])[
                "average_cumulative_late_revenue"]
            .max()
            .reset_index()
            .rename(
                columns={
                    "average_cumulative_late_revenue": "best_case_average_late_revenue"
                }
            )
        )
        return users_flow_df.merge(upper_ltv_df, how="inner", on=["early_class"])[
            "best_case_average_late_revenue"
        ]

    def _get_ecomm_ltv(self, users_flow_df) -> pd.Series:
        """
        Ignore users with no revenue at the beginning. Do nothing on them
        For each (early class), change the values of the class 'Low Value' for the late_class to be the average of the LTV of the classes Medium and High (for late_class), weighted by the number of users
        """
        upper_ltv_df = (
            users_flow_df[
                (users_flow_df["early_class"] != "No spend")
                & (users_flow_df["late_class"] != "Low spend")
            ]
            .groupby(["early_class"])[["cumulative_late_revenue", "customers"]]
            .sum()
            .reset_index()
        )
        upper_ltv_df["best_case_average_late_revenue"] = (
            upper_ltv_df["cumulative_late_revenue"] / upper_ltv_df["customers"]
        )
        upper_ltv_df["late_class"] = "Low spend"
        upper_ltv_df = upper_ltv_df[
            ["early_class", "late_class", "best_case_average_late_revenue"]
        ]
        upper_ltv_df = users_flow_df.merge(
            upper_ltv_df, how="left", on=["early_class", "late_class"]
        )
        return upper_ltv_df.apply(
            lambda x: (
                x["best_case_average_late_revenue"]
                if ~np.isnan(x["best_case_average_late_revenue"])
                else x["average_cumulative_late_revenue"]
            ),
            axis=1,
        )

    def estimate_ltv_impact(
        self,
        days_limit: int,
        early_limit: int,
        spending_breaks: Dict[str, float],
        is_mobile: bool,
    ):
        """
        Estimate the impact of using a predicted LTV (pLTV) strategy for campaign optimization.
        The estimation is an upper limit scenario to show the maximum impact one could ever foresee in just using LTV optimization.
        As a consequence, it describes an unrealistic scenario and should be used only to draw the upper boundary

        The calculation depends on whether the data refers to an ecommerce or a mobile/gaming company.
        """
        # Get users grouped by their early and late revenue
        data = self._group_users_by_spend(
            days_limit, early_limit, spending_breaks.copy(), spending_breaks.copy()
        )

        # Apply the average LTV of the highest-spending class to all spending
        # classes
        data["assumed_average_new_late_revenue"] = self._get_upper_limit_ltv(
            data, is_mobile
        )
        data["assumed_new_late_revenue"] = (
            data["assumed_average_new_late_revenue"] * data["customers"]
        )
        data["abs_revenue_increase"] = (
            data["assumed_new_late_revenue"] - data["cumulative_late_revenue"]
        )

        abs_impact = np.sum(data["abs_revenue_increase"])
        rel_impact = abs_impact / np.sum(data["cumulative_late_revenue"])

        output_txt = f"""
        By adopting a predicted LTV (pLTV) based strategy for your marketing campaigns, we estimate a maximum increase of {100*rel_impact:.1f}% in revenue.
        This increase represents ${abs_impact:.0f} in revenue for the time period and scope used by the data provided.
        """
        print(output_txt)

        return data.round(self.rounding_precision)

    def download_data(
        self,
        df: pd.DataFrame,
        filename: str = "data.csv",
        path: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save the raw data behind the output as a CSV file.
        Args:
            df (pd.DataFrame): The dataframe containing the data to be saved.
            filename (str, optional): The filename for the saved CSV file. Defaults to "data.csv".
            path (str, optional): The custom path where the file will be saved. Defaults to None, which uses the current working directory.
            overwrite (bool, optional): Whether to overwrite an existing file without prompting. Defaults to False.
        Raises:
            ValueError: If the filename is not a string or if the path is not a valid directory path.
            OSError: If the program does not have permission to write to the specified path.
        """
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")
        if path is None:
            path = os.getcwd()
        try:
            # Construct the full file path
            file_path = os.path.join(path, filename)
            # Create the directory if it does not exist
            os.makedirs(path, exist_ok=True)
            # Check if a file with the same name already exists
            if os.path.exists(file_path) and not overwrite:
                overwrite_prompt = input(
                    f"A file named {filename} already exists in the specified directory. Do you want to overwrite it? (y/n): "
                )
                if overwrite_prompt.lower() != "y":
                    return
            elif os.path.exists(file_path) and overwrite:
                print(f"Overwriting existing file {filename}...")
            # Save the dataframe as a CSV file
            df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
        except OSError as e:
            print(f"Error saving data: {e}")
