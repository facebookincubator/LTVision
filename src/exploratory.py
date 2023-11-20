# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from typing import List

sns.set_style("whitegrid")


class LTVexploratory:
    """This class helps to perform som initial analysis"""

    def __init__(
        self,
        data_ancor: pd.DataFrame,
        data_events: pd.DataFrame,
        uuid_col: str = "UUID",
        registration_time_col: str = "timestamp_registration",
        event_time_col: str = "timestamp_event",
        event_name_col: str = "event_name",
        value_col: str = "revenue",
        segment_feature_cols: List[str] = [],
    ):
        self.data_ancor = data_ancor
        self.data_events = data_events
        self._period = 7
        self._period_for_ltv = 7 * 10
        # store information about the columns of the dataframes
        self.uuid_col = uuid_col
        self.registration_time_col = registration_time_col
        self.event_time_col = event_time_col
        self.event_name_col = event_name_col
        self.value_col = value_col
        self.segment_feature_cols = segment_feature_cols
        # run auxiliar methods
        self._prep_df()


    def _prep_df(self) -> None:
        # Left join users and events data, so that you have a dataframe with all users and their events (if no event, then timestamp is null)
        # just select some columns to make it easier to understand what is the information that is used and avoid join complications caused by the name of other columns
        self.joined_df = pd.merge(
            self.data_ancor,
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

        n_uuid_for_deletion = self.joined_df[
            self.joined_df[self.event_time_col].isnull()
        ][self.uuid_col].nunique()
        total_uuids = self.data_ancor.shape[0]

    # Analysis Plots
    def summary(self):
        data_ancor = self.joined_df[
            [self.uuid_col, self.registration_time_col]
        ].drop_duplicates()
        print(
            f"""
            table:         data_ancor
            date start:    {data_ancor[self.registration_time_col].min()}
            date end:      {data_ancor[self.registration_time_col].max()}
            period:        {(data_ancor[self.registration_time_col].max()-data_ancor[self.registration_time_col].min()).days/365:.2f} years
            customers:     {data_ancor[self.uuid_col].nunique():,d}
            events:        {data_ancor.shape[0]:,d}
            """
        )

        data_events = self.joined_df[~self.joined_df[self.event_time_col].isnull()]
        print(
            f"""  
            table:         self.data_event
            date start:    {data_events[self.event_time_col].min()}
            date end:      {data_events[self.event_time_col].max()}
            period:        {(data_events[self.event_time_col].max()-data_events[self.event_time_col].min()).days/365:.2f} years
            customers:     {data_events[self.uuid_col].nunique():,d}
            events:        {data_events.shape[0]:,d}
            unique events: {data_events[self.event_name_col].nunique():,d}
            events / user: {(data_events.shape[0] / data_events[self.uuid_col].nunique()):.2f}
            """
        )

    # All plot methods
    def plot_users_intersection(self):
        """
        Plot the interection between users in the two input data
        We expect that all users in events data are also in ancor data.
        The inverse can be true, as there may be users who never sent an event
        """

        # Get uuids from each input data
        ancor_uuids = pd.DataFrame({"uuid": self.data_ancor[self.uuid_col].unique()})
        ancor_uuids["ancor"] = "Present in Ancor"
        events_uuids = pd.DataFrame({"uuid": self.data_events[self.uuid_col].unique()})
        events_uuids["events"] = "Present in Events"

        # Calculate how many users are in each category
        cross_uuid = pd.merge(ancor_uuids, events_uuids, on="uuid", how="outer")
        cross_uuid["ancor"] = cross_uuid["ancor"].fillna("Not in Ancor")
        cross_uuid["events"] = cross_uuid["events"].fillna("Not in Events")
        cross_uuid = (
            cross_uuid.groupby(["ancor", "events"])["uuid"].count().reset_index()
        )

        # Create a dataframe containing all combinations for the visualization
        complete_data = pd.DataFrame(
            {
                "ancor": [
                    "Present in Ancor",
                    "Not in Ancor",
                    "Present in Ancor",
                    "Not in Ancor",
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
            complete_data, cross_uuid, on=["ancor", "events"], how="left"
        )
        complete_data = complete_data.fillna(0)
        complete_data["description"] = complete_data.apply(
            lambda row: row["ancor"] + " & " + row["events"], axis=1
        )

        graph.bar_plot(
            complete_data,
            x_axis="description",
            y_axis="uuid",
            xlabel="",
            ylabel="Number of users",
        )

    def plot_purchases_distribution(
        self, days_limit: int, truncate_share: float = 0.99
    ):
        """
        Plots an histogram of the number of people by how many purchases they had until [days_limit] after their registration
        We need [days_limit] to ensure that we only select users with the same opportunity window (i.e. all users are at least
        [days_limit] days old at the time the data was collected).
        )
        Input
            days_limit: number of days of the event since registration.
            truncate_share: share of total users/revenue until where the plot shows values
        """
        # Select only users that are at least [days_limit] days old
        end_events_date = self.joined_df[self.event_time_col].max()
        data = self.joined_df[
            (end_events_date - self.joined_df[self.registration_time_col]).dt.days
            >= days_limit
        ].copy()

        # Remove users who never had a purchase and ensure all users have the same opportunity window
        data = data[
            (data[self.event_time_col] - data[self.registration_time_col]).dt.days
            <= days_limit
        ]

        # Count how many purchase users had (defined by value > 0) and then how many users are in each place
        data = data[data[self.value_col] > 0]
        data = (
            data.groupby(self.uuid_col)[self.value_col]
            .agg(["sum", "count"])
            .reset_index()
        )
        data = data.drop(self.uuid_col, axis=1)
        data = data.rename(columns={"count": "purchases"})
        data = data.groupby("purchases")["sum"].agg(["sum", "count"]).reset_index()
        # Calculate the share
        data["sum"] = data["sum"] / data["sum"].sum()
        data["count"] = data["count"] / data["count"].sum()
        # Find treshold truncation
        users_truncation = data["count"].cumsum() <= truncate_share
        value_truncation = data["sum"].cumsum() <= truncate_share
        # plot distribution by users and by revenue
        graph.bar_plot(
            data[users_truncation],
            x_axis="purchases",
            y_axis="count",
            xlabel="Number of purchases",
            ylabel="Share of customers",
            y_format="%",
            title=f"Distribution of paying users (Y, %) by number of purchases of each user until {days_limit} days after registration  (X)",
        )
        graph.bar_plot(
            data[value_truncation],
            x_axis="purchases",
            y_axis="sum",
            xlabel="Number of purchases",
            ylabel="Share of value",
            y_format="%",
            title=f"Distribution of value (Y, %) by number of purchases of each user until {days_limit} days after registration (X)",
        )

    def plot_n(self):
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        self.data_ancor["ancor_event_name"].value_counts().plot.barh(ax=ax[0])
        ax[0].set_title("event_name from df_ancore")
        self.data_events["event_name"].value_counts().plot.barh(ax=ax[1])
        ax[1].set_title("event_name from df_event")
        plt.show()

    def plot_first_purchase(self):
        fig, ax = plt.subplots(figsize=(20, 5))
        days_before_first_purchase = self._df.groupby("UUID")[
            "days_between_purchase_and_reg"
        ].min()
        days_before_first_purchase.hist(ax=ax, bins=50)
        plt.title("Days between registration and the first purchase")
        plt.ylabel("Users")
        plt.xlabel("Days")
        plt.show()
        pd.set_option("display.float_format", lambda x: "%.1f" % x)
        display(
            days_before_first_purchase.reset_index()
            .rename(
                {
                    "days_between_purchase_and_reg": "Days between registration and the first purchase stat:"
                },
                axis=1,
            )
            .describe()
        )

    def plot_second_purchase(self):
        # Let's have a look on days before the second purchase
        fig, ax = plt.subplots(figsize=(20, 5))
        days_after_first_purchase = (
            self._df[self._df["days_after_first_purch"] > 0]
            .groupby("UUID")["days_after_first_purch"]
            .min()
        )
        days_after_first_purchase.hist(ax=ax, bins=50)
        plt.title("Days between the first purchase and the next one")
        plt.ylabel("Users")
        plt.xlabel("Days")
        plt.show()
        pd.set_option("display.float_format", lambda x: "%.1f" % x)
        stat = (
            days_after_first_purchase.reset_index()
            .rename(
                {
                    "days_after_first_purch": "Days between the first purchase and the next one stat:"
                },
                axis=1,
            )
            .describe()
        )
        stat.loc[
            "users without second purchase"
        ] = f"{(self._df[self.uuid_col].nunique() - self._df.loc[self._df['days_after_first_purch'] > 0, 'UUID'].nunique())/self._df[self.uuid_col].nunique():.2%}"
        display(stat)
        print(
            f"""You have {(self._df['purchase_value'] == 0).sum()/self._df.shape[0]:.2%} purchases with zero value"""
        )

    def plot_uuid(self):
        # Let's have a look at distribution of uuid life time
        fig, ax = plt.subplots(figsize=(20, 5))
        n, bins, patches = ax.hist(
            self._df.groupby("UUID")["max_days_for_ltv"].min(), bins=30
        )
        ax.set_title("Life time distribution of users who have purchases history")
        plt.show()

    def plot_linear_corr(self):
        def r2_score_standardised(a, b):
            a = a[(~np.isnan(a)) & (~np.isnan(b))]
            b = b[(~np.isnan(a)) & (~np.isnan(b))]
            if len(a) > 2:
                return r2_score(
                    (a - np.mean(a)) / (np.std(a) if np.std(a) != 0 else 1),
                    (b - np.mean(b)) / (np.std(b) if np.std(b) != 0 else 1),
                )
            return np.nan

        # Calculate linear correlation
        print(
            "The main idea of this chart: Do we have a strong linear correlation in order to build linear model? (1 - yes, 0 - no)"
        )
        _, ax = plt.subplots(figsize=(30, 15))
        sns.heatmap(
            self._ltv_periods.corr(),
            annot=True,
            cmap=sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True),
            ax=ax,
        )
        plt.title("Linear correlation between LTV for different days (Pearson)")

        print(
            """The main idea of this chart: if we take (ltv value for 7 days) as a feture to our linear model in order to predict (ltv value for day 14)
        how well the regression model explains observed data (proportion of the variation in the dependent variable that is predictable from the independent variable)?"""
        )
        _, ax = plt.subplots(figsize=(30, 15))
        cor = self._ltv_periods.corr(method=r2_score_standardised)
        # r2 < 0 means that it's better to get a constant as a feature than ltv for chosing period
        cor[cor < 0] = 0
        sns.heatmap(
            cor,
            annot=True,
            cmap=sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True),
            ax=ax,
        )
        plt.title("R2 between normalised LTV for different days")

    def plot_sankey(self):
        ind = ~self._payer_types[self._period_for_ltv].isna()
        pivot_count_table = (
            self._payer_types.loc[ind, [self._period, self._period_for_ltv]]
            .astype(str)
            .reset_index()
            .pivot_table(
                index=self._period,
                columns=self._period_for_ltv,
                values="UUID",
                aggfunc="count",
            )
            .fillna(0)
        )
        pivot_count_table = pivot_count_table / pivot_count_table.sum().sum()

        source = sum(
            [
                [x] * len(pivot_count_table.columns)
                for x in pivot_count_table.index.to_list()
            ],
            [],
        )
        target = pivot_count_table.columns.to_list() * len(pivot_count_table.index)
        value = [pivot_count_table.loc[x, y] * 100 for x, y in zip(source, target)]
        dict_translate = {
            "0.users_with_zero_purchases": {
                "color": "red",
                "color_link": "rgba(255, 204, 203, 0.8)",
                "ind": 0,
            },
            "1.low_payers": {
                "color": "orange",
                "color_link": "rgba(254, 216, 177, 0.8)",
                "ind": 1,
            },
            "2.mid_payers": {
                "color": "blue",
                "color_link": "rgba(173, 216, 230, 0.8)",
                "ind": 2,
            },
            "3.high_payers": {
                "color": "green",
                "color_link": "rgba(144, 238, 144, 0.8)",
                "ind": 3,
            },
        }
        label = list(dict_translate.keys())
        labels = [x for x in label if x in pivot_count_table.sum(axis=1).index] + [
            x for x in label if x in pivot_count_table.sum(axis=0).index
        ]
        value_for_nodes = [
            f"{x[2:]} ({pivot_count_table.sum(axis=1)[x]:.1%})"
            for x in label
            if x in pivot_count_table.sum(axis=1).index
        ] + [
            f"{x[2:]} ({pivot_count_table.sum(axis=0)[x]:.1%})"
            for x in label
            if x in pivot_count_table.sum(axis=0).index
        ]
        fig = go.Figure(
            data=[
                go.Sankey(
                    valueformat=".0f",
                    valuesuffix="TWh",
                    # Define nodes
                    node=dict(
                        pad=15,
                        thickness=15,
                        line=dict(color="black", width=0.5),
                        label=value_for_nodes,
                        color=[dict_translate[x]["color"] for x in labels],
                    ),
                    # Add links
                    link=dict(
                        source=[dict_translate[x]["ind"] for x in source],
                        target=[
                            dict_translate[x]["ind"] + len(pivot_count_table.index)
                            for x in target
                        ],
                        value=value,
                        label=[f"{x/100:.1%}" for x in value],
                        color=[dict_translate[x]["color_link"] for x in target],
                    ),
                )
            ]
        )
        for x_coordinate, column_name in enumerate(
            [f"{self._period}-day Revenue", f"{self._period_for_ltv}-day Revenue"]
        ):
            fig.add_annotation(
                x=x_coordinate,
                y=1.1,
                xref="x",
                yref="paper",
                text=column_name,
                showarrow=False,
                font=dict(
                    #   family="Courier New, monospace",
                    size=11,
                    color="black",
                ),
                align="center",
            )

        fig.update_layout(
            title_text="Opportunity size for adopting a pLTV strategy",
            xaxis={
                "showgrid": False,  # thin lines in the background
                "zeroline": False,  # thick line at x=0
                "visible": False,  # numbers below
            },
            yaxis={
                "showgrid": False,  # thin lines in the background
                "zeroline": False,  # thick line at x=0
                "visible": False,  # numbers below
            },
            plot_bgcolor="rgba(0,0,0,0)",
            font_size=10,
            title_x=0.5,
        )
        plt.show()

    def plot_class_change(self):
        def accuracy_score_for_not_na_values(a, b):
            return accuracy_score(
                a[(~np.isnan(b)) & (~np.isnan(a))], b[(~np.isnan(b)) & (~np.isnan(a))]
            )

        print(
            "Let's look at this value (how many users will have the same classification flag if we compare different periods between each other)"
        )
        _, ax = plt.subplots(figsize=(30, 15))
        cor = self._payer_types.replace(
            {
                "0.users_with_zero_purchases": 0,
                "1.low_payers": 1,
                "2.mid_payers": 2,
                "3.high_payers": 3,
            }
        ).corr(method=accuracy_score_for_not_na_values)
        sns.heatmap(
            cor,
            annot=True,
            cmap=sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True),
            ax=ax,
        )
        plt.title(
            "Share of users who didn't change their classification flag (no-payers, low-payer, mid-payer, high-payer) across periods"
        )

    def plot_drop_off(self):
        # Let's have a look at distribution of number uuid which can be in the train set depending on life time which we will chose for our model
        fig, ax = plt.subplots(figsize=(30, 5))
        _data = self._df.groupby("UUID")["max_days_for_ltv"].min()
        ax.hist(_data, bins=1000, cumulative=-1)
        ax.set_title(
            "Number of users which we can use in train set depend on horizon which we choose"
        )
        ax.set_xticks(np.arange(_data.min(), _data.max(), 50))
        plt.show()

    def plot_(self):
        # We have users with different life times, it means that building LTV distribution histogram for all users together will be not good.
        # Because in this case we will compare ltv for a user who have couple years purchase history with a user who joined a couple days ago. It doesn't make sense.
        # So the main idea is to see distribution for users groupped by almost the same number of active days on the platform
        active_user_stats = self._df.groupby("UUID").agg(
            {"max_days_for_ltv": "min", "purchase_value": "sum"}
        )
        lifeteime_quantiles = active_user_stats["max_days_for_ltv"].quantile(
            np.linspace(0, 1, 20 + 1)
        )
        active_user_stats["lifetime_bins"] = pd.cut(
            active_user_stats["max_days_for_ltv"],
            bins=lifeteime_quantiles,
            labels=lifeteime_quantiles.iloc[:-1].astype(str)
            + " - "
            + lifeteime_quantiles.shift(-1).iloc[:-1].astype(str),
            include_lowest=True,
        )
        _, ax = plt.subplots(figsize=(40, 10), dpi=50)
        sns.boxplot(
            data=active_user_stats,
            x="lifetime_bins",
            y="purchase_value",
            ax=ax,
            color="orange",
        )
        # sns.violinplot(data=active_user_stats, x="lifetime_bins", y="purchase_value")
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Active days bins")
        ax.set_ylabel("LTV purchases")
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(
            data=active_user_stats.groupby("lifetime_bins")["purchase_value"]
            .median()
            .reset_index(),
            x="lifetime_bins",
            y="purchase_value",
            ax=ax2,
        )
        ax2.set_ylabel("median LTV purchases")
        plt.title("LTV distribution depending on user's active days")
        plt.show()
        active_user_stats["outlier_line"] = (
            active_user_stats["lifetime_bins"]
            .map(
                active_user_stats.groupby("lifetime_bins")["purchase_value"].quantile(
                    0.95
                )
            )
            .astype(np.float64)
        )
        active_user_stats = active_user_stats[
            active_user_stats["purchase_value"] <= active_user_stats["outlier_line"]
        ]
        _, ax = plt.subplots(figsize=(40, 10), dpi=50)
        # sns.boxplot(data=active_user_stats, x='lifetime_bins', y='purchase_value', ax=ax, color='orange')
        sns.violinplot(
            data=active_user_stats,
            x="lifetime_bins",
            y="purchase_value",
            width=1,
            linewidth=1,
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Active days bins")
        ax.set_ylabel("LTV purchases")
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(
            data=active_user_stats.groupby("lifetime_bins")["purchase_value"]
            .median()
            .reset_index(),
            x="lifetime_bins",
            y="purchase_value",
            ax=ax2,
        )
        ax2.set_ylabel("median LTV purchases")
        plt.title(
            "LTV distribution depending on user's active days (without outliers in each group)"
        )
        plt.show()

    def plot_freq(self):
        # Let's have a look on the first component of LTV: frequency
        active_user_stats = self._df.groupby("UUID").agg(
            {"max_days_for_ltv": "min", "purchase_value": "count"}
        )
        active_user_stats["purchase_value"] = (
            active_user_stats["purchase_value"]
            / active_user_stats["max_days_for_ltv"]
            * 30
        )
        lifeteime_quantiles = active_user_stats["max_days_for_ltv"].quantile(
            np.linspace(0, 1, 20 + 1)
        )
        active_user_stats["lifetime_bins"] = pd.cut(
            active_user_stats["max_days_for_ltv"],
            bins=lifeteime_quantiles,
            labels=lifeteime_quantiles.iloc[:-1].astype(str)
            + " - "
            + lifeteime_quantiles.shift(-1).iloc[:-1].astype(str),
            include_lowest=True,
        )

        _, ax = plt.subplots(figsize=(40, 10), dpi=50)
        sns.boxplot(
            data=active_user_stats,
            x="lifetime_bins",
            y="purchase_value",
            ax=ax,
            color="orange",
        )
        # sns.violinplot(data=active_user_stats, x="lifetime_bins", y="purchase_value")
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Active days bins")
        ax.set_ylabel("Purchases frequency (number purchases per 30 days)")
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(
            data=active_user_stats.groupby("lifetime_bins")["purchase_value"]
            .median()
            .reset_index(),
            x="lifetime_bins",
            y="purchase_value",
            ax=ax2,
        )
        ax2.set_ylabel("median Purchases frequency (number purchases per 30 days)")
        plt.title("Purchases frequency distribution depending on user's active days")
        plt.show()

        active_user_stats["outlier_line"] = (
            active_user_stats["lifetime_bins"]
            .map(
                active_user_stats.groupby("lifetime_bins")["purchase_value"].quantile(
                    0.95
                )
            )
            .astype(np.float64)
        )
        active_user_stats = active_user_stats[
            active_user_stats["purchase_value"] <= active_user_stats["outlier_line"]
        ]
        _, ax = plt.subplots(figsize=(40, 10), dpi=50)
        # sns.boxplot(data=active_user_stats, x='lifetime_bins', y='purchase_value', ax=ax, color='orange')
        sns.violinplot(
            data=active_user_stats,
            x="lifetime_bins",
            y="purchase_value",
            width=1,
            linewidth=1,
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Active days bins")
        ax.set_ylabel("Purchases frequency (number purchases per 30 days)")
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(
            data=active_user_stats.groupby("lifetime_bins")["purchase_value"]
            .median()
            .reset_index(),
            x="lifetime_bins",
            y="purchase_value",
            ax=ax2,
        )
        ax2.set_ylabel("median Purchases frequency (number purchases per 30 days)")
        plt.title(
            "Purchases frequency distribution depending on user's active days (without outliers in each group)"
        )
        plt.show()

    def plot_apv(self):
        active_user_stats = self._df.groupby("UUID").agg(
            {"max_days_for_ltv": "min", "purchase_value": "median"}
        )
        lifeteime_quantiles = active_user_stats["max_days_for_ltv"].quantile(
            np.linspace(0, 1, 20 + 1)
        )
        active_user_stats["lifetime_bins"] = pd.cut(
            active_user_stats["max_days_for_ltv"],
            bins=lifeteime_quantiles,
            labels=lifeteime_quantiles.iloc[:-1].astype(str)
            + " - "
            + lifeteime_quantiles.shift(-1).iloc[:-1].astype(str),
            include_lowest=True,
        )

        _, ax = plt.subplots(figsize=(40, 10), dpi=50)
        sns.boxplot(
            data=active_user_stats,
            x="lifetime_bins",
            y="purchase_value",
            ax=ax,
            color="orange",
            width=1,
            linewidth=1,
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Active days bins")
        ax.set_ylabel("Purchase value")
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(
            data=active_user_stats.groupby("lifetime_bins")["purchase_value"]
            .median()
            .reset_index(),
            x="lifetime_bins",
            y="purchase_value",
            ax=ax2,
        )
        ax2.set_ylabel("median purchase value purchases")
        plt.title("Purchase value distribution depending on user's active days")
        plt.show()

        active_user_stats["outlier_line"] = (
            active_user_stats["lifetime_bins"]
            .map(
                active_user_stats.groupby("lifetime_bins")["purchase_value"].quantile(
                    0.95
                )
            )
            .astype(np.float64)
        )
        active_user_stats = active_user_stats[
            active_user_stats["purchase_value"] <= active_user_stats["outlier_line"]
        ]
        _, ax = plt.subplots(figsize=(40, 10), dpi=50)
        # sns.boxplot(data=active_user_stats, x='lifetime_bins', y='purchase_value', ax=ax, color='orange')
        sns.violinplot(
            data=active_user_stats,
            x="lifetime_bins",
            y="purchase_value",
            width=1,
            linewidth=1,
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Active days bins")
        ax.set_ylabel("Purchase value")
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(
            data=active_user_stats.groupby("lifetime_bins")["purchase_value"]
            .median()
            .reset_index(),
            x="lifetime_bins",
            y="purchase_value",
            ax=ax2,
        )
        ax2.set_ylabel("median purchase value purchases")
        plt.title(
            "Purchase value distribution depending on user's active days (without outliers in each group)"
        )
        plt.show()

    def plot_corr_freq_apv(self):
        print(
            """Interpretation: Correlation value can be [-1,1]. If you have correlation closer to 0, it means that you can use models where these two variables describes different models."""
        )
        _, ax = plt.subplots(figsize=(30, 10), dpi=50)
        stat = self._df.groupby("UUID")[["purchase_value", "max_days_for_ltv"]].agg(
            {"purchase_value": ["mean", "count"], "max_days_for_ltv": "min"}
        )
        stat[("purchase_value", "count")] = (
            stat[("purchase_value", "count")] / stat[("max_days_for_ltv", "min")] * 30
        )
        stat["purchase_value"].plot.scatter(x="mean", y="count", ax=ax)
        ax.set_ylabel("frequencuy")
        ax.set_xlabel("mean purchase")
        plt.title("Check independancy frequency from purchase value")
        plt.show()
        print(
            f"linear_corr={stat['purchase_value']['mean'].corr(stat['purchase_value']['count']):.2f}"
        )
        # stat['purchase_value'].phik_matrix() <- not availible, only in local notebooks (for this data it equals 0 - we don't have correlation between these values).
        # Let's delete outliers, because now it seems that we have dependancy 'as higher price, as lower frequency'
        _, ax = plt.subplots(figsize=(30, 10), dpi=50)
        stat = stat[
            (
                stat[("purchase_value", "count")]
                <= stat[("purchase_value", "count")].quantile(0.99)
            )
            & (
                stat[("purchase_value", "mean")]
                <= stat[("purchase_value", "mean")].quantile(0.99)
            )
        ]
        stat["purchase_value"].plot.scatter(x="mean", y="count", ax=ax)
        ax.set_ylabel("frequencuy")
        ax.set_xlabel("mean purchase")
        plt.title("Check independancy frequency from purchase value (without outliers)")
        plt.show()
        print(
            f"linear_corr={stat['purchase_value']['mean'].corr(stat['purchase_value']['count']):.2f}"
        )
        # stat['purchase_value'].phik_matrix() <- not availible, only in local notebooks (for this data it equals 0.14 - we don't have correlation between these values).
