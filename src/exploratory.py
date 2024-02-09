# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module providing a class for initial analysis"""

from typing import List, Dict
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype , is_object_dtype, is_any_real_numeric_dtype, is_dtype_equal
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from src.graph import Graph
from src.aux import lag, cumsum, drop_duplicates

sns.set_style("whitegrid")

class LTVexploratory:
    """This class helps to perform som initial analysis"""

    def __init__(
        self,
        data_customers: pd.DataFrame,
        data_events: pd.DataFrame,
        uuid_col: str = 'UUID',
        registration_time_col: str = "timestamp_registration",
        event_time_col: str = "timestamp_event",
        event_name_col: str = "event_name",
        value_col: str = "purchase_value",
        segment_feature_cols: List[str] = None,
    ):
        self.data_customers = data_customers
        self.data_events = data_events
        self._period = 7
        self._period_for_ltv = 7 * 10
        self.graph = Graph()
        # store information about the columns of the dataframes
        self.uuid_col = uuid_col
        self.registration_time_col = registration_time_col
        self.event_time_col = event_time_col
        self.event_name_col = event_name_col
        self.value_col = value_col
        self.segment_feature_cols = [] if segment_feature_cols is None else segment_feature_cols
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
        assert((isinstance(
            self.data_customers[self.uuid_col].dtype, pd.StringDtype) or 
            is_object_dtype(self.data_customers[self.uuid_col]))) , f"The column [{self.uuid_col}] referencing to the customer-id in the customers dataset was expected to be of data pd.StringDtype or object. But it is of type {self.data_customers[self.uuid_col].dtype}"
        assert(is_datetime64_any_dtype(self.data_customers[self.registration_time_col])), f"The column [{self.registration_time_col}] referencing to the registrationtime in the customers dataset was expected to be of type [datime]. But it is of type {self.data_customers[self.registration_time_col].dtype}"

        # events dataset checks
        assert((isinstance(
            self.data_events[self.uuid_col].dtype, pd.StringDtype) or 
            is_object_dtype(self.data_events[self.uuid_col]))) , f"The column [{self.uuid_col}] referencing to the customer-id in the events dataset was expected to be of data pd.StringDtype or object. But it is of type {self.data_customers[self.uuid_col].dtype}"
        assert(is_datetime64_any_dtype(self.data_events[self.event_time_col])) , f"The column [{self.event_time_col}] referencing to the registrationtime in the events dataset was expected to be of type [datime]. But it is of type {self.data_customers[self.event_time_col].dtype}"
        assert(is_any_real_numeric_dtype(self.data_events[self.value_col])) , f"The column [{self.value_col}] referencing value of a transaction in the events dataset was expected to be of numeric. But it is of type {self.data_customers[self.value_col].dtype}"

        # consistency checks
        assert(is_dtype_equal(self.data_customers[self.uuid_col], self.data_events[self.uuid_col])), f"The customer-id columns of the two input datasets are not the same. In the customers dataset it is of type [{self.data_customers[self.uuid_col].dtype}], while in the events dataset it is of type [{self.data_customers[self.uuid_col].dtype}]"
        assert(is_dtype_equal(self.data_customers[self.registration_time_col], self.data_events[self.event_time_col])), f"The timestamp columns of the two input datasets are not the same. In the customers dataset it is of type [{self.data_customers[self.registration_time_col].dtype}], while in the events dataset it is of type [{self.data_customers[self.event_time_col].dtype}]"


    def _prep_df(self) -> None:
        # Left join customers and events data, so that you have a dataframe with all customers and their events (if no event, then timestamp is null)
        # just select some columns to make it easier to understand what is the information that is used and avoid join complications caused by the name of other columns
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
            table:         data_customers
            date start:    {data_customers[self.registration_time_col].min()}
            date end:      {data_customers[self.registration_time_col].max()}
            period:        {(data_customers[self.registration_time_col].max()-data_customers[self.registration_time_col].min()).days/365:.2f} years
            customers:     {data_customers[self.uuid_col].nunique():,d}
            events:        {data_customers.shape[0]:,d}
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
            events / customer: {(data_events.shape[0] / data_events[self.uuid_col].nunique()):.2f}
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
        customers_uuids = pd.DataFrame({self.uuid_col: self.data_customers[self.uuid_col].unique()})
        customers_uuids["customers"] = "Present in Customers"
        events_uuids = pd.DataFrame({self.uuid_col: self.data_events[self.uuid_col].unique()})
        events_uuids["events"] = "Present in Events"

        # Calculate how many customers are in each category
        cross_uuid = pd.merge(customers_uuids, events_uuids, on=self.uuid_col, how="outer")
        cross_uuid["customers"] = cross_uuid["customers"].fillna("Not in customers")
        cross_uuid["events"] = cross_uuid["events"].fillna("Not in Events")
        cross_uuid = (
            cross_uuid.groupby(["customers", "events"])[self.uuid_col].count().reset_index()
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
        complete_data[self.uuid_col] = complete_data[self.uuid_col]/np.sum(complete_data[self.uuid_col])
        fig = self.graph.grid_plot(complete_data, "customers", "events", self.uuid_col)
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
            (end_events_date - self.joined_df[self.registration_time_col]).dt.days
            >= days_limit
        ].copy()

        # Remove customers who never had a purchase and ensure all customers have the same opportunity window
        data = data[
            (data[self.event_time_col] - data[self.registration_time_col]).dt.days
            <= days_limit
        ]

        # Count how many purchase customers had (defined by value > 0) and then how many customers are in each place
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
        customers_truncation = data["count"].cumsum() <= truncate_share
        value_truncation = data["sum"].cumsum() <= truncate_share
        # plot distribution by customers
        fig = self.graph.bar_plot(
            data[customers_truncation],
            x_axis="purchases",
            y_axis="count",
            xlabel="Number of purchases",
            ylabel="Share of customers",
            y_format="%",
            title=f"Distribution of paying customers (Y, %) by number of purchases of each customer until {days_limit} days after registration  (X)",
        )
        return fig, data


    def plot_revenue_pareto(
        self, days_limit: int, granularity: int=1000
    ):
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
            (end_events_date - self.joined_df[self.registration_time_col]).dt.days
            >= days_limit
        ].copy()

        # Remove customers who never had a purchase and ensure all customers have the same opportunity window
        data = data[
            (data[self.event_time_col] - data[self.registration_time_col]).dt.days
            <= days_limit
        ]

        # Count how many purchase customers had (defined by value > 0) and then how many customers are in each place
        data = data[data[self.value_col] > 0]
        data = (
            data.groupby(self.uuid_col)[self.value_col].sum()
            .reset_index()
            .sort_values(self.value_col, ascending=False)
        )

        # Calculate total revenue and group customers together based on the granilarity
        total_revenue = data[self.value_col].sum()
        total_customers = data.shape[0]
        data['cshare_customers'] = [(i+1)/total_customers for i in range(total_customers)]
        data['cshare_revenue'] = data[self.value_col].cumsum() / total_revenue
        data['group'] = data['cshare_customers'].apply(lambda x: np.ceil(x*granularity))
        data = (
            data
            .groupby("group")[["cshare_customers", "cshare_revenue"]]
            .max()
            .reset_index()
        )

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

    def plot_customers_histogram_per_conversion_day(
            self,
            days_limit: int,
            optimization_window: int=7,
            truncate_share = 1.0) -> None:
        """
        Plots the distribution of all customers that converted until (days_limit) days after registration per conversion day
        Inputs:
            days_limit: number of days of the event since registration.
            optimization_window: the number of days since registration of a customer that matters for the optimization of campaigns
            truncate_share: the total share of purchasing customers that the histogram includes
        """

        end_events_date = self.joined_df[self.event_time_col].max()
        data = self.joined_df[
            (end_events_date - self.joined_df[self.registration_time_col]).dt.days
            >= days_limit
        ].copy()

        # Remove customers who never had a purchase and ensure all customers have the same opportunity window
        data = data[data[self.value_col] > 0]
        data['dsi'] = ((data[self.event_time_col] - data[self.registration_time_col]).dt.days).fillna(0)
        data = data[data['dsi'] <= days_limit]

        # calculate data for the histogram
        data = (
            data
            .groupby(self.uuid_col)['dsi']
            .min()
            .reset_index()
            .groupby('dsi')[self.uuid_col]
            .count()
            .reset_index()
        )

        # calculate the share of customers instead of absolute numbers and numbers for the title
        data[self.uuid_col]  = data[self.uuid_col] / data[self.uuid_col].sum()
        data = data[data[self.uuid_col].cumsum() < truncate_share]

        share_customers_within_window = data[data['dsi'] <= optimization_window][self.uuid_col].sum()
        title = f"Initial Purchase Cycle\n{100*share_customers_within_window:.1f}% of first purchases happened within the first {optimization_window} days since registration\n\nShare of paying customers (Y) versus conversion days since registration (X)"

        fig = self.graph.bar_plot(
                    data,
                    x_axis="dsi",
                    y_axis=self.uuid_col,
                    xlabel="",
                    ylabel="",
                    y_format="%",
                    title=title
                )
        return fig, data
    
    def plot_early_late_revenue_correlation(
        self,
        days_limit: int,
        optimization_window: int=7) -> None:
        """
        Calculates and plots correlation between customer-level revenue
        The correlation is between the revenue in the first [optimization_window] days after registration and the following [days_limit - optimization_window] days
        This can be useful to decide what number of days should define the 'Lifetime Value Window' of a customer
        
        Inputs
             - days_limit: max number of days after registration to be considered in the analysis
             - optimization_window: number of days from registration that the optimization of the marketing campaigns are operated
        """
        # Filters customers to ensure that all have the same opportunity to generate revenue until [days_limits] after registration
        end_events_date = self.joined_df[self.event_time_col].max()
        cohort_filter = (end_events_date - self.joined_df[self.registration_time_col]).dt.days >= days_limit # ensure to only gets cohorts that are 'days_limit' old
        opportunity_filter = (self.joined_df[self.event_time_col] - self.joined_df[self.registration_time_col]).dt.days <= days_limit # only consider the first 'days_limit' days of a customer
        customer_revenue_data = self.joined_df[cohort_filter & opportunity_filter].copy()

        # Create a new dataframe for cross join, so we can ensure that all customers have (days_limit + 1) days to calculate correlation
        days_data = pd.DataFrame({'days_since_install': np.linspace(optimization_window, days_limit, days_limit - optimization_window + 1).astype(np.int32)})
        days_data['key'] = 1
        customer_revenue_data['key'] = 1
        customer_revenue_data = pd.merge(customer_revenue_data, days_data, on='key').drop('key', axis=1)

        # Calculates the revenue of each customer until N days after registration (as the customer doesn't necessarily spend on all days)
        customer_revenue_data[self.value_col] = (customer_revenue_data['days_since_registration'] <= customer_revenue_data['days_since_install']) * customer_revenue_data[self.value_col]
        customer_revenue_data = (
            customer_revenue_data
            .groupby([self.uuid_col, 'days_since_install'])
            [self.value_col].sum().reset_index()
        )
        # Calculate correlation, extract the correlation only for the 'early revenue' and plot it
        customer_revenue_data = customer_revenue_data.pivot(index=self.uuid_col, columns='days_since_install', values=self.value_col).corr().reset_index()
        customer_revenue_data = customer_revenue_data.rename(columns={optimization_window: 'correlation'})
        customer_revenue_data = customer_revenue_data[['days_since_install', 'correlation']]
        
        fig = self.graph.line_plot(
            customer_revenue_data,
            x_axis='days_since_install',
            y_axis='correlation',
            xlabel='Days Since Customer Registration',
            ylabel='Pearson Correlation',
            title=f'Correlation (Y) between revenue until {optimization_window} after registration with revenue until (X) days after registration'
            )
        
        return fig, customer_revenue_data

    def plot_paying_customers_flow(self, days_limit: int, early_limit: int, spending_breaks: Dict[str, float]):
        """
        Plots the flow of customers from early spending class to late spending class
        Inputs:
            days_limit: number of days of the event since registration used to define the late spending class
            early_limit: number of days of the event since registration that is considered 'early'. Usually refers to optimization window of marketing platforms
            spending_breaks: dictionary, in which the keys defines the name of the class and the values the upper limit of the spending associated with the class. Lower limit is considered to be the lower limit of the previous class, else 0
        """

        def classify_spend(x: float):
            key = np.argmax(x <= np.array(list(spending_breaks.values())))
            return list(spending_breaks.keys())[key]


        # Select only customers that are at least [days_limit] days old
        end_events_date = self.joined_df[self.event_time_col].max()
        data = self.joined_df[
            (end_events_date - self.joined_df[self.registration_time_col]).dt.days
            >= days_limit
        ].copy()

        # Remove customers who never had a purchase and ensure all customers have the same opportunity window
        data = data[
            (data[self.event_time_col] - data[self.registration_time_col]).dt.days
            <= days_limit
        ]

        # Count how many purchase customers had (defined by value > 0) and then how many customers are in each place
        data = data[data[self.value_col] > 0]
        data['dsi'] = ((data[self.event_time_col] - data[self.registration_time_col]).dt.days).fillna(0)
        data['early_revenue'] = data.apply(lambda x: (x['dsi'] <= early_limit) * x[self.value_col], axis=1)
        data['late_revenue'] = data.apply(lambda x: (x['dsi'] <= days_limit) * x[self.value_col], axis=1)

        data = (
            data
            .groupby(self.uuid_col)[['early_revenue', 'late_revenue']].sum()
            .reset_index()
        )

        data['early_class'] = data['early_revenue'].apply(classify_spend)
        data['late_class'] = data['late_revenue'].apply(classify_spend)

        def summary(data: pd.DataFrame):
            output  = {}
            output['customers'] = len(data)
            output['early_revenue'] = data['early_revenue'].sum()
            output['early_ltv'] = data['early_revenue'].mean()
            output['median_early_ltv'] = data['early_revenue'].median()
            output['late_revenue'] = data['late_revenue'].sum()
            output['late_ltv'] = data['late_revenue'].mean()
            output['median_late_ltv'] = data['late_revenue'].median()
            return pd.Series(output)
                
        data = (
            data
            .groupby(['early_class', 'late_class'])
            [[self.uuid_col, 'early_revenue', 'late_revenue']]
            .apply(summary)
            .sort_values(['early_ltv', 'late_ltv'])
            .reset_index()
        )
        data['customers'] = data['customers'] / data['customers'].sum()

        def _plot_sankey(sources: list, targets: list, quantity: list) -> go.Figure:
            """
            This method creates a dataframe of flow from 'source' to 'target' based on a given quantity
            It is expected that the data is already orders in a way that it wants to be displayed. If not,
            the quality of the chart will be compromised, as the classes in 'source' to the equivalent 'targets'
            won't be in the same order
            """
            # Define the nodes and links for the Sankey diagram
            nodes = drop_duplicates(sources + targets) # find the unique nodes in both source and targets
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
                return [0.001] * n_nodes + [0.999] * n_nodes # they cannot be 0 and 1 because it overlaps with title
            
            def get_vertical_positions(sources: list, targets: list, quantity: list, classes_gap: float=0.05) -> List[float]:
                """
                Return the vertical positions for the nodes. First values refers to [sources] and last to [targets]
                """
                y_positions_source = {}
                y_positions_target = {}
                # calculate the vertical size of each node
                for i, source in enumerate(sources):
                    y_positions_source[source] = y_positions_source.get(source, 0) + quantity[i]
                    y_positions_target[targets[i]] = y_positions_target.get(targets[i], 0) + quantity[i]
                
                # Calculate where the node's vertical position should end
                y_positions_source = cumsum(list(y_positions_source.values()), constant_delta=classes_gap)
                y_positions_target = cumsum(list(y_positions_target.values()), constant_delta=classes_gap)

                # Lag to get next position, because plotly receives where nodes begin
                y_positions_source = lag(y_positions_source, 1, coalesce=classes_gap)
                y_positions_target = lag(y_positions_target, 1, coalesce=classes_gap)

                return y_positions_source + y_positions_target
            
            def get_nodes_positions(sources: list, targets: list, quantity: list, n_nodes: int) -> (List[float], List[float]):
                return get_horizontal_positions(n_nodes), get_vertical_positions(sources, targets, quantity)

            x_positions, y_positions = get_nodes_positions(sources, targets, quantity, n_nodes)
            # replicate nodes with the same values. Necessary because each
            # value in nodes represent a node. So we have to create 2 nodes with the same names
            nodes = nodes * 2

            node = dict(
                pad=15,
                thickness=20,
                line=dict(color='grey', width=0.5),
                color = palette,
                x=x_positions,
                y=y_positions,
                label=nodes
            )
            link = dict(
                source=sources_idxs,
                target=targets_idxs,
                value=quantity
            )

            # replicate nodes so that the firsts represent the sources and the others the targets
            nodes = nodes * 2 
            # # Create the Sankey diagram
            fig = go.Figure(go.Sankey(
                node=node,
                link=link
            ))
            # Update the layout of the figure
            fig.update_layout(title_text='User Flow Between Classes', font_size=10)
            # Show the figure
            return fig

        return _plot_sankey(data['early_class'].to_list(), data['late_class'].to_list(), data['customers'].to_list()), data
