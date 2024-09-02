# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module providing a class for initial analysis"""

from typing import List, Dict
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype , is_object_dtype, is_any_real_numeric_dtype, is_dtype_equal
import seaborn as sns
from src.graph import Graph, InteractiveChart

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
        self.interactive_chart = InteractiveChart()

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
        assert(is_datetime64_any_dtype(self.data_customers[self.registration_time_col])), f"The column [{self.registration_time_col}] referencing to the registrationtime in the customers dataset was expected to be of type [datetime]. But it is of type {self.data_customers[self.registration_time_col].dtype}"

        # events dataset checks
        assert((isinstance(
            self.data_events[self.uuid_col].dtype, pd.StringDtype) or 
            is_object_dtype(self.data_events[self.uuid_col]))) , f"The column [{self.uuid_col}] referencing to the customer-id in the events dataset was expected to be of data pd.StringDtype or object. But it is of type {self.data_events[self.uuid_col].dtype}"
        assert(is_datetime64_any_dtype(self.data_events[self.event_time_col])) , f"The column [{self.event_time_col}] referencing to the time in the events dataset was expected to be of type [datetime]. But it is of type {self.data_events[self.event_time_col].dtype}"
        assert(is_any_real_numeric_dtype(self.data_events[self.value_col])) , f"The column [{self.value_col}] referencing value of a transaction in the events dataset was expected to be of numeric. But it is of type {self.data_events[self.value_col].dtype}"

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
        optimization_window: int=7,
        interval_size: int=None) -> None:
        """
        Calculates and plots correlation between customer-level revenue
        The correlation is between the revenue in the first [optimization_window] days after registration and the following [days_limit - optimization_window] days
        This can be useful to decide what number of days should define the 'Lifetime Value Window' of a customer
        
        Inputs
             - days_limit: max number of days after registration to be considered in the analysis
             - optimization_window: number of days from registration that the optimization of the marketing campaigns are operated
             - interval_size: number of days between two values shown in the correlation matrix. If None, the method finds the best interval based in the data size
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
        customer_revenue_data = customer_revenue_data.pivot(index=self.uuid_col, columns='days_since_install', values=self.value_col).reset_index()
        customer_revenue_data = customer_revenue_data.drop(self.uuid_col, axis=1).corr()

        # Filter out only some of the days, otherwise there will have too much granularity for visualization
        interval_size = interval_size if interval_size is not None else np.round((days_limit - optimization_window) / 20)
        interval_size = interval_size.astype(int) # doesn't work if applied directly on the output of np.round for some reason
        days_of_interest = [i for i in range(optimization_window, days_limit, interval_size)]
        customer_revenue_data = customer_revenue_data[days_of_interest][customer_revenue_data.index.isin(days_of_interest)]
        mask = np.zeros_like(customer_revenue_data, dtype=bool)
        mask[np.tril_indices_from(mask)] = True

        fig = self.graph.heatmap_plot(customer_revenue_data, title="Correlation matrix of revenue per user by days since registration\n")
        return fig, customer_revenue_data

    @staticmethod
    def _classify_spend(x: float, spending_breaks: Dict[str, float]):
            key = np.argmax(x <= np.array(list(spending_breaks.values())))
            return list(spending_breaks.keys())[key]
    
    def _group_users_by_spend(self, days_limit: int, early_limit: int, spending_breaks: Dict[str, float], end_spending_breaks: Dict[str, float]) -> pd.DataFrame:
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

        # Adding default spending breaks if there was none.
        if len(spending_breaks) == 0:
            zero_mask = data['early_revenue'] != 0
            non_zero_data = data[zero_mask]
            spending_breaks['No spend'] = 0
            spending_breaks['Low spend'] = np.percentile(non_zero_data['early_revenue'], 33.33).round(2)
            spending_breaks['Medium spend'] = np.percentile(non_zero_data['early_revenue'], 66.67).round(2)
            spending_breaks['High spend'] = np.ceil(data['early_revenue'].max())
            print("Starting spending breaks:", spending_breaks)
        
        # Adding default end spending breaks if there was none.
        if len(end_spending_breaks) == 0:
            zero_mask = data['late_revenue'] != 0
            non_zero_data = data[zero_mask]
            end_spending_breaks['No spend'] = 0
            end_spending_breaks['Low spend'] = np.percentile(non_zero_data['late_revenue'], 33.33).round(2)
            end_spending_breaks['Medium spend'] = np.percentile(non_zero_data['late_revenue'], 66.67).round(2)
            end_spending_breaks['High spend'] = np.ceil(data['late_revenue'].max())
            print("Ending spending breaks:", end_spending_breaks)
        

        # Spending breaks needs to be sorted in ascending order
        sorted_spending_breaks = dict(sorted(spending_breaks.items(), key=lambda x: x[1]))
        sorted_end_spending_breaks = dict(sorted(end_spending_breaks.items(), key=lambda x: x[1]))

        data['early_class'] = data['early_revenue'].apply(lambda x: self._classify_spend(x, sorted_spending_breaks))
        data['late_class'] = data['late_revenue'].apply(lambda x: self._classify_spend(x, sorted_end_spending_breaks))

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
                
        return (
            data
            .groupby(['early_class', 'late_class'])
            [[self.uuid_col, 'early_revenue', 'late_revenue']]
            .apply(summary)
            .sort_values(['early_ltv', 'late_ltv'])
            .reset_index()
        )
    
    def plot_paying_customers_flow(self, days_limit: int, early_limit: int, spending_breaks: Dict[str, float], end_spending_breaks: Dict[str, float]):
        """
        Plots the flow of customers from early spending class to late spending class
        Inputs:
            days_limit: number of days of the event since registration used to define the late spending class
            early_limit: number of days of the event since registration that is considered 'early'. Usually refers to optimization window of marketing platforms
            spending_breaks: dictionary, in which the keys defines the name of the class and the values the upper limit of the spending associated with the class. Lower limit is considered to be the lower limit of the previous class, else 0
            end_spending_breaks: dictionary, in which the keys defines the name of the class and the values the upper limit of the spending associated with the class. Lower limit is considered to be the lower limit of the previous class, else 0
        """

        data = self._group_users_by_spend(days_limit, early_limit, spending_breaks, end_spending_breaks)
        data['customers'] = data['customers'] / data['customers'].sum()

        self.interactive_chart.legend_out = True
        fig = self.interactive_chart.flow_chart(data, 'early_class', 'late_class', 'customers', title='User Flow Between Classes')

        return fig, data


    def estimate_ltv_impact(
            self,
            days_limit: int,
            spending_breaks: Dict[str, float],
            population_increase: Dict[str, float]
            ):
        """
        Estimate the impact of using a predicted LTV (pLTV) strategy for campaign optimization.
        The estimate is calculated by increasing the number of customers according to [population_increase] and their
        classification at a later moment [days_limit]. 
        The impact is calculated by seeing the increase in revenue at a late date [days limit] caused by this increase
        in the number of users.
        The classification of users follow the same principle of the method [plot_paying_customers_flow()]

        Inputs:
            - days_limit: the number of days since registration to define as the 'late' revenue classification of a 
                           customer. Usually this value is considered the effective number of days for the LTV
            - spending_breaks: the classification of a user depending on its revenue
            - population_increase: a dictionary of dictionaries. The initial keys refers the early revenue classification
                                   and the values the relative increase for them

        Example:
            population_increase = {
                'Low Spend': 0.1, 
                'Medium Spend: 0.2,
                'High Spend: 0.05
                }
            In this example, we assume that the LTV Optimization is going to:
                - increase the number of users that generated low revenue by 10%
                - increase the number of users that at the had medium revenue by 20%
                - increase the number of users that already had high revenue by 5%
                - all other classifications not specified remain unchaged
            
        """
        # Get users grouped by their early and late revenue
        data = self._group_users_by_spend(days_limit, 0, spending_breaks.copy(), spending_breaks.copy())

        # Apply the average LTV of the highest-spending class to all spending classes
        data['assumed_new_late_revenue'] = data['late_revenue'] * (data['late_class'].map(population_increase).fillna(0) + 1)
        data['assumed_new_customers'] = data['customers'] * (data['late_class'].map(population_increase).fillna(0) + 1)
        data = data[['early_class', 'late_class', 'customers', 'late_revenue', 'assumed_new_customers', 'assumed_new_late_revenue']]
        data['abs_revenue_increase'] = data['assumed_new_late_revenue'] - data['late_revenue']

        abs_impact = np.sum(data['abs_revenue_increase'])
        rel_impact = abs_impact / np.sum(data['late_revenue'])

        output_txt = f"""
        By adopting a predicted LTV (pLTV) based strategy for your marketing campaigns, we estimate up to {100*rel_impact:.1f}% increase in revenue.
        This increase represents ${abs_impact:.0f} in revenue for the time period and scope used by the data provided.  
        We find this  impact of the pLTV strategy by assuming an increase in the number of paying customers as passed down by argument [population_increase], and assuming that (1) the average LTVs don't change, and (2) the number of users in other classes don't change
        """
        print(output_txt)
        
        return data