# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import accuracy_score, r2_score
from IPython.display import display

class LTVexploratory:
    """This class helps to perform some initial analyses
    """

    def __init__(self,
                 data_ancor,
                 data_events,
                 initial_period = 7,
                 ltv_horizon = 70):
        self.data_ancor = data_ancor
        self.data_events = data_events
        self._period = initial_period
        # TODO - define ltv_max_horizon based on data
        self._ltv_horizon = ltv_horizon
        self._prep_df()
        self._prep_LTV_periods()
        self._prep_payer_types()

 # Data prep 

    def _prep_df(self) -> None:
        ancor_uuid = set(self.data_ancor['UUID'])
        event_uuid = set(self.data_events['UUID'])
        uuid_stat = pd.DataFrame(columns=['table', 'number_uuid'])
        uuid_stat['table'] = ['ancor only', 'event only', 'ancor & event']
        uuid_stat['number_uuid'] = [len(ancor_uuid - event_uuid), len(event_uuid - ancor_uuid), len(event_uuid & ancor_uuid)]
        uuid_stat['share_uuid'] = uuid_stat['number_uuid'].map(lambda x: f"{x/uuid_stat['number_uuid'].sum():.2%}")

        display(uuid_stat.rename(columns={"table": "Source table", "number_uuid": "Number of UUID", "share_uuid" : "Shared UUID between tables"}))

        data_upload_date = self.data_ancor['time_of_the_event'].max()
        df = self.data_ancor.merge(self.data_events, on='UUID', how='inner', suffixes=('_ancor', '_event'))
        df['max_days_for_ltv'] = (data_upload_date - df['time_of_the_event_ancor']).dt.days + 1
        df['year-month'] = df['time_of_the_event_event'].dt.year.astype(str) + '-' + ("0" + df['time_of_the_event_event'].dt.month.astype(str)).str[-2:]
        df['days_between_purchase_and_reg'] = (pd.to_datetime(df['time_of_the_event_event'].dt.date) -
                                            pd.to_datetime(df['time_of_the_event_ancor'].dt.date)).dt.days
        df['time_of_first_purchase'] = pd.to_datetime(df['UUID'].map(df.groupby('UUID')['time_of_the_event_event'].min()))
        df['days_after_first_purch'] = (df['time_of_the_event_event'] - df['time_of_first_purchase']).dt.days

        # Delete uuid-outliers
        uuid_for_deletion = df.loc[df['days_between_purchase_and_reg'] < 0, 'UUID'].unique()
        print(f'{len(uuid_for_deletion)} ({len(uuid_for_deletion)/df["UUID"].nunique():.2%}) UUID with a first purchase date that predated registration have been removed from the analysis dataset.')
        df = df[~df['UUID'].isin(uuid_for_deletion)].reset_index(drop=True)
        self._df = df

    def _prep_LTV_periods(self) -> None:
        # calculate ltv by days for each uuid
        periods = np.append([1,3,5], np.arange(7, self._ltv_horizon*2+7, 7))
        # If the two user-defined values are not in the period list, add them
        if np.any(periods == self._period) == False:
            periods = np.append(periods, self._period)
        if np.any(periods == self._ltv_horizon) == False:
            periods = np.append(periods, self._ltv_horizon)

        periods = np.sort(periods)    
        for period in periods:
            self._df[period] = self._df['purchase_value'] *(self._df['days_between_purchase_and_reg'] <= period)
            # if uuid doesn't have data for calculating ltv for this period (for example if our user joined only 1 day ago, he doesn't have enough data in order to calculate ltv for 7 days), I want to see None in this cell, because it will be fair
            self._df.loc[self._df['max_days_for_ltv'] < period, period] = None
        ltv_periods = self._df.groupby(['UUID'])[periods].sum()
        # function sum converts None to 0, so I use proxy function which set None where is necessary
        ltv_periods[self._df.groupby(['UUID'])[periods].max().isna()] = None
        self._ltv_periods = ltv_periods


    def _prep_payer_types(self):
        payers_types = pd.DataFrame()
        for col in self._ltv_periods.columns:
            quatile_for_type_of_payers = np.unique(np.append(0, self._ltv_periods.loc[self._ltv_periods[col] > 0, col].quantile(np.linspace(0, 1, 3 + 1))))
            while len(quatile_for_type_of_payers) < 5:
                quatile_for_type_of_payers = np.append(quatile_for_type_of_payers, max(quatile_for_type_of_payers) + 1)
            payers_types[col] = pd.cut(self._ltv_periods[col], bins=quatile_for_type_of_payers,
            labels=['0.users_with_zero_purchases', '1.low_payers', '2.mid_payers', '3.high_payers'], include_lowest=True)
        self._payer_types = payers_types


# Analysis

    def summary(self):
            print(f"""
            table:      data_ancor
            date start: {self.data_ancor['time_of_the_event'].min()}
            date end:   {self.data_ancor['time_of_the_event'].max()}
            period:     {(self.data_ancor['time_of_the_event'].max()-self.data_ancor['time_of_the_event'].min()).days/365:.2f} years
            customers:  {self.data_ancor['UUID'].nunique():,d}
            events:     {self.data_ancor.shape[0]:,d}
            """)
            display(self.data_ancor.head())

            print(f"""  
            table:      self.data_event
            date start: {self.data_events['time_of_the_event'].min()}
            date end:   {self.data_events['time_of_the_event'].max()}
            period:     {(self.data_events['time_of_the_event'].max()-self.data_events['time_of_the_event'].min()).days/365:.2f} years
            customers:  {self.data_events['UUID'].nunique():,d}
            events:     {self.data_events.shape[0]:,d}
            """)
            display(self.data_events.head())

# Plotting

    def plot_n(self):
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        self.data_ancor['ancor_event_name'].value_counts().plot.barh(ax=ax[0])
        ax[0].set_title('event_name from df_ancore')
        self.data_events['event_name'].value_counts().plot.barh(ax=ax[1])
        ax[1].set_title('event_name from df_event')
        plt.show()
 
    def plot_first_purchase(self):
        fig, ax = plt.subplots(figsize=(20, 5))
        days_before_first_purchase = self._df.groupby('UUID')['days_between_purchase_and_reg'].min()
        days_before_first_purchase.hist(ax=ax, bins=50)
        plt.title('Days between registration and the first purchase')
        plt.ylabel('Users')
        plt.xlabel('Days')
        plt.show()
        pd.set_option('display.float_format', lambda x: '%.1f' % x)
        display(days_before_first_purchase.reset_index().rename({"days_between_purchase_and_reg": "Days between registration and the first purchase stat:"}, axis=1).describe())


    def plot_second_purchase(self):
        # Let's have a look on days before the second purchase
        fig, ax = plt.subplots(figsize=(20, 5))
        days_after_first_purchase = self._df[self._df['days_after_first_purch'] > 0].groupby('UUID')['days_after_first_purch'].min()
        days_after_first_purchase.hist(ax=ax, bins=50)
        plt.title('Days between the first purchase and the next one')
        plt.ylabel('Users')
        plt.xlabel('Days')
        plt.show()
        pd.set_option('display.float_format', lambda x: '%.1f' % x)
        stat = days_after_first_purchase.reset_index().rename({"days_after_first_purch": "Days between the first purchase and the next one stat:"}, axis=1).describe()
        stat.loc['users without second purchase'] = f"{(self._df['UUID'].nunique() - self._df.loc[self._df['days_after_first_purch'] > 0, 'UUID'].nunique())/self._df['UUID'].nunique():.2%}"
        display(stat)
        print(f"""You have {(self._df['purchase_value'] == 0).sum()/self._df.shape[0]:.2%} purchases with zero value""")

    def plot_uuid(self):
        # Let's have a look at distribution of uuid life time
        fig, ax = plt.subplots(figsize=(20, 5))
        n, bins, patches = ax.hist(self._df.groupby('UUID')['max_days_for_ltv'].min(), bins=30)
        ax.set_title('Life time distribution of users who have purchases history')
        plt.show()

    def plot_linear_corr(self):
        def r2_score_standardised(a, b):
            a = a[(~np.isnan(a)) & (~np.isnan(b))]
            b = b[(~np.isnan(a)) & (~np.isnan(b))]
            if len(a) > 2:
                return r2_score((a-np.mean(a))/(np.std(a) if np.std(a) != 0 else 1),
                                (b-np.mean(b))/(np.std(b) if np.std(b) != 0 else 1))
            return np.nan

        # Calculate linear correlation
        print("The main idea of this chart: Do we have a strong linear correlation in order to build linear model? (1 - yes, 0 - no)")
        _, ax = plt.subplots(figsize=(30, 15))
        sns.heatmap(self._ltv_periods.corr(), annot=True,
                    cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True),
                    ax=ax)
        plt.title('Linear correlation between LTV for different days (Pearson)')

        print("""The main idea of this chart: if we take (ltv value for 7 days) as a feture to our linear model in order to predict (ltv value for day 14)
        how well the regression model explains observed data (proportion of the variation in the dependent variable that is predictable from the independent variable)?""")
        _, ax = plt.subplots(figsize=(30, 15))
        cor = self._ltv_periods.corr(method=r2_score_standardised)
        # r2 < 0 means that it's better to get a constant as a feature than ltv for chosing period
        cor[cor < 0 ] = 0
        sns.heatmap(cor, annot=True,
                    cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True),
                    ax=ax)
        plt.title('R2 between normalised LTV for different days')

    def plot_sankey(self, sankey_initial_period=None, sankey_ltv_horizon=None):
        if sankey_initial_period is None:
            sankey_initial_period = self._period

        if sankey_ltv_horizon is None:
            sankey_ltv_horizon = self._ltv_horizon
        
        ind = ~self._payer_types[sankey_ltv_horizon].isna()
        pivot_count_table = self._payer_types.loc[ind, [sankey_initial_period, sankey_ltv_horizon]].astype(str).reset_index().pivot_table(index=sankey_initial_period, columns=sankey_ltv_horizon,
        values='UUID', aggfunc='count').fillna(0)
        pivot_count_table = pivot_count_table/pivot_count_table.sum().sum()
        source = sum([[x]*len(pivot_count_table.columns) for x in pivot_count_table.index.to_list()], [])
        target = pivot_count_table.columns.to_list() * len(pivot_count_table.index)
        value = [pivot_count_table.loc[x,y]*100 for x,y in zip(source, target)]
        dict_translate = {'0.users_with_zero_purchases': {'color': 'red', 'color_link': 'rgba(255, 204, 203, 0.8)', 'ind':0},
                        '1.low_payers': {'color': 'orange', 'color_link': 'rgba(254, 216, 177, 0.8)', 'ind':1},
                        '2.mid_payers': {'color': 'blue', 'color_link': 'rgba(173, 216, 230, 0.8)', 'ind':2},
                        '3.high_payers': {'color': 'green', 'color_link': 'rgba(144, 238, 144, 0.8)', 'ind':3}}
        label = list(dict_translate.keys())
        labels = [x for x in label if x in pivot_count_table.sum(axis=1).index] + [x for x in label if x in pivot_count_table.sum(axis=0).index]
        value_for_nodes = [f"{x[2:]} ({pivot_count_table.sum(axis=1)[x]:.1%})" for x in label if x in pivot_count_table.sum(axis=1).index
                        ] + [f"{x[2:]} ({pivot_count_table.sum(axis=0)[x]:.1%})" for x in label if x in pivot_count_table.sum(axis=0).index]
        fig = go.Figure(data=[go.Sankey(
            valueformat = ".0f",
            valuesuffix = "TWh",
            # Define nodes
            node = dict(
            pad = 15,
            thickness = 15,
            line = dict(color = "black", width = 0.5),
            label =  value_for_nodes,
            color =  [dict_translate[x]['color'] for x in labels]
            ),
            # Add links
            link = dict(
            source =  [dict_translate[x]['ind'] for x in source],
            target =  [dict_translate[x]['ind']+len(pivot_count_table.index) for x in target],
            value =  value,
            label =  [f"{x/100:.1%}" for x in value],
            color =  [dict_translate[x]['color_link'] for x in target]
        )
        )])
        for x_coordinate, column_name in enumerate([f"{sankey_initial_period}-day Revenue", f"{sankey_ltv_horizon}-day Revenue"]):
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
                        color="black"
                        ),
                    align="center",
                    )

        fig.update_layout(title_text="Opportunity size for adopting a pLTV strategy",
        xaxis={
        'showgrid': False, # thin lines in the background
        'zeroline': False, # thick line at x=0
        'visible': False,  # numbers below
        },
        yaxis={
        'showgrid': False, # thin lines in the background
        'zeroline': False, # thick line at x=0
        'visible': False,  # numbers below
        }, plot_bgcolor='rgba(0,0,0,0)', font_size=10, title_x=0.5)
        fig.show()

    def plot_class_change(self):
        def accuracy_score_for_not_na_values(a, b):
            return accuracy_score(a[(~np.isnan(b)) & (~np.isnan(a))], b[(~np.isnan(b)) & (~np.isnan(a))])

        print("Let's look at this value (how many users will have the same classification flag if we compare different periods between each other)")
        _, ax = plt.subplots(figsize=(30, 15))
        cor = self._payer_types.replace({'0.users_with_zero_purchases':0,
                                    '1.low_payers':1,
                                    '2.mid_payers':2,
                                    '3.high_payers':3}).corr(method=accuracy_score_for_not_na_values)
        sns.heatmap(cor, annot=True,
                    cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True),
                    ax=ax)
        plt.title("Share of users who didn't change their classification flag (no-payers, low-payer, mid-payer, high-payer) across periods")


    def plot_drop_off(self):
        # Let's have a look at distribution of number uuid which can be in the train set depending on life time which we will chose for our model
        fig, ax = plt.subplots(figsize=(30, 5))
        _data = self._df.groupby('UUID')['max_days_for_ltv'].min()
        ax.hist(_data, bins=1000, cumulative=-1)
        ax.set_title('Number of users which we can use in train set depend on horizon which we choose')
        ax.set_xticks(np.arange(_data.min(), _data.max(), 50))
        plt.show() 



    def plot_(self):
        # We have users with different life times, it means that building LTV distribution histogram for all users together will be not good.
        # Because in this case we will compare ltv for a user who have couple years purchase history with a user who joined a couple days ago. It doesn't make sense.
        # So the main idea is to see distribution for users groupped by almost the same number of active days on the platform
        active_user_stats = self._df.groupby('UUID').agg({'max_days_for_ltv': 'min', 'purchase_value': 'sum'})
        lifeteime_quantiles = active_user_stats['max_days_for_ltv'].quantile(np.linspace(0, 1, 20 + 1))
        active_user_stats['lifetime_bins'] = pd.cut(active_user_stats['max_days_for_ltv'],
                                                    bins=lifeteime_quantiles,
                                                    labels=lifeteime_quantiles.iloc[:-1].astype(str) + ' - ' + lifeteime_quantiles.shift(-1).iloc[:-1].astype(str),
                                                    include_lowest=True)
        _, ax = plt.subplots(figsize=(40,10), dpi=50)
        sns.boxplot(data=active_user_stats, x='lifetime_bins', y='purchase_value', ax=ax, color='orange')
        # sns.violinplot(data=active_user_stats, x="lifetime_bins", y="purchase_value")
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel('Active days bins')
        ax.set_ylabel('LTV purchases')
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(data=active_user_stats.groupby('lifetime_bins')['purchase_value'].median().reset_index(), x='lifetime_bins', y='purchase_value', ax=ax2)
        ax2.set_ylabel('median LTV purchases')
        plt.title("LTV distribution depending on user's active days")
        plt.show()
        active_user_stats['outlier_line'] = active_user_stats['lifetime_bins'].map(active_user_stats.groupby('lifetime_bins')['purchase_value'].quantile(0.95)).astype(np.float64)
        active_user_stats = active_user_stats[active_user_stats['purchase_value'] <= active_user_stats['outlier_line']]
        _, ax = plt.subplots(figsize=(40,10), dpi=50)
        # sns.boxplot(data=active_user_stats, x='lifetime_bins', y='purchase_value', ax=ax, color='orange')
        sns.violinplot(data=active_user_stats, x="lifetime_bins", y="purchase_value", width=1, linewidth=1)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel('Active days bins')
        ax.set_ylabel('LTV purchases')
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(data=active_user_stats.groupby('lifetime_bins')['purchase_value'].median().reset_index(), x='lifetime_bins', y='purchase_value', ax=ax2)
        ax2.set_ylabel('median LTV purchases')
        plt.title("LTV distribution depending on user's active days (without outliers in each group)")
        plt.show()


    def plot_freq(self):
        # Let's have a look on the first component of LTV: frequency
        active_user_stats = self._df.groupby('UUID').agg({'max_days_for_ltv': 'min', 'purchase_value': 'count'})
        active_user_stats['purchase_value'] = active_user_stats['purchase_value']/active_user_stats['max_days_for_ltv'] * 30
        lifeteime_quantiles = active_user_stats['max_days_for_ltv'].quantile(np.linspace(0, 1, 20 + 1))
        active_user_stats['lifetime_bins'] = pd.cut(active_user_stats['max_days_for_ltv'],
                                                    bins=lifeteime_quantiles,
                                                    labels=lifeteime_quantiles.iloc[:-1].astype(str) + ' - ' + lifeteime_quantiles.shift(-1).iloc[:-1].astype(str),
                                                    include_lowest=True)

        _, ax = plt.subplots(figsize=(40,10), dpi=50)
        sns.boxplot(data=active_user_stats, x='lifetime_bins', y='purchase_value', ax=ax, color='orange')
        # sns.violinplot(data=active_user_stats, x="lifetime_bins", y="purchase_value")
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel('Active days bins')
        ax.set_ylabel('Purchases frequency (number purchases per 30 days)')
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(data=active_user_stats.groupby('lifetime_bins')['purchase_value'].median().reset_index(), x='lifetime_bins', y='purchase_value', ax=ax2)
        ax2.set_ylabel('median Purchases frequency (number purchases per 30 days)')
        plt.title("Purchases frequency distribution depending on user's active days")
        plt.show()

        active_user_stats['outlier_line'] = active_user_stats['lifetime_bins'].map(active_user_stats.groupby('lifetime_bins')['purchase_value'].quantile(0.95)).astype(np.float64)
        active_user_stats = active_user_stats[active_user_stats['purchase_value'] <= active_user_stats['outlier_line']]
        _, ax = plt.subplots(figsize=(40,10), dpi=50)
        # sns.boxplot(data=active_user_stats, x='lifetime_bins', y='purchase_value', ax=ax, color='orange')
        sns.violinplot(data=active_user_stats, x="lifetime_bins", y="purchase_value", width=1, linewidth=1)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel('Active days bins')
        ax.set_ylabel('Purchases frequency (number purchases per 30 days)')
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(data=active_user_stats.groupby('lifetime_bins')['purchase_value'].median().reset_index(), x='lifetime_bins', y='purchase_value', ax=ax2)
        ax2.set_ylabel('median Purchases frequency (number purchases per 30 days)')
        plt.title("Purchases frequency distribution depending on user's active days (without outliers in each group)")
        plt.show()


    def plot_apv(self):
        active_user_stats = self._df.groupby('UUID').agg({'max_days_for_ltv': 'min', 'purchase_value': 'median'})
        lifeteime_quantiles = active_user_stats['max_days_for_ltv'].quantile(np.linspace(0, 1, 20 + 1))
        active_user_stats['lifetime_bins'] = pd.cut(active_user_stats['max_days_for_ltv'],
                                                    bins=lifeteime_quantiles,
                                                    labels=lifeteime_quantiles.iloc[:-1].astype(str) + ' - ' + lifeteime_quantiles.shift(-1).iloc[:-1].astype(str),
                                                    include_lowest=True)

        _, ax = plt.subplots(figsize=(40,10), dpi=50)
        sns.boxplot(data=active_user_stats, x='lifetime_bins', y='purchase_value', ax=ax, color='orange', width=1, linewidth=1)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel('Active days bins')
        ax.set_ylabel('Purchase value')
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(data=active_user_stats.groupby('lifetime_bins')['purchase_value'].median().reset_index(), x='lifetime_bins', y='purchase_value', ax=ax2)
        ax2.set_ylabel('median purchase value purchases')
        plt.title("Purchase value distribution depending on user's active days")
        plt.show()

        active_user_stats['outlier_line'] = active_user_stats['lifetime_bins'].map(active_user_stats.groupby('lifetime_bins')['purchase_value'].quantile(0.95)).astype(np.float64)
        active_user_stats = active_user_stats[active_user_stats['purchase_value'] <= active_user_stats['outlier_line']]
        _, ax = plt.subplots(figsize=(40,10), dpi=50)
        # sns.boxplot(data=active_user_stats, x='lifetime_bins', y='purchase_value', ax=ax, color='orange')
        sns.violinplot(data=active_user_stats, x="lifetime_bins", y="purchase_value", width=1, linewidth=1)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel('Active days bins')
        ax.set_ylabel('Purchase value')
        # ax.set( yscale="log")
        ax2 = ax.twinx()
        sns.lineplot(data=active_user_stats.groupby('lifetime_bins')['purchase_value'].median().reset_index(), x='lifetime_bins', y='purchase_value', ax=ax2)
        ax2.set_ylabel('median purchase value purchases')
        plt.title("Purchase value distribution depending on user's active days (without outliers in each group)")
        plt.show()


    def plot_corr_freq_apv(self):
        print('''Interpretation: Correlation value can be [-1,1]. If you have correlation closer to 0, it means that you can use models where these two variables describes different models.''')
        _, ax = plt.subplots(figsize=(30,10), dpi=50)
        stat = self._df.groupby('UUID')[['purchase_value', 'max_days_for_ltv']].agg({'purchase_value': ['mean', 'count'], 'max_days_for_ltv': 'min'})
        stat[('purchase_value', 'count')] = stat[('purchase_value', 'count')] / stat[('max_days_for_ltv', 'min')] * 30
        stat['purchase_value'].plot.scatter(x='mean', y='count', ax=ax)
        ax.set_ylabel('frequencuy')
        ax.set_xlabel('mean purchase')
        plt.title("Check independancy frequency from purchase value")
        plt.show()
        print(f"linear_corr={stat['purchase_value']['mean'].corr(stat['purchase_value']['count']):.2f}")
        # stat['purchase_value'].phik_matrix() <- not availible, only in local notebooks (for this data it equals 0 - we don't have correlation between these values).
        # Let's delete outliers, because now it seems that we have dependancy 'as higher price, as lower frequency'
        _, ax = plt.subplots(figsize=(30,10), dpi=50)
        stat = stat[(stat[('purchase_value', 'count')] <= stat[('purchase_value', 'count')].quantile(0.99)) &
            (stat[('purchase_value', 'mean')] <= stat[('purchase_value', 'mean')].quantile(0.99))]
        stat['purchase_value'].plot.scatter(x='mean', y='count', ax=ax)
        ax.set_ylabel('frequencuy')
        ax.set_xlabel('mean purchase')
        plt.title("Check independancy frequency from purchase value (without outliers)")
        plt.show()
        print(f"linear_corr={stat['purchase_value']['mean'].corr(stat['purchase_value']['count']):.2f}")
        # stat['purchase_value'].phik_matrix() <- not availible, only in local notebooks (for this data it equals 0.14 - we don't have correlation between these values).
