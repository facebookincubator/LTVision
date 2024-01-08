# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple
from scipy.stats import logistic, pareto

class LTVSynthetiseData():
    def __init__(self, 
                 n_users: int=100000,
                 registration_event_name: str='first_app_open',
                 purchase_event_name: str='purchase') -> None:
        self.n_users = n_users
        self.registration_event_name = registration_event_name
        self.purchase_event_name = purchase_event_name
        self.customer_data = None

    def get_customers_data(self):
        """
        Get dataframe containing base costumer data, such as their user id, country, device, and download method for the app
        """
        customer_data = self._get_base_user_ids()
        customer_data = self._set_registration_event(customer_data) # create registration_name column
        customer_data = self._set_demographic_properties(customer_data) # create and give demographic data
        self.customer_data = customer_data
        return customer_data

    def get_events_data(self):
        raise NotImplementedError

    def _set_logit_map(self):
        raise NotImplementedError

    @staticmethod
    def _sample_from_pareto(expected_value: float, size: int=None) -> np.ndarray:
        """
        Generate (size) samples of a Pareto distribution with expected value (expected_value)
        Inputs
            - expected_value: desired expected value of the Pareto distribution 
            - size: number of samples to generate
        """
        alpha = expected_value / (expected_value - 1)
        return (np.random.pareto(alpha, size=size) + 1)
    
    @staticmethod
    def _sample_from_lognormal(expected_value: float, stddev: float, size: int=None) -> np.ndarray:
        """
        Generate (size) samples of a log-normal distribution with expected value (expected_value)
        and standard deviation (stddev). 
        Inputs
            - expected_value: desired expected value of the log-normal distribution 
            - stddev: desired standard deviation of the log-normal distribution
            - size: number of samples to generate
        """
        mean = np.log(expected_value/(
            ((stddev**2)/(expected_value**2) + 1
            )**0.5))
        sigma = np.log((stddev**2)/(expected_value**2)+ 1)**0.5
        return np.random.lognormal(mean, sigma, size=size)
    
    @staticmethod
    def _sample_from_beta(expected_value: float, beta_samples: int, size:int=None) -> np.ndarray:
        """
        Inputs
            - expected_value: desired expected value of the Beta distribution, i.e. the probability positives/(positives+negatives)
            - beta_samples: positives + negatives
            - size: number of samples to generate
        """
        positives = expected_value * beta_samples
        negatives = beta_samples - positives
        return np.random.beta(positives, negatives, size=size)
    
    def _calculate_event_conditional_value(self, events_data: pd.DataFrame):
        """
        Returns for each user, the value a purchase *conditional* to the characteristics
        defined in the self.value_map, such as
            - country
            - device
            - days since registration
        We do this by first calculating the conditional expected value
        E(x | country, device, ...)
        and then samplying based on a given distribution
        """
        expected_value = events_data.apply(self.value_map).sum(axis=1)
        return expected_value
    
    def _calculate_probability(self, events_data: pd.DataFrame) -> np.ndarray:
        logits = events_data.apply(self.logit_map).sum(axis=1)
        return logistic.cdf(logits)
    
    def _get_base_user_ids(self):
        """
        Generate a sequence of 1 to N numbers representing the unique user ID, where N is the number of users
        """
        return pd.DataFrame({'UUID': np.linspace(1, self.n_users, num=self.n_users).astype(np.int64)})
    
    @staticmethod
    def _set_registration_event(customer_data: pd.DataFrame, event_name: str) -> pd.DataFrame:
        customer_data['registration_event_name'] = event_name
        return customer_data

    
    @staticmethod
    def _set_demographic_properties(customer_data: pd.DataFrame) -> pd.DataFrame:

        size = len(customer_data)
        demography_data =  pd.DataFrame({
            'country': np.random.choice(['US', 'CA', 'GB', 'BR', 'IN', 'ES', 'FR'], size=size, p=[0.2, 0.1, 0.05, 0.15, 0.3, 0.1, 0.1], replace=True),
            'device': np.random.choice(['ios', 'android'], size=size, p=[0.4, 0.6], replace=True),
            'download_method': np.random.choice(['wifi', 'roaming'], size=size, p=[0.7, 0.3], replace=True),
        })
        return pd.concat([customer_data, demography_data], axis=1)
    
    def _set_dates(self, events_data: pd.DataFrame) -> None:
        raise NotImplementedError
    
    def _set_conversion_prob(self, events_data: pd.DataFrame) -> None:
        raise NotImplementedError
    
    def _set_conditional_return_prob(self, events_data: pd.DataFrame) -> None:
        raise NotImplementedError
    
    def _set_conditional_purchase_prob(self, events_data: pd.DataFrame) -> None:
        raise NotImplementedError
    
    def _set_conditional_purchase_value(self, events_data: pd.DataFrame) -> None:
        raise NotImplementedError
    



class LTVSyntheticData:
    """This class helps to create sample of random data which can be used in
    LTV package

    Parameters:
    ----------------------------------------------------------------------------
    n_users: int, how many users there will be in your data
    share_payers: float, share of active users or share of users, who make
                         purchases value must be in the interval (0,1]
    n_days: int, how many days is in data sample
    share_high_payers: float, share of active users, who has purchases higher
                               than others
    share_low_payers: float, share of active users, who has purchases lower
                             than others
    date_start: str, the minimum date in the sample data in format '%Y-%m-%d'
    is_subscr: bool, do you want to generate subscribtion data? (otherwise it
                     will be comm)
    mean_for_value: float, mean value for purchase distribution, must be
                           positive (required if is_subscr = False)
    std_for_value: float, std value for purchase distribution, must be positive
                          (required if is_subscr = False)
    ----------------------------------------------------------------------------
    mean_for_day_number: float, mean value for number of purchases during a day,
                                must be positive (required if is_subscr = False)
    std_for_day_number: float, std value for number of purchases during the day,
                               must be positive (required if is_subscr = False)
    payment_for_each_subscr_option: tuple[float], payment for each subscribtion
                                                  option (required if is_subscr
                                                  = True)
    days_for_each_subscr_option: tuple[float], subscribtion period for each
                                               subscribtion option (required if
                                               is_subscr = True)
    ----------------------------------------------------------------------------
    """
    def __init__(self,
                n_users: int = 10000,
                share_payers: float = 0.2,
                n_days: int = 365,
                share_high_payers: float = 0.2,
                share_low_payers: float = 0.5,
                date_start: str = '2020-01-01',
                is_subscr: bool = True,
                mean_for_value: float = 10.,
                std_for_value: float = 5.,
                mean_for_day_number: float = 3.,
                std_for_day_number: float = 4.,
                payment_for_each_subscr_option: Tuple[float] = (1.0, 5.0, 9.0),
                days_for_each_subscr_option: Tuple[int] = (30, 60, 90)
                ):
        self.n_users = n_users
        self.share_payers = share_payers
        self.n_days = n_days
        self.share_high_payers = share_high_payers
        self.share_low_payers = share_low_payers
        self.date_start = date_start
        self.is_subscr = is_subscr
        self.mean_for_value = mean_for_value
        self.std_for_value = std_for_value
        self.mean_for_day_number = mean_for_day_number
        self.std_for_day_number = std_for_day_number
        self.payment_for_each_subscr_option = payment_for_each_subscr_option
        self.days_for_each_subscr_option = days_for_each_subscr_option

        self._uuids = None
        self._active_uuid = None
        self._high_payers = None
        self._low_payers = None
        self._mid_payers = None
        self._ancor_table = None

        self._check_inputs()

        self._generate_uuids()

    def _check_inputs(self):
        """Simple check for input values"""
        if self.share_payers > 1:
            raise Exception('share_payers must be in the interval (0,1]')
        if self.share_high_payers + self.share_low_payers > 1:

            raise Exception('share_high_payers + share_low_payers must be in ' +
                            'the interval [0,1]')
        if self.is_subscr:
            if len(self.payment_for_each_subscr_option) != len(
                self.days_for_each_subscr_option):
                raise Exception('Parameters payment_for_each_subscr_option ' +
                                'and days_for_each_subscr_option must have ' +
                                'the same size')
        else:
            for name, val in zip(['mean_for_value', 'std_for_value',
                                  'mean_for_day_number', 'std_for_day_number'],
                           [self.mean_for_value, self.std_for_value,
                            self.mean_for_day_number, self.std_for_day_number]):
                if val < 0:
                    raise Exception(
                        f'Parameters {name} must  be more than 0')

    def _generate_uuids(self) -> None:
        """Generate uuids according to n_users parameter"""
        self._uuids = [f'#{x}' for x in np.arange(self.n_users)]

    def get_ancor_table(self, force_update: bool = False) -> pd.DataFrame:
        """Create ancor table

        Parameters:
        ------------------------------------------------------------------------
        force_update: bool, do you want to force update the ancor table?
        ------------------------------------------------------------------------

        Returns
        ------------------------------------------------------------------------
        pd.DataFrame(columns=[UUID, timestamp, ancor_event_name]
                     ): table with registration date
        ------------------------------------------------------------------------
        """
        if self._ancor_table is None or force_update:
            registration_day = np.random.choice(np.arange(self.n_days),
                                                size=self.n_users)
            self._ancor_table = pd.DataFrame(
                {'UUID': self._uuids, 'timestamp': registration_day})
            self._ancor_table['ancor_event_name'] = 'registration'
            self._ancor_table['timestamp'] = (
                pd.to_datetime(self.date_start) + pd.to_timedelta(
                    self._ancor_table['timestamp'], unit='D'))
        return self._ancor_table

    def _define_active_users(self) -> None:
        """Choose active_uuid, high_payers, mid_payers and low_payers"""
        self._active_uuid = np.random.choice(
            self._uuids,
            size=int(self.n_users*self.share_payers),
            replace=False)
        self._high_payers = np.random.choice(
            self._active_uuid,
            size=int(len(self._active_uuid)*self.share_high_payers),
            replace=False)
        self._low_payers = np.random.choice(
            list(set(self._active_uuid)-set(self._high_payers)),
            size=int(len(self._active_uuid)*self.share_low_payers),
            replace=False)
        self._mid_payers = np.array(list(set(self._active_uuid) -
                                         set(self._high_payers) -
                                         set(self._low_payers)))

    @staticmethod
    def _get_payments_subscr(active_uuid: list,
                             registration_table: pd.DataFrame,
                             days_for_each_subscr_option: Tuple[int],
                             value_for_each_subscr_option: Tuple[float],
                             p_0: float, max_day: str) -> pd.DataFrame:
        """Return subscription data for active_uuid

        Parameters:
        ------------------------------------------------------------------------
        active_uuid: list, uuids for which data will be generated
        registration_table: pd.DataFrame, table with registration data, required
                                          columns=[UUID, timestamp]
        days_for_each_subscr_option: Tuple[int], number days when subscriptions
                                                 will be active
        value_for_each_subscr_option: Tuple[float], price for each subscription
        p_0: share, the probability that a user will not buy a subscription
        max_day: str, maximum date in datasample in format %Y-%M-%d
        ------------------------------------------------------------------------

        Returns
        ------------------------------------------------------------------------
        pd.DataFrame(columns=[UUID, timestamp, purchase_value]
                     ): table with purchases
        ------------------------------------------------------------------------
        """
        n_subscr_options = len(value_for_each_subscr_option)
        pay = dict(zip(np.arange(len(value_for_each_subscr_option)),
                       value_for_each_subscr_option))
        days = dict(zip(np.arange(len(days_for_each_subscr_option)),
                        days_for_each_subscr_option))
        registration_time = registration_table.set_index('UUID').loc[
            active_uuid]['timestamp']
        # choose random plan including without plan (-1)
        user_info = ((max_day - registration_time).dt.days + 1).reset_index()
        user_info = user_info.rename(
            columns={'timestamp': 'num_active_days'})
        user_info['registration_day'] = user_info['UUID'].map(registration_time)
        user_info['plans_order'] = user_info['num_active_days'].map(
            lambda x: np.random.choice(
                [-1] + list(days.keys()),
                size=x//min(days_for_each_subscr_option),
                p=[p_0] + [(1-p_0)/n_subscr_options]*n_subscr_options))
        user_info = user_info.explode('plans_order')
        user_info = user_info[~user_info['plans_order'].isna()]
        user_info = user_info.reset_index(drop=True)
        # calculate day when purchase will be made
        user_info['num_days'] = user_info['plans_order'].map(days)
        ind = user_info['plans_order'] == -1
        user_info.loc[ind, 'num_days'] = [np.random.choice(
            int(max(days_for_each_subscr_option)*1.5)) for x in range(ind.sum())
            ]
        user_info['plans_order_num_day_from_reg'] = \
            user_info.groupby('UUID')['num_days'].shift().fillna(0)
        user_info['plans_order_num_day_from_reg'] = \
            user_info.groupby('UUID')['plans_order_num_day_from_reg'].cumsum()
        user_info = user_info[user_info['plans_order'] != -1]
        user_info['timestamp'] = (
            pd.to_timedelta(user_info['plans_order_num_day_from_reg'], unit='D')
            + user_info['registration_day'])
        user_info = user_info[user_info['timestamp'] <= max_day]
        user_info = user_info.reset_index(drop=True)
        user_info['purchase_value'] = user_info['plans_order'].map(pay)
        return user_info[['UUID', 'timestamp', 'purchase_value']]

    @staticmethod
    def _get_purchases(mean_for_day_number: float,
                       std_for_day_number: float,
                       mean_for_value: float,
                       std_for_value: float,
                       active_uuid: list,
                       max_day: str,
                       registration_table: pd.DataFrame) -> pd.DataFrame:
        """Return purchases for active_uuid

        Parameters:
        ------------------------------------------------------------------------
        mean_for_day_number: float, mean value for number purchases a day
        std_for_day_number: float, std value for number purchases a day
        mean_for_value: float, mean value for purchase value
        std_for_value: float, mean value for purchase value
        active_uuid: list, uuids for which data will be generated
        max_day: str, maximum date in datasample in format %Y-%M-%d
        registration_table: pd.DataFrame, table with registration data, required
                                          columns=[UUID, timestamp]
        ------------------------------------------------------------------------

        Returns
        ------------------------------------------------------------------------
        pd.DataFrame(columns=[UUID, timestamp, purchase_value]
                     ): table with purchases
        ------------------------------------------------------------------------
        """
        mean = np.log(mean_for_day_number/(
            ((std_for_day_number**2)/(mean_for_day_number**2) + 1
            )**0.5))
        sigma = np.log((std_for_day_number**2)/(mean_for_day_number**2)+ 1)**0.5
        registration_time = registration_table.set_index('UUID').loc[
            active_uuid]['timestamp']

        user_info = ((max_day - registration_time).dt.days + 1).reset_index()
        user_info = user_info.rename(
            columns={'timestamp': 'num_active_days'})
        user_info['registration_day'] = user_info['UUID'].map(registration_time)
        user_info['day'] = 1
        user_info = user_info.loc[user_info.index.repeat(
            user_info['num_active_days']), :]
        user_info = user_info.reset_index(drop=True)
        user_info['day'] = user_info.groupby('UUID')['day'].cumsum()
        user_info['timestamp'] = (
            pd.to_timedelta(user_info['day'], unit='D')
            + user_info['registration_day'] )
        user_info['num_purchases'] = np.int_(np.random.lognormal(
            mean=mean, sigma=sigma, size=np.sum(user_info.shape[0])))
        user_info = user_info[user_info['num_purchases'] > 0]
        user_info = user_info.loc[user_info.index.repeat(
            user_info['num_purchases']), :]
        user_info = user_info.reset_index(drop=True)
        user_info['purchase_value'] = np.maximum(1, np.random.normal(
            mean_for_value, std_for_value, size=user_info.shape[0]))
        return user_info[['UUID', 'timestamp', 'purchase_value']]

    def get_purchases(self) -> pd.DataFrame:
        """Create table with purchases according to registration table

        Returns
        ------------------------------------------------------------------------
         pd.DataFrame(columns=[UUID, timestamp, event_name,
                               purchase_value]): table with purchases
        ------------------------------------------------------------------------
        """

        if self._ancor_table is None:
            raise Exception('Run get_ancor_table method first')
        self._define_active_users()
        purchases = pd.DataFrame(columns=['UUID',
                                          'timestamp',
                                          'purchase_value'])
        max_day = (pd.to_datetime(self.date_start)
                   + pd.to_timedelta(self.n_days, unit='D'))

        if self.is_subscr:
            high_payers_payments = self._get_payments_subscr(
                active_uuid=self._high_payers,
                registration_table=self._ancor_table,
                days_for_each_subscr_option=self.days_for_each_subscr_option,
                value_for_each_subscr_option=self.payment_for_each_subscr_option,
                p_0=0.1,
                max_day=max_day)
            mid_payers_payments = self._get_payments_subscr(
                active_uuid=self._mid_payers,
                registration_table=self._ancor_table,
                days_for_each_subscr_option=self.days_for_each_subscr_option,
                value_for_each_subscr_option=self.payment_for_each_subscr_option,
                p_0=0.3,
                max_day=max_day)
            low_payers_payments = self._get_payments_subscr(
                active_uuid=self._low_payers,
                registration_table=self._ancor_table,
                days_for_each_subscr_option=self.days_for_each_subscr_option,
                value_for_each_subscr_option=self.payment_for_each_subscr_option,
                p_0=0.6,
                max_day=max_day)
            purchases = pd.concat([purchases, high_payers_payments, mid_payers_payments, low_payers_payments], ignore_index=True)
        else:
            mid_payers_payments = self._get_purchases(
                mean_for_day_number=self.mean_for_day_number,
                std_for_day_number=self.std_for_day_number,
                mean_for_value=self.mean_for_value,
                std_for_value=self.std_for_value,
                active_uuid=self._mid_payers,
                max_day=max_day,
                registration_table=self._ancor_table)
            high_payers_payments = self._get_purchases(
                mean_for_day_number=self.mean_for_day_number*2,
                std_for_day_number=self.std_for_day_number*2**0.5,
                mean_for_value=self.mean_for_value*2,
                std_for_value=self.std_for_value*2**0.5,
                active_uuid=self._high_payers,
                max_day=max_day,
                registration_table=self._ancor_table)
            low_payers_payments = self._get_purchases(
                mean_for_day_number=self.mean_for_day_number*0.5,
                std_for_day_number=self.std_for_day_number*0.5**0.5,
                mean_for_value=self.mean_for_value*0.5,
                std_for_value=self.std_for_value*0.5**0.5,
                active_uuid=self._low_payers,
                max_day=max_day,
                registration_table=self._ancor_table)
            purchases = pd.concat([purchases, high_payers_payments, mid_payers_payments, low_payers_payments], ignore_index=True)
        purchases['event_name'] = 'purchase'

        purchases['registration_day'] = purchases['UUID'].map(
            self._ancor_table.set_index('UUID')['timestamp'])
        purchases = purchases[purchases['timestamp'] >=
                              purchases['registration_day']]
        purchases = purchases.drop(columns=['registration_day'])

        purchases = purchases.sample(frac=1).reset_index(drop=True)

        return purchases

    def generarte_datasets(self) -> (pd.DataFrame, pd.DataFrame):
        """Create ancor table and table with purchases according to
        registration table

        Returns
        ------------------------------------------------------------------------
        (
            pd.DataFrame(columns=[UUID, ancor_event_name, timestamp]),
            pd.DataFrame(columns=[UUID, event_name, timestamp,
                                  purchase_value])
        ): table with registration date and table with purchases
        ------------------------------------------------------------------------
        """
        ancor_table = self.get_ancor_table(force_update=True)
        purchases = self.get_purchases()
        return ancor_table, purchases
