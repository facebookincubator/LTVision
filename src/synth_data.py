# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import numpy as np
from src.synth_scenarios import BaseScenario, IAPAppScenario

class LTVSyntheticData():
    def __init__(self, 
                 n_users: int=100000,
                 registration_event_name: str='first_app_open',
                 event_name: str='purchase',
                 start_date: str='2020-01-01',
                 end_date: str='2022-12-31',
                 synthetic_scenario: BaseScenario=None,
                 random_seed:int=None
                 ) -> None:
        """
        Inputs
            - n_users: number of customers 
            - registration_event_name: the name of the event associated with the registration (i.e. identification) of a customer
            - event_name: name of the revenue event
            - start_date: start date for the dataset. Earliest time when a customer could be registered or generate a revenue event
            - end_date: end date for the dataset. Latest time when a customer could be registered or generate a revenue event
            - synthetic_scenario: the type of scenario to generate the data for. Takes BaseScenario instance as input or name of the scenario.
                Default scenario: IAPAppScenario
            - random_seed: random seed of the pseud-random number generator used in the synthetic_scenario
        """
        # Base Parameters
        self.n_users = n_users
        self.registration_event_name = registration_event_name
        self.event_name = event_name
        self.customer_data = None
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = pd.date_range(self.start_date, self.end_date)

        # Parameter to define the columns of the data
        self.synthetic_scenario = synthetic_scenario if synthetic_scenario is not None else IAPAppScenario(n_users, start_date, end_date, random_seed)
        self.event_name = event_name if event_name is not None else 'iap_purchase'

    def get_customers_data(self):
        """
        Get dataframe containing base costumer data, such as their user id, country, device, and download method for the app
        """
        customer_data = self._get_base_user_ids()
        customer_data = self._set_registration_event(customer_data, self.registration_event_name) # create registration_name column
        customer_data = self._set_demographic_properties(customer_data) # create and give demographic data
        self.customer_data = customer_data
        return customer_data

    def get_events_data(self):
        
        events_data = self.customer_data.copy()
        demographic_cols = [col for col in events_data if col != 'UUID'] # store the which are demographic features to drop later
        # add dates
        events_data = self._set_dates(events_data)
        events_data['days_since_registration'] = (events_data['event_date'] - events_data['registration_date']).dt.days
        events_data['event_name'] = self.event_name
        # generate revenue events
        events_data = self.synthetic_scenario.get_revenue_events(self.customer_data, events_data)
        # remove information already present in customer data
        events_data = events_data.drop(demographic_cols, axis=1)
        return events_data
    
    def _get_base_user_ids(self):
        """
        Generate a sequence of 1 to N numbers representing the unique user ID, where N is the number of users
        """
        return pd.DataFrame({'UUID': np.linspace(1, self.n_users, num=self.n_users).astype(str)})
    
    @staticmethod
    def _set_registration_event(customer_data: pd.DataFrame, event_name: str) -> pd.DataFrame:
        customer_data['registration_event_name'] = event_name
        return customer_data

    def _set_demographic_properties(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        
        demography_data = self.synthetic_scenario.get_demography_data()
        return pd.concat([customer_data, demography_data], axis=1)
    
    def _set_dates(self, events_data: pd.DataFrame) -> None:
        dates_data = pd.DataFrame({'event_date': self.date_range})

        events_data['cross_join_col'] = 1
        dates_data['cross_join_col'] = 1
        output_data = pd.merge(events_data, dates_data, on=['cross_join_col']).drop('cross_join_col', axis=1)
        return output_data[output_data['event_date'] >= output_data['registration_date']]

    