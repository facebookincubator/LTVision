# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import numpy as np
from src.event_generator import BinomialEventGenerator, ParetoEventGenerator, LognormalEventGenerator

class BaseScenario():

    def __init__(self, n_users: int, date_start: str, date_end: str, random_seed:int=None) -> None:
        """
        TODO: 
        """
        self.n_users = n_users
        self.date_start = date_start
        self.date_end = date_end
        self.date_range = pd.date_range(date_start, date_end)
        self.random_seed = random_seed

        self.conv_event_gen = None
        self.cond_purchases_quantity_gen = None
        self.cond_purchase_event_gen = None
        self.cond_purchase_value_gen = None

    def get_default_demography_properties(self) -> pd.DataFrame:
        raise NotImplementedError
     
    def get_conversion_prob(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        return self.conv_event_gen.generate_events(customer_data)

    def get_conditional_purchases(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        return self.cond_purchases_quantity_gen.generate_events(customer_data)
    
    def get_conditional_purchase_prob(self, data: pd.DataFrame, trials: int=None) -> pd.DataFrame:
        trials = trials if trials is not None else 1
        return self.cond_purchase_event_gen.generate_events(data, scale=trials)
    
    def get_conditional_purchase_value(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.cond_purchase_value_gen.generate_events(data)
    
    def set_revenue_events(self, customer_data: pd.DataFrame, events_data: pd.DataFrame) -> pd.DataFrame:

        data = customer_data.copy()
        data['converted'] = self.get_conversion_prob(data) # probability of user ever converting 
        data = data[data['converted'] == 1]

        data['total_purchases'] = self.get_conditional_purchases(data)
        data = pd.merge(data, events_data)
        # create probabilty based on exponential distribution. Given PDF(x) = lambda*exp(-lambda*x), E[PDF(x)] = 1/lambda
        data['purchase_probability'] = data.apply(
            lambda x: .1/x['total_purchases']*np.exp(-(.1/x['total_purchases'])*x['days_since_registration']), 
            axis=1)
        data['purchased'] = self.get_conditional_purchase_prob(data, data['total_purchases']) # probability of making a purchase conditional to returning
        data = data[data['purchased'] > 0]

        data['purchase_value'] = self.get_conditional_purchase_value(data) # expected value of purchase conditional to making purchase
        data['value'] = data['converted'] * data['purchased'] * data['purchase_value']
        data = data.drop(['converted', 'purchased', 'purchase_value', 'total_purchases', 'purchase_probability'], axis=1)
        return data

class BaseAppScenario(BaseScenario):

    def get_demography_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            'country': np.random.choice(['US', 'CA', 'GB', 'BR', 'IN', 'ES', 'FR'], size=self.n_users, p=[0.2, 0.1, 0.05, 0.15, 0.3, 0.1, 0.1], replace=True),
            'device':  np.random.choice(['ios', 'android'], size=self.n_users, p=[0.4, 0.6], replace=True),
            'download_method': np.random.choice(['wifi', 'mobile_data'], size=self.n_users, p=[0.7, 0.3], replace=True),
            'registration_date': np.random.choice(self.date_range, size=self.n_users, replace=True),
        })
    
class IAPAppScenario(BaseAppScenario):

    def __init__(self, n_users: int, date_start: str, date_end: str, random_seed:int=None) -> None:
        super().__init__(n_users, date_start, date_end)

        self.conv_event_gen = BinomialEventGenerator(
            {
                'country': lambda x: {'US': 0.5, 'CA': 0.3, 'GB': 0.2, 'BR': 0.1, 'IN': -0.2, 'ES': 0, 'FR': 0.1}.get(x, 0),
                'device':  lambda x: 0.5 if x == 'ios' else 0,
                'download_method': lambda x: 0 if x == 'wifi' else -1
            },
            baseline=-3, # around 1.8%
            random_seed=self.random_seed
        )
        self.cond_purchases_quantity_gen = self.cond_purchase_value_gen = ParetoEventGenerator(
            {
                'country': lambda x: {'US': 2, 'CA': 1, 'GB': 1.2, 'BR': -2, 'IN': -1.5, 'ES': 0, 'FR': 0.0}.get(x, 0),
                'device':  lambda x: 2 if x == 'ios' else 0,
                'download_method': lambda x: 2 if x == 'wifi' else 0
            },
            baseline=5, # we expect as baseline 5 purchases
            random_seed=self.random_seed
        )
        self.cond_purchase_event_gen = BinomialEventGenerator(
            {
                'purchase_probability': lambda x: x
            },
            random_seed=self.random_seed,
            logit_output=False
        )

        self.cond_purchase_value_gen = ParetoEventGenerator(
            {
                'country': lambda x: {'US': 2, 'CA': 1, 'GB': 1.2, 'BR': -2, 'IN': -1.5, 'ES': 0, 'FR': 0.0}.get(x, 0),
                'device':  lambda x: 2 if x == 'ios' else 0,
                'total_purchases': lambda x: np.log(x)
            },
            baseline=10, # we put the expected value at 5. It *must* always be over 1
            random_seed=self.random_seed
        )