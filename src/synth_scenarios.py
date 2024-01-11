from typing import Dict
import pandas as pd
import numpy as np
from event_generator import BinomialEventGenerator, ParetoEventGenerator, LognormalEventGenerator

class BaseScenario():

    def __init__(self, n_users: int, date_start: str, date_end: str, max_revenue_events: int=None, random_seed:int=None) -> None:
        """
        TODO: 
        """
        self.n_users = n_users
        self.date_start = date_start
        self.date_end = date_end
        self.date_range = pd.date_range(date_start, date_end)
        self.max_revenue_events = 500 if max_revenue_events is None else max_revenue_events
        self.random_seed = random_seed

        self.conv_event_gen = None
        self.cond_return_event_gen = None
        self.cond_ltv_value_gen = None
        self.cond_purchase_event_gen = None
        self.cond_purchase_value_gen = None

    def get_demography_data(self) -> pd.DataFrame:
        raise NotImplementedError
     
    def _get_conversion_prob(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        return self.conv_event_gen.generate_events(customer_data)

    def _get_conditional_return_prob(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.cond_return_event_gen.generate_events(data)
    
    def _get_conditional_ltv_value(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.cond_ltv_value_gen.generate_events(data)
    
    def _get_conditional_purchase_value(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.cond_purchase_value_gen.generate_events(data)
    
    def _get_conditional_purchase_event(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.cond_purchase_event_gen.generate_events(data)
    

    def _get_purchase_probability(self, data: pd.DataFrame) -> pd.DataFrame:
        total_users_return = data.groupby(['UUID'])['returned'].sum().reset_index()
        total_users_return = total_users_return.rename(columns={'returned': 'total_returns'})
        data = pd.merge(data, total_users_return, on='UUID')
        return (data['ltv'] / data['event_value'] / self.max_revenue_events).clip(upper=1)
    
    def set_revenue_events(self, customer_data: pd.DataFrame, events_data: pd.DataFrame) -> pd.DataFrame:

        customer_data['converted'] = self._get_conversion_prob(customer_data) # probability of user ever converting 
        customer_data = customer_data[customer_data['converted'] == 1] # filter so it speeds up calculations
        customer_data['ltv'] = self._get_conditional_ltv_value(customer_data) # calculate the expect ltv now(instead as a result) to ensure the distribution matches

        data = pd.merge(customer_data, events_data)
        data['returned'] = self._get_conditional_return_prob(data) # probability of returning in a specific day
        data = data[data['returned'] == 1] # filter so it speeds up calculations
        data['event_value'] = self._get_conditional_purchase_value(data) # expected value of purchase conditional to making purchase
        data['revenue_event_probability'] = self._get_purchase_probability(data)
        data['purchased'] = self._get_conditional_purchase_event(data)
        data = data[data['purchased'] == 1] # filter so it speeds up calculations

        data['value'] = data['converted'] * data['returned'] * data['purchased'] * data['event_value']
        data = data.drop(['converted', 'ltv', 'returned', 'event_value', 'revenue_event_probability', 'revenue_event_probability'], axis=1)
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
            baseline=-3, # around 4.75%
            random_seed=self.random_seed
        )
        self.cond_return_event_gen = BinomialEventGenerator(
            {
                'country': lambda x: {'US': 0.1, 'CA': -0.1, 'GB': 0.0, 'BR': 0.1, 'IN': 0.1, 'ES': 0, 'FR': 0.1}.get(x, 0),
                'device':  lambda x: 0.5 if x == 'ios' else 0,
                'download_method': lambda x: 0 if x == 'wifi' else -1,
                'days_since_registration': lambda x: 6*np.exp(-0.05*x), # decay exponentianly and stabilizes at a point where p>0
                'converted': lambda x: 1.5*x
            },
            baseline=-5.5, # around 60% if not converted, else 95%
            random_seed=self.random_seed
        )

        self.cond_ltv_value_gen = ParetoEventGenerator(
            {
                'country': lambda x: {'US': 10, 'CA': 5, 'GB': 6, 'BR': -10, 'IN': -5, 'ES': 0, 'FR': 0}.get(x, 0),
                'device':  lambda x: 10 if x == 'ios' else 0,
                'download_method': lambda x: 0 if x == 'wifi' else -5,
            },
            baseline=24, # we put the expected value at 5. It *must* always be over 1
            random_seed=self.random_seed
        )
        
        self.cond_purchase_value_gen = ParetoEventGenerator(
            {
                'country': lambda x: {'US': 2, 'CA': 1, 'GB': 1.2, 'BR': -2, 'IN': -1.5, 'ES': 0, 'FR': 0.0}.get(x, 0),
                'device':  lambda x: 2 if x == 'ios' else 0,
                'days_since_registration': lambda x: 0 if x < 500 else 5
            },
            baseline=12, # we put the expected value at 5. It *must* always be over 1
            random_seed=self.random_seed
        )

        self.cond_purchase_event_gen = BinomialEventGenerator(
            {
                'revenue_event_probability': lambda x: x
            },
            baseline=0,
            scale=self.max_revenue_events,
            random_seed=self.random_seed,
            logit_output=False
        )