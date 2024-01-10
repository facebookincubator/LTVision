from typing import Dict
import pandas as pd
import numpy as np
from event_generator import BinomialEventGenerator, ParetoEventGenerator, LognormalEventGenerator

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
        self.cond_return_event_gen = None
        self.cond_purchase_event_gen = None
        self.cond_purchase_value_gen = None

    def get_default_demography_properties(self) -> pd.DataFrame:
        raise NotImplementedError
     
    def get_conversion_prob(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.conv_event_gen.generate_events(data)

    def get_conditional_return_prob(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.cond_return_event_gen.generate_events(data)

    def get_conditional_purchase_prob(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.cond_purchase_event_gen.generate_events(data)

    def get_conditional_purchase_value(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.cond_purchase_value_gen.generate_events(data)
    
    def get_events_data(self, data: pd.DataFrame) -> pd.DataFrame:

        data['converted'] = self.get_conversion_prob(data) # probability of user ever converting 
        # data = data[data['converted'] == 1] # users who never made a purchase event. This speeds up the following methods
        data['returned'] = self.get_conditional_return_prob(data) # probability of returning in a specific day
        data['purchased'] = self.get_conditional_purchase_prob(data) # probability of making a purchase conditional to returning
        data['purchase_value'] = self.get_conditional_purchase_value(data) # expected value of purchase conditional to making purchase
        data['value'] = data['converted'] * data['converted'] * data['converted'] * data['converted']
        data = data.drop(['converted', 'returned', 'purchased', 'purchase_value'], axis=1)
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
            baseline=-4, # around 1.8%
            random_seed=self.random_seed
        )
        self.cond_return_event_gen = BinomialEventGenerator(
            {
                'country': lambda x: {'US': 0.1, 'CA': -0.1, 'GB': 0.0, 'BR': 0.1, 'IN': 0.1, 'ES': 0, 'FR': 0.1}.get(x, 0),
                'device':  lambda x: 0.5 if x == 'ios' else 0,
                'download_method': lambda x: 0 if x == 'wifi' else -1,
                'days_since_registration': lambda x: -0.2*x if x < 30 else (-0.1*x - 3),
                'converted': lambda x: 2.5*x
            },
            baseline=0.5, # around 60% if not converted, else 95%
            random_seed=self.random_seed
        )
        
        self.cond_purchase_event_gen = BinomialEventGenerator(
            {
                'country': lambda x: {'US': 0.2, 'CA': 0.1, 'GB': 0.0, 'BR': -0.1, 'IN': -0.1, 'ES': 0, 'FR': 0.0}.get(x, 0),
                'device':  lambda x: 0.3 if x == 'ios' else 0,
                'days_since_registration': lambda x: 1 if x < 7 else 0,
            },
            baseline=-1, # ~ 27% purchase chance if return (given that the person will convert)
            random_seed=self.random_seed
        )
        self.cond_purchase_value_gen = ParetoEventGenerator(
            {
                'country': lambda x: {'US': 2, 'CA': 1, 'GB': 1.2, 'BR': -2, 'IN': -1.5, 'ES': 0, 'FR': 0.0}.get(x, 0),
                'device':  lambda x: 2 if x == 'ios' else 0,
                'days_since_registration': lambda x: 0 if x < 500 else 5
            },
            baseline=18, # we put the expected value at 5. It *must* always be over 1
            random_seed=self.random_seed
        )