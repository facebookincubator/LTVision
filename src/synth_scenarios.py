# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pandas as pd
from src.event_generator import BinomialEventGenerator, ParetoEventGenerator


class BaseScenario:

    def __init__(
        self, n_users: int, date_start: str, date_end: str, seed: int = None
    ) -> None:
        """
        Base class to create the characteristics associated to different products that can use
        the LTVision library.
        """
        self.n_users = n_users
        self.date_start = date_start
        self.date_end = date_end
        self.date_range = pd.date_range(date_start, date_end)
        self.rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )

        self.conv_event_gen = None
        self.cond_purchases_quantity_gen = None
        self.cond_purchase_event_gen = None
        self.cond_purchase_value_gen = None
        self.event_decay_scale = 1

    def get_default_demography_properties(self) -> pd.DataFrame:
        raise NotImplementedError

    def _get_conversion_prob(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Given the characteristic of a customer (e.g. country, device) decide whether he/she converted or not
        """
        return self.conv_event_gen.generate_events(customer_data)

    def _get_conditional_purchases(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Based on the characteristics of a customer, calculate the expected number of revenue events
        """
        return self.cond_purchases_quantity_gen.generate_events(customer_data)

    def _get_conditional_purchase_prob(
        self, data: pd.DataFrame, trials: int = None
    ) -> pd.DataFrame:
        """
        Based on the characteristics of the customer, product, and their interaction, calculate the number of
        revenue events
        """
        trials = trials if trials is not None else 1
        return self.cond_purchase_event_gen.generate_events(data, scale=trials)

    def _get_conditional_purchase_value(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Based on the characteristics of the customer, product, and their interaction, calculate the expected value
        of a revenue event
        """
        return self.cond_purchase_value_gen.generate_events(data)

    def get_revenue_events(
        self, customer_data: pd.DataFrame, events_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Given the customer and events data, uses the information contained in the 2 datasets to generate
        revenue events. Only converted users can generate revenue events, and they down't do every day.
        The steps to calculate whether a given user generates a revenue events are:
        1) Decide whether the customer is going to be considered as converted or not based on its demographic data
        2) If customer converted, then define a total number of purchases based solely on the demographic data
        3) Based on the total number of purchases expected for a customer, define the probability of a purchase event
           in a given day. The probability of a purchase event decays exponentially by how long the customer has been registered
        4) Based on the probability of purchase event on each day, decide the number of purchases by using a binomial distribution
           with p='probability of purchase' and n='total_revenue_events' (n is the number of trials for a binomial distribution)
        5) Then calculate the expected value of a revenue event based on the total number of events expected for the customer and
           demographic characteristics
        """
        data = customer_data.copy()
        data["converted"] = self._get_conversion_prob(
            data
        )  # probability of user ever converting
        data = data[data["converted"] == 1]

        data["total_revenue_events"] = self._get_conditional_purchases(data)
        data = pd.merge(data, events_data)
        # create probabilty based on exponential distribution. Given PDF(x) =
        # lambda*exp(-lambda*x), E[PDF(x)] = 1/lambda
        data["event_probability"] = data.apply(
            lambda x: self.event_decay_scale
            / x["total_revenue_events"]
            * np.exp(
                -(self.event_decay_scale / x["total_revenue_events"])
                * x["days_since_registration"]
            ),
            axis=1,
        )
        # probability of making a purchase conditional to returning
        data["events_number"] = self._get_conditional_purchase_prob(
            data, data["total_revenue_events"]
        )
        data = data[data["events_number"] > 0]

        # expected value of purchase conditional to making purchase
        data["event_value"] = self._get_conditional_purchase_value(data)
        data["value"] = data["converted"] * \
            data["events_number"] * data["event_value"]
        data = data.drop(
            [
                "converted",
                "events_number",
                "event_value",
                "total_revenue_events",
                "event_probability",
            ],
            axis=1,
        )
        return data


class BaseAppScenario(BaseScenario):
    """
    Base scenario for mobile app
    """

    def get_demography_data(self) -> pd.DataFrame:
        """
        Sets demography of the users to resemble that of mobile app.
        Adds to each user the following characteristics
            - country: country associated to first even
            - device: devices (ios or android) used in first event
            - download_method: whether customer was on wifi or mobile data when first logged in
            - registration_date: when was the first login of the customer
        """
        return pd.DataFrame(
            {
                "country": self.rng.choice(
                    ["US", "CA", "GB", "BR", "IN", "ES", "FR"],
                    size=self.n_users,
                    p=[0.2, 0.1, 0.05, 0.15, 0.3, 0.1, 0.1],
                    replace=True,
                ),
                "device": self.rng.choice(
                    ["ios", "android"], size=self.n_users, p=[0.4, 0.6], replace=True
                ),
                "download_method": self.rng.choice(
                    ["wifi", "mobile_data"],
                    size=self.n_users,
                    p=[0.7, 0.3],
                    replace=True,
                ),
                "registration_date": self.rng.choice(
                    self.date_range, size=self.n_users, replace=True
                ),
            }
        )


class IAPAppScenario(BaseAppScenario):
    """
    Base scenario for a mobile app based with monetization based in in-app purchases
    """

    def __init__(
        self, n_users: int, date_start: str, date_end: str, seed: int = None
    ) -> None:
        super().__init__(n_users, date_start, date_end, seed)

        self.conv_event_gen = BinomialEventGenerator(
            {
                "country": lambda x: {
                    "US": 0.5,
                    "CA": 0.3,
                    "GB": 0.2,
                    "BR": 0.1,
                    "IN": -0.2,
                    "ES": 0,
                    "FR": 0.1,
                }.get(x, 0),
                "device": lambda x: 0.5 if x == "ios" else 0,
                "download_method": lambda x: 0 if x == "wifi" else -1,
            },
            baseline=-3,  # around 4.7%
            seed=self.rng,
        )
        self.cond_purchases_quantity_gen = self.cond_purchase_value_gen = (
            ParetoEventGenerator(
                {
                    "country": lambda x: {
                        "US": 2,
                        "CA": 1,
                        "GB": 1.2,
                        "BR": -2,
                        "IN": -1.5,
                        "ES": 0,
                        "FR": 0.0,
                    }.get(x, 0),
                    "device": lambda x: 2 if x == "ios" else 0,
                    "download_method": lambda x: 2 if x == "wifi" else 0,
                },
                baseline=5,  # we expect as baseline 5 purchases
                seed=self.rng,
            )
        )
        self.cond_purchase_event_gen = BinomialEventGenerator(
            {"event_probability": lambda x: x}, seed=self.rng, logit_output=False
        )

        self.cond_purchase_value_gen = ParetoEventGenerator(
            {
                "country": lambda x: {
                    "US": 2,
                    "CA": 1,
                    "GB": 1.2,
                    "BR": -2,
                    "IN": -1.5,
                    "ES": 0,
                    "FR": 0.0,
                }.get(x, 0),
                "device": lambda x: 2 if x == "ios" else 0,
                "total_revenue_events": lambda x: np.log(x),
            },
            baseline=10,  # we put the expected value at 5. It *must* always be over 1
            seed=self.rng,
        )
        self.event_decay_scale = 0.1


class EcommScenario(BaseScenario):
    """
    Scenario for eCommerce clients with monetization based on direct purchases.
    """

    def __init__(
        self, n_users: int, date_start: str, date_end: str, seed: int = None
    ) -> None:
        super().__init__(n_users, date_start, date_end, seed)

        # Adjust conversion probabilities and purchase values for eCommerce
        self.conv_event_gen = BinomialEventGenerator(
            {
                "country": lambda x: {
                    "US": 0.6,
                    "CA": 0.5,
                    "GB": 0.4,
                    "BR": 0.3,
                    "IN": 0.2,
                    "ES": 0.3,
                    "FR": 0.4,
                }.get(x, 0),
                "device_type": lambda x: 0.3 if x == "mobile" else 0.2,
                "age_group": lambda x: {
                    "18-24": 0.2,
                    "25-34": 0.3,
                    "35-44": 0.2,
                    "45-54": 0.1,
                    "55+": 0.1,
                }.get(x, 0),
                "customer_segment": lambda x: {
                    "budget": 0.1,
                    "mid-range": 0.3,
                    "premium": 0.5,
                }.get(x, 0),
                "marketing_channel": lambda x: {
                    "email": 0.2,
                    "social media": 0.4,
                    "search": 0.3,
                }.get(x, 0),
            },
            baseline=-0.5,  # Adjusted baseline for higher conversion rate
            seed=self.rng,
        )
        self.cond_purchases_quantity_gen = ParetoEventGenerator(
            {
                "country": lambda x: {
                    "US": 1.5,
                    "CA": 1.2,
                    "GB": 1.0,
                    "BR": 0.8,
                    "IN": 0.7,
                    "ES": 0.9,
                    "FR": 1.0,
                }.get(x, 0),
                "device_type": lambda x: 1.5 if x == "mobile" else 1.0,
                "age_group": lambda x: {
                    "18-24": 1.0,
                    "25-34": 1.2,
                    "35-44": 1.0,
                    "45-54": 0.9,
                    "55+": 0.8,
                }.get(x, 0),
                "customer_segment": lambda x: {
                    "budget": 0.8,
                    "mid-range": 1.0,
                    "premium": 1.5,
                }.get(x, 0),
                "main_interest": lambda x: {
                    "electronics": 1.2,
                    "fashion": 1.0,
                    "home goods": 0.9,
                }.get(x, 0),
            },
            baseline=3,  # Baseline for expected number of purchases
            seed=self.rng,
        )
        self.cond_purchase_event_gen = BinomialEventGenerator(
            {"event_probability": lambda x: x}, seed=self.rng, logit_output=False
        )
        self.cond_purchase_value_gen = ParetoEventGenerator(
            {
                "country": lambda x: {
                    "US": 3,
                    "CA": 2.5,
                    "GB": 2.0,
                    "BR": 1.5,
                    "IN": 1.2,
                    "ES": 1.8,
                    "FR": 2.0,
                }.get(x, 0),
                "device_type": lambda x: 2.5 if x == "mobile" else 2.0,
                "total_revenue_events": lambda x: np.log(x),
                "customer_segment": lambda x: {
                    "budget": 1.0,
                    "mid-range": 1.5,
                    "premium": 2.0,
                }.get(x, 0),
                "main_interest": lambda x: {
                    "electronics": 1.5,
                    "fashion": 1.2,
                    "home goods": 1.0,
                }.get(x, 0),
                "age_group": lambda x: {
                    "18-24": 2.0,
                    "25-34": 2.5,
                    "35-44": 2.0,
                    "45-54": 1.8,
                    "55+": 1.5,
                }.get(x, 0),
            },
            baseline=20,  # Baseline for expected purchase value
            seed=self.rng,
        )
        self.event_decay_scale = 0.05  # Adjusted decay scale for eCommerce

    def get_demography_data(self) -> pd.DataFrame:
        """
        Sets demography of the users to resemble that of eCommerce clients.
        Adds to each user the following characteristics:
            - country: country associated with the first event
            - device: devices (ios or android) used in the first event
            - download_method: whether customer was on wifi or mobile data when first logged in
            - registration_date: when was the first login of the customer
            - customer_segment: segment of the customer (e.g., budget, mid-range, premium)
            - main_interest: primary category of interest (e.g., electronics, fashion, home goods)
            - marketing_channel: channel through which the customer was acquired (e.g., email, social media, search)
        """
        return pd.DataFrame(
            {
                "country": self.rng.choice(
                    ["US", "CA", "GB", "BR", "IN", "ES", "FR"],
                    size=self.n_users,
                    p=[0.25, 0.15, 0.1, 0.1, 0.2, 0.1, 0.1],
                    replace=True,
                ),
                "registration_date": self.rng.choice(
                    self.date_range, size=self.n_users, replace=True
                ),
                "device_type": self.rng.choice(
                    ["mobile", "desktop"], size=self.n_users, p=[0.6, 0.4], replace=True
                ),
                "customer_segment": self.rng.choice(
                    ["budget", "mid-range", "premium"],
                    size=self.n_users,
                    p=[0.4, 0.4, 0.2],
                    replace=True,
                ),
                "main_interest": self.rng.choice(
                    ["electronics", "fashion", "home goods"],
                    size=self.n_users,
                    p=[0.3, 0.4, 0.3],
                    replace=True,
                ),
                "age_group": self.rng.choice(
                    ["18-24", "25-34", "35-44", "45-54", "55+"],
                    size=self.n_users,
                    p=[0.2, 0.3, 0.2, 0.2, 0.1],
                    replace=True,
                ),
                "marketing_channel": self.rng.choice(
                    ["email", "social media", "search"],
                    size=self.n_users,
                    p=[0.3, 0.4, 0.3],
                    replace=True,
                ),
            }
        )
