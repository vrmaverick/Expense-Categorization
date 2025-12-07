import requests
import numpy as np
import conversion

RAPIDAPI_KEY = "aea1c9d920mshc5251b8f872c9e1p1a5a26jsn3106a948362e"
RAPIDAPI_HOST = "cost-of-living-and-prices.p.rapidapi.com"
BASE_URL = "https://cost-of-living-and-prices.p.rapidapi.com/prices"

# INR_PER_USD = 83.0  # update to current FX rate if you want

def fetch_city_data(city_name: str, country_name: str):
    url = BASE_URL
    querystring = {
        "city_name": city_name,
        "country_name": country_name,
    }
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }
    resp = requests.get(url, headers=headers, params=querystring)
    resp.raise_for_status()
    data = resp.json()
    # RapidAPI wrapper may nest data inside 'prices' or similar
    # Your pasted sample is already the inner object, so adapt if needed:
    if isinstance(data, dict) and "prices" in data:
        return data
    elif isinstance(data, dict) and "data" in data:
        # some RapidAPI endpoints wrap actual object in data["prices"]
        return data["data"]
    else:
        return data

def get_city_basket_cost_usd(city_data: dict) -> float:
    """
    Build a simple monthly basket cost in USD from the API response.
    Adjust which items you include as per your use case.
    """
    prices = city_data["prices"]

    def avg_price(item_name: str):
        for p in prices:
            if p["item_name"] == item_name:
                # use avg in USD
                return float(p["usd"]["avg"])
        return 0.0

    # Rent (1 BHK outside centre) + utilities + internet + groceries + transport pass
    rent = avg_price("One bedroom apartment outside of city centre")
    utilities = avg_price("Basic utilities for 85 square meter Apartment including Electricity, Heating or Cooling, Water and Garbage")
    internet = avg_price("Internet, 60 Mbps or More, Unlimited Data, Cable/ADSL")
    groceries = (
        avg_price("Milk, Regular,1 liter") * 30 +
        avg_price("Loaf of Fresh White Bread, 0.5 kg") * 8 +
        avg_price("Eggs, 12 pack") * 4
    )
    transport = avg_price("Monthly Pass, Regular Price")

    basket_usd = rent + utilities + internet + groceries + transport
    return basket_usd

def get_city_basket_cost_inr(city_data: dict, currency_conversion_factor) -> float:
    basket_usd = get_city_basket_cost_usd(city_data)
    return basket_usd * currency_conversion_factor

def get_scale_factor(mumbai_data: dict, target_data: dict, currency_scale) -> float:

    mumbai_basket_inr = get_city_basket_cost_inr(mumbai_data,currency_scale)
    target_basket_inr = get_city_basket_cost_inr(target_data, currency_scale)
    if mumbai_basket_inr == 0:
        raise ValueError("Mumbai basket cost is zero, cannot scale.")
    return target_basket_inr / mumbai_basket_inr

if __name__ == "__main__":
    # 1) Fetch data for Mumbai and Boston
    currency_scale = conversion.API_Call_Conversion(amount=1,desired_currency= 'USD')
    mumbai_data = fetch_city_data("Mumbai", "India")
    boston_data = fetch_city_data("Boston", "United States")

    # 2) Compute scale factor: how much more expensive Boston is vs Mumbai
    scale = get_scale_factor(mumbai_data, boston_data,currency_scale)
    print(f"Scale factor (Boston vs Mumbai): {scale:.3f}")

    # 3) Example: scale a Mumbai monthly forecast (dummy data here)
    # Replace this with your real Prophet / ensemble forecast in INR
    forecast_mumbai_inr = np.array([20000, 21000, 20500, 22000])  # example values

    forecast_boston_inr = forecast_mumbai_inr * scale
    print("Mumbai forecast (INR):      ", forecast_mumbai_inr)
    print("Boston-scaled forecast (INR):", forecast_boston_inr)
    print("Boston-scaled forecast (USD):", forecast_boston_inr * currency_scale)
