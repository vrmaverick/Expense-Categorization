from . import API_Config
# import API_Config
import requests
from . import city_index
# import city_index
# def Convert(response,amount,User_output = 'USD'):
#     INR_rate = response["rates"]["INR"]
#     if User_output != 'USD':
#         desired_rate = response["rates"][User_output]
#         amount = amount/INR_rate
#         amount = amount * desired_rate
#         return amount
#     else:
#         amount = amount/INR_rate
#         amount = amount * desired_rate

def city(city_name, country_name, output_currency):
    """
    Returns a scale factor to adjust forecast from base INR to the target city/currency.
    Falls back to a default value if API-based scaling fails.
    """
    try:
        currency_scale = API_Call_Conversion(amount=1, desired_currency=output_currency)
        mumbai_data = city_index.fetch_city_data("Mumbai", "India")
        target_data = city_index.fetch_city_data(city_name, country_name)

        scale = city_index.get_scale_factor(mumbai_data, target_data, currency_scale)
        print(f"Scale factor ({city_name} vs Mumbai): {scale:.3f}")
        return float(scale)

        # # For now, still using placeholder until above is enabled
        # return 6.756

    except Exception as e:
        # Log or print the error for debugging
        print(f"[city] Failed to compute scale for {city_name}, {country_name} ({output_currency}): {e}")
        # Fallback scale factor
        return 6.756



def Convert(response, amount, User_output='USD'):
    """
    Convert `amount` from INR to `User_output` currency.
    response: dict from currencyapi.net (base = 'USD')
    """
    rates = response["rates"]
    inr_rate = rates["INR"]          # 1 USD = inr_rate INR

    # First convert INR -> USD
    amount_usd = amount / inr_rate

    if User_output == 'USD':
        return amount_usd

    # Then USD -> desired currency
    desired_rate = rates[User_output]   # 1 USD = desired_rate of that currency
    amount_converted = amount_usd * desired_rate
    return amount_converted
    # print("AED Rate:", aed_rate)

def API_Call_Conversion(amount,desired_currency): 
    key = API_Config.Get_Currency_API_key()
    # key = "8401a4bcdecb7fa57a417007042e42b661e0"
    base = "USD"
    output = "json"

    url = f"https://currencyapi.net/api/v1/rates?key={key}&base={base}&output={output}"
    headers = {
        'Accept': 'application/json'
    }

    response = requests.get(url, headers=headers)
    # Convert the response to JSON
    data = response.json()
    print(data)
    return Convert(data,amount,User_output = desired_currency)