## Model called here once finalized
# ..It will return a forecasted value value
# Output currency according to the user...................
import API_Config
import requests
from conversion import API_Call_Conversion
import city_index

if __name__ == '__main__':

    output_currency = 'AED'
    Model_currency = 'INR'
    Model_Output = 10000

        # 1) Fetch data for Mumbai and Boston
    currency_scale = API_Call_Conversion(amount=1,desired_currency= output_currency)
    mumbai_data = city_index.fetch_city_data("Mumbai", "India")
    boston_data = city_index.fetch_city_data("Boston", "United States")

    # 2) Compute scale factor: how much more expensive Boston is vs Mumbai
    scale = city_index.get_scale_factor(mumbai_data, boston_data,currency_scale)
    print(f"Scale factor (Boston vs Mumbai): {scale:.3f}")

    # 3) Example: scale a Mumbai monthly forecast (dummy data here)
    # Replace this with your real Prophet / ensemble forecast in INR
    forecast_mumbai_inr = np.array([20000, 21000, 20500, 22000])  # example values

    forecast_boston_inr = forecast_mumbai_inr * scale
    print("Mumbai forecast (INR):      ", forecast_mumbai_inr)
    print("Boston-scaled forecast (INR):", forecast_boston_inr)
    print("Boston-scaled forecast (USD):", forecast_boston_inr * currency_scale)

