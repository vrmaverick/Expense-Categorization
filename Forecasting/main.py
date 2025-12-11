## Model called here once finalized
# ..It will return a forecasted value value
# Output currency according to the user...................
import API_Config
import requests
from conversion import API_Call_Conversion, city
import city_index
import numpy as np
import Model as md
import pandas as pd

if __name__ == '__main__':

    output_currency = 'AED'
    Model_currency = 'INR'
    scale = city(city_name="Boston",country_name="United States",output_currency=output_currency)




    user_expenses,user_dates= md.Generate_sample()
    print('Generated')
    # avg = md.Predict()
    # forecast_mumbai_inr = np.array(avg)  # example values
    # forecast_city_inr = forecast_mumbai_inr * scale

    # print("Mumbai forecast (INR):      ", forecast_mumbai_inr)
    # print("Boston-scaled forecast (INR):", forecast_city_inr)
    # print("Boston-scaled forecast (USD):", forecast_city_inr * scale)
    #         # Predict_Model(model=model)