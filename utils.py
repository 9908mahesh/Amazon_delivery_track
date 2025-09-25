import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def prepare_input(distance, pickup_delay, order_hour, order_day, order_weekday,
                  is_peak, is_weekend, weather, traffic, vehicle, area, category):
    return pd.DataFrame([[
        distance, pickup_delay, order_hour, order_day, order_weekday,
        is_peak, is_weekend,
        np.digitize(distance, [0,2,5,10,20,50]),   # Distance_Bucket
        traffic*distance,                          # Traffic_Distance
        weather*pickup_delay,                      # Weather_Delay
        weather, traffic, vehicle, area, category
    ]], columns=[
        "Distance_km","Pickup_Delay","Order_Hour","Order_Day","Order_Weekday",
        "Is_Peak_Hour","Is_Weekend","Distance_Bucket","Traffic_Distance","Weather_Delay",
        "Weather","Traffic","Vehicle","Area","Category"
    ])
