from turtle import position
import orbitlib as ol
import pickle
import tqdm
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import json
import pickle

from importlib import reload  # TESTING

from orbitlib_data import solar_dict, solar_df, rates_dict, rates_df, moons_dict, moons_df

reload(ol)


# system = ol.create_system(solar_dict)

# system = ol.create_system(solar_dict, observer='Earth')
system = ol.create_system(solar_dict, rates_dict=rates_dict, observer='Earth')
positions = ol.get_positions(system, time_index=pd.DatetimeIndex(['2010-01-01 12:00']))
print(positions)
# ol.plot_positions(positions)

# reload(ol)

# Warsaw
obs_latitude =  40.75127 * np.pi / 180
obs_longitude = -73.98482 * np.pi / 180



# print(positions)

sky_positions = ol.get_sky_positions(positions, coordinates='horizontal', obs_latitude=obs_latitude, obs_longitude=obs_longitude)
print(sky_positions)

# print("2010")
# positions = ol.get_positions(system, time_index=pd.DatetimeIndex(['2010-01-01 12:00']))
# sky_positions = ol.get_sky_positions(positions, coordinates='horizontal', obs_latitude=obs_latitude, obs_longitude=obs_longitude)
# print(sky_positions)


# sky_positions = ol.get_sky_positions(positions, coordinates='horizontal', obs_latitude=obs_latitude, obs_longitude=obs_longitude)


# print(sky_positions)

# ol.plot_sky_positions_Polar(sky_positions)