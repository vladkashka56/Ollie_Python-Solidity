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




# def Test(latitude, longitude, time_index):
#     system = ol.create_system(solar_dict, rates_dict=rates_dict)
#     positions = ol.get_ts_positions(system, time_index)
#     obs_latitude =  latitude * np.pi / 180
#     obs_longitude = longitude * np.pi / 180

#     sky_positions = ol.get_sky_positions(positions, coordinates='horizontal', obs_latitude=obs_latitude, obs_longitude=obs_longitude)
#     print(sky_positions)


# Test(40.75127, -73.98482, 205421)


system = ol.create_system(solar_dict, rates_dict=rates_dict, observer='Earth')
reload(ol)
ol.show_sky_view(system, 40.71273, -74.00602, '2008-06-01 4:13')
