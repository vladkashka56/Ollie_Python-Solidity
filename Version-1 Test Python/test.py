from turtle import position
import orbitlib as ol
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from importlib import reload  # TESTING

Solar_System_Keplerian_Elements = {
  "Mercury":dict(a=0.38709893, e=0.20563069, T=0.240846, I=7.00487, Omega=48.33167, omega_bar=77.45645, L=252.25084),
  "Venus":dict(a=0.72333199, e=0.00677323, T=0.615, I=3.39471, Omega=76.68069, omega_bar=131.53298, L=181.97973),
  "Earth":dict(a=1.00000011, e=0.01671022, T=1, I=0.00005, Omega=0, omega_bar=102.94719, L=100.46435),
  "Mars":dict(a=1.52366231, e=0.09341233, T=1.881, I=1.85061, Omega=49.57854, omega_bar=336.04084, L=355.45332),
  "Jupiter":dict(a=5.20336301, e=0.04839266, T=11.86, I=1.30530, Omega=100.55615, omega_bar=14.75385, L=34.40438),
  "Saturn":dict(a=9.53707032, e=0.05415060, T=29.46, I=2.48446, Omega=113.71504, omega_bar=92.43194, L=49.94432),
  "Uranus":dict(a=19.19126393, e=0.04716771, T=84.01, I=0.76986, Omega=74.22988, omega_bar=170.96424, L=313.23218),
  "Neptune":dict(a=30.06896348, e=0.00858587, T=164.8, I=1.76917, Omega=131.72169, omega_bar=44.97135, L=304.88003),
  "Pluto":dict(a=39.48168677, e=0.24880766, T=248.1, I=17.14175, Omega=110.30347, omega_bar=224.06676, L=238.92881)
}

solar_df = pd.DataFrame.from_dict(Solar_System_Keplerian_Elements, orient='index')

# converting units to SI and transforming L into tauHello
for col in ['I','Omega','omega_bar','L']:
  solar_df[col] *= 2 * np.pi / 360  # deg -> rad
solar_df['omega'] = solar_df['omega_bar'] - solar_df['Omega']
solar_df['T'] =  solar_df['T'].apply(lambda x: timedelta(days=365.25636 * x).total_seconds())  # sideral years -> sideral days
solar_df['M'] = solar_df['L'] - solar_df['omega_bar']  # mean anomaly at J2000

J2000 = datetime(2000,1,1,12,0,0).timestamp()  # seconds from "1970-1-1"
  
# dictionary with keys are Planet names and value as dictionary of orbital parameters 
solar_dict = solar_df.drop(['L', 'omega_bar'], axis=1).to_dict(orient='index')


system = ol.create_system(solar_dict)
# system = ol.create_system(solar_dict, observer='Earth')

# positions = ol.get_positions(system, time_index=pd.DatetimeIndex(['2022-03-25 1:13']))
J2000 = datetime(2000, 1, 1, 12)
Jcur = datetime(2022, 3, 25, 1)

timeIDX = Jcur.timestamp() - J2000.timestamp()
print(timeIDX)

positions = ol.get_ts_3d_positions(system, timeIDX)

print(positions)

# print(positions)

# positions = ol.get_ts_2d_positions(system, 2345)

# print(positions)




