# Observational data of Solar planets and selected moons
# Transformations are done explicitly, although it's not a good practice in import file, but I left them for future reference. 


import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Orbital parameters for Solar System planets
#
# Source: Source: Standish and Williams (1992) through https://farside.ph.utexas.edu/teaching/celestial/Celestialhtml/node34.html
#
# Units a:AU, e:[0,1], T:yr, I:deg, Omega:deg, omega:deg, L:deg
# Epoch J2000 (2000-1-1.5)

Solar_System_Keplerian_Elements = {
  "Mercury":dict(a=0.38709893, e=0.20563069, T=0.240846, I=7.00487, Omega=48.33167, omega_bar=77.45645, L=252.25084),
  "Venus":dict(a=0.72333199, e=0.00677323, T=0.615, I=3.39471, Omega=76.68069, omega_bar=131.53298, L=181.97973),
  "EM":dict(a=1.00000011, e=0.01671022, T=1, I=0.00005, Omega=0, omega_bar=102.94719, L=100.46435),
  "Mars":dict(a=1.52366231, e=0.09341233, T=1.881, I=1.85061, Omega=49.57854, omega_bar=336.04084, L=355.45332),
  "Jupiter":dict(a=5.20336301, e=0.04839266, T=11.86, I=1.30530, Omega=100.55615, omega_bar=14.75385, L=34.40438),
  "Saturn":dict(a=9.53707032, e=0.05415060, T=29.46, I=2.48446, Omega=113.71504, omega_bar=92.43194, L=49.94432),
  "Uranus":dict(a=19.19126393, e=0.04716771, T=84.01, I=0.76986, Omega=74.22988, omega_bar=170.96424, L=313.23218),
  "Neptune":dict(a=30.06896348, e=0.00858587, T=164.8, I=1.76917, Omega=131.72169, omega_bar=44.97135, L=304.88003),
  "Pluto":dict(a=39.48168677, e=0.24880766, T=248.1, I=17.14175, Omega=110.30347, omega_bar=224.06676, L=238.92881)
}

# Calculating orbital parameters for the Earth and the Moon. Can be hardcoded, but here the equations are left explicitely for future reference

# The Moon's orbit around the Earth
# Values from "Lunar Tables and Programs From 4000 B.C. TO A.D. 8000" by Michelle Chapront-Touze and Jean Chapront.
a_rel = 383_397.6  # [km]
a_rel /= 149_597_871  # [km -> AU]
e_rel = 0.055544
T_rel = 27.32166  # [day]
T_rel /= 365.25636  # [day -> yr]
gamma_rel = 0.0449858  # sin(I/2)
omega_bar_rel = 83.353  # [deg]
Omega_rel = 125.0446  # [deg]
L_rel = 218.31665  # [deg] Mean longitude
r = 0.0123  # Moon to Earth mass ratio (this one from Wikipedia)

I_rel = 2 * np.arcsin(gamma_rel)  # [rad]
I_rel *= 180/np.pi  # [rad -> deg]

# The Earth's and Moon's orbits around the barycenter ('EM') are similar to the Moon's orbit around the Earth.
# Only a few parameters need scaling/shifting what we perform below
a_E = a_rel * r / (1.0 + r)
a_M = a_rel / (1.0 + r)
omega_bar_E = (omega_bar_rel + 180) % 360
L_E = (L_rel + 180) % 360

Solar_System_Keplerian_Elements['Earth'] = dict(a=a_E, e=e_rel, T=T_rel, I=I_rel, Omega=Omega_rel, omega_bar=omega_bar_E, L=L_E, parent='EM')
Solar_System_Keplerian_Elements['Moon'] = dict(a=a_M, e=e_rel, T=T_rel, I=I_rel, Omega=Omega_rel, omega_bar=omega_bar_rel, L=L_rel, parent='EM')

solar_df = pd.DataFrame.from_dict(Solar_System_Keplerian_Elements, orient='index').replace({np.nan:None})

# converting units to SI and transforming L into tauHello
for col in ['I','Omega','omega_bar','L']:
  solar_df[col] *= 2 * np.pi / 360  # deg -> rad
solar_df['omega'] = solar_df['omega_bar'] - solar_df['Omega']
solar_df['T'] =  solar_df['T'].apply(lambda x: timedelta(days=365.25636 * x).total_seconds())  # sideral years -> sideral days
solar_df['M'] = solar_df['L'] - solar_df['omega_bar']  # mean anomaly at J2000

J2000 = datetime(2000,1,1,12,0,0).timestamp()  # seconds from "1970-1-1"
  

# dictionary with keys are Planet names and value as dictionary of orbital parameters 
solar_dict = solar_df.drop(['L', 'omega_bar'], axis=1).to_dict(orient='index')



# Change rates of orbital paramters for Solar System planets
#
# Source https://ssd.jpl.nasa.gov/planets/approx_pos.html
#
# units au/Cy, rad/Cy, deg/Cy, deg/Cy, deg/Cy, deg/Cy

Solar_System_Rates = {
"Mercury": dict(a=0.00000037, e=0.00001906, I=-0.00594749, L=149472.67411175, omega_bar=0.16047689, Omega=-0.12534081),
"Venus":  dict(a=0.00000390, e=-0.00004107, I=-0.00078890, L=58517.81538729, omega_bar=0.00268329, Omega=-0.27769418),
"EM": dict(a=0.00000562, e=-0.00004392, I=-0.01294668, L=35999.37244981, omega_bar=0.32327364, Omega=0.0),
"Mars": dict(a=0.00001847, e=0.00007882, I=-0.00813131, L=19140.30268499, omega_bar=0.44441088, Omega=-0.29257343),
"Jupiter": dict(a=-0.00011607, e=-0.00013253, I=-0.00183714, L=3034.74612775, omega_bar=0.21252668, Omega=0.20469106),
"Saturn": dict(a=-0.00125060, e=-0.00050991, I=0.00193609, L=1222.49362201, omega_bar=-0.41897216, Omega=-0.28867794),
"Uranus": dict(a=-0.00196176, e=-0.00004397, I=-0.00242939, L=428.48202785, omega_bar=0.40805281, Omega=0.04240589),
"Neptune": dict(a=0.00026291, e=0.00005105, I=0.00035372, L=218.45945325, omega_bar=-0.32241464, Omega=-0.00508664),
"Pluto": dict(a=0.0026291, e=0.0005105, I=0.0035372, L=218.45945325, omega_bar=-0.32241464, Omega=-0.0508664)
}

rates_df = pd.DataFrame.from_dict(Solar_System_Rates, orient='index')

for col in ['I','L','omega_bar','Omega']:
  rates_df[col] *= 2 * np.pi / 360  # deg -> rad
rates_df['omega'] = rates_df['omega_bar'] - rates_df['Omega']

# transformation from 1/Cy to 1/sec
year_in_sec = 31536000
rates_df /= 100 * year_in_sec
rates_dict = rates_df.drop(['L', 'omega_bar'], axis=1).to_dict(orient='index')


# Orbital parameters for selected moons in the Solar System
#
# Source: Wikipedia
#
# Units a:km, T:days
# M wasn't provided and will be fitted using errors

Moons_Keplerian_Elements = {
  "Io":dict(a=421_700,  T=1.7691, parent='Jupiter'),
  "Europa":dict(a=671_034,  T=3.5512, parent='Jupiter'),
  "Ganymede":dict(a=1_070_412,  T=7.1546, parent='Jupiter'),
  "Callisto":dict(a=1_882_709,  T=16.689, parent='Jupiter'),
  "Rhea":dict(a=527_108,  T=4.5, parent='Saturn'),
  "Titan":dict(a=1_221_870,  T=16, parent='Saturn'),
  "Iapetus":dict(a=3_560_820,  T=79, parent='Saturn'),

}

moons_df = pd.DataFrame.from_dict(Moons_Keplerian_Elements, orient='index')

moons_df['a'] = moons_df['a'].div(149_597_871)  # km -> AU
moons_df['T'] =  moons_df['T'].apply(lambda x: timedelta(x).total_seconds())  # sideral days -> seconds
  
# dictionary with keys are Planet names and value as dictionary of orbital parameters 
moons_dict = moons_df.to_dict(orient='index')
