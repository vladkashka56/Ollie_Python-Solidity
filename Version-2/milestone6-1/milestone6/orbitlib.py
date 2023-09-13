# ORBIT SIMULATOR
# software by Grzegorz Wiktorowicz (gwiktoro@gmail.com)
# version: 4 (220107)
#
# see notebook for exemplary usage
#
import numpy as np
from scipy.optimize import fsolve  # numerical equation solver
from datetime import datetime, timedelta

from collections import namedtuple

from copy import copy, deepcopy

# Methods related to the orbital motion in the orbital plane
# To be used as a component of the Orbit3D class
class Orbit2D:

    def __init__(self, a, e, M, T, verbose=False):

        self.a = a  # semi-major axis of the orbit
        self.e = e  # eccentricity
        self.M = M  # time of periastron passage
        self.T = T  # orbital period
        self.verbose = verbose

        self.E_init = np.pi  # [rad] Initial value for Euler equation solver

        self._update_n()  # setting mean angular velocity

    def _update_n(self):
        # sets the mean angular velocity (n) for the class
        # neets to be run every time the period (T) is set or updated
        self.n = 2 * np.pi / self.T
        if self.verbose:
            print('n:',self.n)

    def update(self, error='raise', **kwargs):
        """
        Usage: update([param=value])

        updates class parameters

        To be used for changes in orbital parameters, e.g. perihelion shift
        """

        for key, value in kwargs.items():
            try:
                value_old = getattr(self, key)
                setattr(self, key, value)
                if self.verbose:
                    print(f"Attr: {key} {value_old} -> {value}")
                if key == 'T':
                    # if period (T) is updated, update also the mean angular velocity (n)
                    self._update_n()
            except AttributeError:
                if error=='raise':
                    raise AttributeError(f"Error: Orbit2D: No such Attribute: {key}")
                else:
                    continue

    def _Kepler_equation(self, E, M):
        # to be used with equation solver to get eccentric anomaly (E) from mean anomaly (M) 
        return E - self.e * np.sin( E ) - M

    def get_rv(self, t):
        """ 
        Arguments: t [sec] - relative time from point zero (default J2000 epoch) 

        Returns: radius [AU] and true anomaly [rad]
        """ 

        M = self.n * t  + self.M  # actual mean anomaly
        if self.verbose:
            print("M:", M)

        E = fsolve( lambda E: self._Kepler_equation( E, M ), self.E_init )  # actual eccentric anomaly
        if self.verbose:
            print("E:", E)

        r = self.a * ( 1.0 - self.e * np.cos( E ) )  # radius (distance from baricenter)
        v = 2.0 * np.arctan( np.sqrt( ( 1.0 + self.e ) / ( 1.0 - self.e ) ) * np.tan( E / 2.0 ) )  # true anomaly

        return r[0], v[0]

    def get_xy(self, t):
        """
        Same as get_rv() but returns the objects position in cartesian coordinates

        Returns: x, y [AU] - x-axis is in the direction of pericenter, y-axis is right-hand perpendicular
        """

        r, v = self.get_rv(t)
        x = r * np.cos( v )
        y = r * np.sin( v )

        return np.array([x, y]).reshape(-1, 1)


class Orbit3D:

    def __init__(self, a=1, e=0, M=0, T=1, Omega=0, omega=0, I=0, verbose=False):
        """
        Arguments:
          a [AU] - semi-major axis (default 1)
          e [0-1] - eccentricity (default 0)
          M0 - mean anomaly at time zero (default 0; recommended to use J2000 epoch for Solar System)
          T [sec] - orbital period (default 1)
          Omega [rad] - longitude of the ascending node (default 0)
          omega [rad] - argument of the pericenter (default 0)
          i [rad] - inclination (default 0)
          verbose - debugging output (default False)
        """

        self.orbit2d = Orbit2D(a, e, M, T, verbose)

        self.a = a
        self.e = e
        self.M = M
        self.T = T
        self.Omega = Omega
        self.omega = omega
        self.I = I
        self.verbose = verbose

        self._update_Euler_angle_transformation_matrix()

    def _update_Euler_angle_transformation_matrix(self):


        cosO = np.cos(self.Omega)
        sinO = np.sin(self.Omega)
        rot_Omega = np.array([[cosO, -sinO, 0], [sinO, cosO, 0], [0, 0, 1]])
        cosI = np.cos(self.I)
        sinI = np.sin(self.I)
        rot_I = np.array([ [1, 0, 0], [0, cosI, -sinI], [0, sinI, cosI]])
        coso = np.cos(self.omega)
        sino = np.sin(self.omega)
        rot_omega = np.array([[coso, -sino, 0], [sino, coso, 0], [0, 0, 1]])

        self.Euler_angle_transformation_matrix =  rot_Omega @ rot_I @ rot_omega

    def update(self, error='raise', **kwargs):
        """
        Usage: update([param=value])

        updates class parameters

        To be used for changes in orbital parameters, e.g. perihelion shift
        """

        for key, value in kwargs.items():
            try:
                value_old = getattr(self, key)
                setattr(self, key, value)
                if self.verbose:
                    print(f"Attr: {key} {value_old} -> {value}")

                self.orbit2d.update(error='ignore', **kwargs)

            except AttributeError:
                if error=='raise':
                    raise AttributeError(f"Error: Orbit3D: No such Attribute: {key}")
                else:
                    continue

        self._update_Euler_angle_transformation_matrix()


    def get_xyz(self, t):
        """
        Argument: t [sec] - time since time zero [default to J2000 epoch]

        Returns: objects possition x, y, z in [AU] - x is directed at vernal equinox; y,z are right-handed ortogonal
        """

        p_2d = self.orbit2d.get_xy(t)
        p0 = np.pad(p_2d, ((0, 1), (0, 0)))
        p1 = self.Euler_angle_transformation_matrix @ p0

        # x, y, z = p1

        return p1

# Main class for orbits
class Orbit:

    def __init__(self, a=1, e=0, M=0, T=1, Omega=0, omega=0, I=0, barycenter = [], roc_funcs = None, verbose=False):
        """
        Arguments at a reference time:
          a [AU] - semi-major axis (default 1)
          e [0-1] - eccentricity (default 0)
          M0 - mean anomaly at time zero (default 0; recommended to use J2000 epoch for Solar System)
          T [sec] - orbital period (default 1)
          Omega [rad] - longitude of the ascending node (default 0)
          omega [rad] - argument of the pericenter (default 0)
          I [rad] - inclination (default 0)

          barycenter TODO

          roc_funcs TODO

          verbose - debugging output (default False)
        """
        self.orbit3d = Orbit3D(a, e, M, T, Omega, omega, I, verbose)

        self.a = a
        self.e = e
        self.M = M
        self.T = T
        self.Omega = Omega
        self.omega = omega
        self.I = I

        self.barycenter = np.array(barycenter).reshape(-1,1)

        self.roc_funcs = roc_funcs if roc_funcs is not None else {}

        self.verbose = verbose

    def __repr__(self):
        return f"Orbit(a={self.a}; e={self.e}; M={self.M}; T={self.T}; Omega={self.Omega}; omega={self.omega}; I={self.I})"

    def update(self, error='raise', **kwargs):
        """
        Usage: update([param=value])

        updates class parameters

        To be used for changes in orbital parameters, e.g. perihelion shift
        """

        for key, value in kwargs.items():
            try:
                value_old = getattr(self, key)
                setattr(self, key, deepcopy(value))
                if self.verbose:
                    print(f"Attr: {key} {value_old} -> {value}")

                self.orbit3d.update(error='ignore', **kwargs)

            except AttributeError:
                if error=='raise':
                    raise AttributeError(f"Error: Orbit: No such Attribute: {key}")
                else:
                    continue


    def get_xyz(self, t):

        # updating params
        # for param in self.params  
        updates={}
        for param, roc_func in self.roc_funcs.items():
            updates[param] = getattr(self, param) + roc_func(t)

        if len(updates)>0:
            self.orbit3d.update(**updates)

        return self.orbit3d.get_xyz(t)

# wrapper class to connect orbit with additional atributes
# puts orbit in the correct place in the system hierarchy
Odata = namedtuple('orbit_data','orbit parent')  

# Convenience class to create a system of orbits
class System:

    def __init__(self, verbose = False):

        self.verbose = verbose

        self.orbits = {}

    def add_orbit(self, name, parent=None, update=False, **params):
        """
        Arguments: 
          name [hashable] - uniq identifier
          parent          - name of the parent orbit (e.g. for the orbit of a moon, a name of the planet should be provided)
          update          - set to True if you want to overwrite an existing orbit
          [param = value] - Orbit class arguments
        """

        assert not name in self.orbits.keys() or update, "Orbit names must be unique!!! (or use update=True)"
        assert parent is None or parent in self.orbits.keys(), "Parent must be an existing orbit!!! (add parent orbit first)"

        self.orbits[name] = Odata(orbit=Orbit(verbose=self.verbose, **params), parent=parent)

    def get_xyzs(self, t, observer=None):
        """
        Argument: t [sec] - time since time zero [default to J2000 epoch]
                  observer = list-like of length three or str; if list-like these are
                             positions of the observer in kartesian coordinates
                             if string, it's a body from the added orbits list
        Returns: dictionary with key: (x, y, z)  # see Orbit3D.get_xyz()
        """

        ret = {}
        to_calculate = [*self.orbits.keys()]
        while to_calculate:
            to_calculate_new = []
            for name in to_calculate:
                if self.verbose:
                    print('name:', name)
                if name not in to_calculate:
                    continue
                odata = self.orbits[name]
                if odata.parent is None:
                    ret[name] = odata.orbit.get_xyz(t)
                else:
                    if odata.parent in ret.keys():
                        ret[name] = odata.orbit.get_xyz(t) + ret[odata.parent]
                    else:
                        # if parent haven't been already calculated, wait for the parent to be calculated
                        to_calculate_new.append(name)
            to_calculate = copy(to_calculate_new)



        # changing the observers position
        observer = observer or self.observer
        if observer:
            if type(observer) == str:
                observer_pos = ret[observer]
            else:
                observer_pos = np.array(observer).reshape(-1,1)

            for name, pos in ret.items():
                ret[name] = pos - observer_pos

        return ret


def cart2sph(p):
    """
    Conversion from cartesian to spherical coordinates
    Arguments: p - array-like of shape [n,3] where n>0 is the number of data points

    x axis points towards the (alpha = 0, delta = 0) point and xy plain is the delta = 0 plane 

    Changelog: 
      220321: ignoring errors about zero division - such objects 
      relate to the location of the center of spherical system, which is 
      general the location of the observer. Such sky bodies should be 
      removed which is efectively done by asigning them nan delta
    """
    d = np.linalg.norm(p, axis=1)
    x, y, z = p.T
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.arctan2(y, x) % (2 * np.pi)
        delta = np.arcsin(z / d)

    return alpha, delta

def cart_ecl2cart_eq(p, epsilon=0.4091):
    """
    conversion from ecliptic cartesian coordinates to equatorial ones
    Arguments: p - array-like of shape [n,3] where n>0 is the number of data points
               epsilon - axial tilt (default value for the Earth)
    """
    rot = np.array([[1, 0, 0], [0, np.cos(epsilon), -np.sin(epsilon)], [0, np.sin(epsilon), np.cos(epsilon)]])

    return p @ rot.T


def sph_eq2sph_hor_OLD(p, lat, lon, t):
    """
    Conversion from Equatorial to Horizontal coordinates (both spherical)
    Arguments: p - array-like of shape [n,2] where n>0 is the number of data points (alpha, delta)
               lat - latutude of the observer (north is positive) [rad]
               lon - longitude of the observer (east is positive) [rad]
               t   - time since J2000 [sec]

    """

    # calculating Greenwich Sideral Time
    tu = t / 3600 / 24  # sec -> day
    GST = 2*np.pi * ((0.7790572732640 + 1.00273781191135448 * tu) % 1)

    # calculating Local Sideral Time
    LST = GST + lon

    # calculating the hour angle
    alpha = p[:,0]
    h = LST - alpha

    # calculating the altitudes (a) and the azimuths (A) using spherical triangles
    delta = p[:,1]
    a = np.arcsin(np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(h))
    A = -np.arctan2(np.cos(delta) * np.sin(h), -np.cos(delta) * np.cos(h) * np.sin(lat) + np.sin(delta) * np.cos(lat) )

    return a, A

def sph_eq2sph_hor(df, lat, lon):
    """
    Conversion from Equatorial to Horizontal coordinates (both spherical)
    Arguments: df - pandas dataframe index - times [sec since J2000], columns = [RA, DEC]
               lat - latutude of the observer (north is positive) [rad]
               lon - longitude of the observer (east is positive) [rad]

    """
    # calculating Greenwich Sideral Time
    tu = df.index / 3600 / 24  # sec -> day get_ts_positions use case
    # tu = ((df.index - pd.to_datetime('2000-01-01 12:00')).total_seconds() / 24 / 3600).values
    


    GST = 2*np.pi * ((0.7790572732640 + 1.00273781191135448 * tu) % 1)
    # calculating Local Sideral Time
    LST = GST + lon

    # print(GST, lon, LST)

    # calculating the hour angle
    
    h = LST - df['RA']
    # print(h)

    # calculating the altitudes (a) and the azimuths (A) using spherical triangles
    delta = df['DEC']
    a = np.arcsin(np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(h))
    # print(np.cos(delta) * np.sin(h), -np.cos(delta) * np.cos(h) * np.sin(lat) + np.sin(delta) * np.cos(lat) )
    A = -np.arctan2(np.cos(delta) * np.sin(h), -np.cos(delta) * np.cos(h) * np.sin(lat) + np.sin(delta) * np.cos(lat) )
    # print("A:", A)

    return pd.DataFrame({'a':a, 'A':A}, index = df.index)


#######################################################################
#######################################################################

####   Convenience functions used druing the development process
from collections import defaultdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_system(orbits_dict, rates_dict={}, observer = None, verbose=False):
    """ Creates a system for positions calculations
    Arguments: orbits_dict - orbital parameters and names
               rates_dict - change rates of the orbital parameters for orbits
               observer - observer location
    """
    system = System(verbose=verbose)

    system.observer = observer  

    for name, params in orbits_dict.items():
        system.add_orbit(name, **params)
        if name in rates_dict.keys():
            roc_funcs={}
            for rate_name, rate_value in rates_dict[name].items():

                # lambda function here is a place holder for more complicated relations that can be used later without any additional complication to the code (e.g. sine functions)
                # `rate_value=rate_value` added to avoid the late binding side effect.
                roc_funcs[rate_name] = lambda t, rate_value=rate_value: rate_value * t 
            system.orbits[name].orbit.update(roc_funcs = roc_funcs)

    return system

def get_positions(system, start_time=None, end_time=None, n_points=2, time_index=None, use_tqdm=True):
    """
    get positions for a specified time range or time points
    Arguments: system - system of orbits for which positions will be calculated
               start_time - start time ( inclusive )
               end_tim - end time (inclusive)
               n_points - number of time points between start_time and end_time (must be >=2)
               time_index - specific points for which positions should be calculated
    Changelog:
      220321: secufing that pos matrix is at least 2D. Previously the code 
      doesn't work properly for single observations
    """
    positions = defaultdict(list)  # container for planets positions 
                                   # in this example we create multiple positions to 
                                   # show the orbits
    if time_index is None and start_time and end_time:
        time_index = pd.date_range(start_time, end_time, periods=n_points)
    elif time_index is None:
        raise AttributeError("Either time_stamp or start_time and end_time has to be provided")

    J2000 = datetime(2000, 1, 1, 12)
    time_points = (time_index - J2000).view(np.int64) // 1e9

    if use_tqdm:
        my_tqdm = tqdm
    else:
        my_tqdm = lambda x:x

    for t in my_tqdm(time_points):
        ret = system.get_xyzs(t)
        for name, pos in ret.items():
            positions[name].append(pos)
    for name, pos in positions.items():
        positions[name] = pd.DataFrame(np.atleast_2d(np.array(pos).squeeze()), index = time_index, columns=['x','y','z'])

    return positions

# convenience ploting function
def plot_positions(positions, size=2):
    # TODO
    layout = go.Layout(
            scene = dict(
                camera=dict(eye=dict(x=1, y=1, z=1)),
                aspectmode = 'data'
                )
            )
    fig = go.Figure( layout=layout)
    for name, data in positions.items():
        fig.add_trace(go.Scatter3d(
            x=data.x,
            y=data.y,
            z=data.z,
            customdata=data.index,
            mode='markers',
            name=name,
            hovertemplate='%{customdata}<br>x:%{x}<br>y:%{y}<br>z:%{z}',
            marker=dict(
                size=size,
                opacity=0.8
                )
            ))

    fig.show()


def get_sky_positions(positions, obliquity = 0.4091, coordinates='equatorial', obs_latitude=None, obs_longitude=None):
    """ Transforms positions in space to positions on the sky relative to an observer (currently only Earth-based observer is supported
    Arguments: positions - dictionary with orbit names and positions in space relative to the observer
               coordinates - 'equatorial' - equatorial coordinates 
                             'horizontal' - converts positions to horizontal 
                             coordinates (needs the position of the observer 
                             (obs_latitude and obs_longitude) to be specified
               obs_latitude - latitude of the observer (relevant only for coordinates = 'horizontal'
               obs_longitude - longitude of the observer (relevant only for coordinates = 'horizontal'
    """
    # valid only for the Earth Observer
    sky_positions = {}

    for name, data_cart_ecl in positions.items():
        data_cart_eq = cart_ecl2cart_eq(data_cart_ecl.values, epsilon=obliquity)
        RA, DEC = cart2sph(data_cart_eq)
        sky_positions[name] = pd.DataFrame({"RA":RA, "DEC":DEC}, index=data_cart_ecl.index)
        if coordinates == 'horizontal':
            if obs_latitude is None or obs_longitude is None:
                raise Exception("Observer location (obs_latitude and obs_longitude) has to be provided for coordinates = 'equatorial'")
            sky_positions[name] = sph_eq2sph_hor(sky_positions[name], lat=obs_latitude, lon=obs_longitude)


    return sky_positions

# convenience plotting function for the Mercator projection of the sky
def plot_sky_positions_Mercator(sky_positions, xcol='RA', ycol='DEC', size=2):
    """ Plots positions on the sky in equatorial coordinates as observed by the observer
        Uses Mercator projection
        Arguments: sky_positions - positions on the sky as computed with 
                       get_sky_positions(coordinates='equatorial'), i.e. 
                       distionary with body names as keys and dataframes with 
                       columns RA, DEC and DatetimeIndex as values 
                   xcol, ycol - column names for ploting (default to 'RA' and 'DEC')
                   size - size of points on the plot
    """
    
    fig = go.Figure()
    for name, data in sky_positions.items():
        fig.add_trace(go.Scatter(
            x=data[xcol],
            y=data[ycol],
            customdata=data.index,
            mode='markers',
            name=name,
            hovertemplate='%{customdata}<br>RA :%{x}<br>DEC:%{y}',
            marker=dict(
                size=size,
                opacity=0.8
                )
            ))
        fig.update_layout(
                xaxis_title='Right Ascension [rad]',
                yaxis_title='Declination [rad]'
                )
    fig.show()

def plot_sky_positions_Polar(sky_positions, rcol='a', thetacol='A', size=10):
    """ Plots positions on the sky in horizontal coordinates as observed by the observer
        Uses Polar projection
        Arguments: sky_positions - positions on the sky as computed with 
                       get_sky_positions(coordinates='horizontal'), i.e. 
                       distionary with body names as keys and dataframes with
                       columns RA, DEC and DatetimeIndex as values 
                   rcol, thetacol - column names for ploting (default to 'a' and 'A')
                   size - size of points on the plot
    """       
    
    fig = go.Figure()
    for name, data in sky_positions.items():
        fig.add_trace(go.Scatterpolar(
            r=(np.pi/2 - data.a)*180/np.pi,
            theta = data.A*180/np.pi,
            customdata=data.index,
            mode='markers',
            name=name,
            hovertemplate='%{customdata}<br>z :%{r}<br>A:%{theta}',
            marker=dict(
                size=size,
                opacity=0.8
                )
            ))
        fig.update_polars(radialaxis=dict(range=[0, 90]))

    fig.show()

def show_sky_view(system, lat, lon, time):
    """ Convenience funtion to show the system (e.g. the Solar System) as view
    from a specific location and in specific time in the Fish Eye view (Polar 
    progection of horizontal coordinates)

    Arguments: system - System class instance
               lat, lon - latitude and longitude of the observer [degrees]
               time - observation time (UTC)
    """

    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    positions = get_positions(system, time_index=pd.DatetimeIndex(list(time)), use_tqdm=False)
    sky_positions = get_sky_positions(positions, coordinates='horizontal', obs_latitude=lat, obs_longitude=lon)
    # print("sky position")
    # print(sky_positions)
    plot_sky_positions_Polar(sky_positions)

# Tests for default parameters
def get_error(obs, system, obj, plot=True, verbose=True):
    positions = ol.get_positions(system, time_index = obs.Time)

    sky_positions = ol.get_sky_positions({obj:positions[obj]})[obj]

    tmp = obs.set_index('Time')
    tmp['RA_sim'] = sky_positions.RA
    tmp['Dec_sim'] = sky_positions.DEC
    rad2arcmin = 360 * 60 / (2 * np.pi)

    error_RA = (tmp['RA'] - tmp['RA_sim']) * rad2arcmin
    if plot:
        error_RA.plot(marker='o')
        plt.ylabel('Error [arcmin]')
        plt.title("Right ascension")
        plt.show()

    error_Dec = (tmp['Dec'] - tmp['Dec_sim']) * rad2arcmin
    if plot:
        error_Dec.plot(marker='o')
        plt.ylabel('Error [arcmin]')
        plt.title('Declination')
        plt.show()
    
    error = np.mean((error_RA**2 + error_Dec**2)**0.5)
    if verbose:
        print("Mean Error:",error,'[arcmin]')
    
    return error


# Most of the observational data, esp. old, is in the form of the card format

def read_card_format(filename):
    data = []
    for line in open(filename, 'r').readlines():
        planet = line[1:4]
        JD = int(line[4:16]) / 1e5
        Time = pd.to_datetime(JD, unit='D', origin='julian')
        RA_h = int(line[33:35])
        RA_m = int(line[35:37])
        RA_s = int(line[37:42])/1e3
        RA = (RA_h + RA_m / 60 + RA_s / 3600) * np.pi / 12
        DEC_d_str = line[50:53]
        DEC_sign = -1 if '-' in DEC_d_str else 1
        DEC_d = int(DEC_d_str)
        DEC_m = int(line[53:55])
        DEC_s = int(line[55:59])/1e2
        
        # sign for declination is for the entire number, not only for the degree part
        DEC = (np.abs(DEC_d) + DEC_m / 60 + DEC_s / 3600) * np.pi / 180 * DEC_sign
        data.append({'planet':planet, 'Time':Time, 'RA':RA, 'DEC':DEC})
    
    return pd.DataFrame(data)



def get_ts_positions(system, time_index):
    """
    get positions for a specified time range or time points
    Arguments: system - system of orbits for which positions will be calculated
               start_time - start time ( inclusive )
               end_tim - end time (inclusive)
               n_points - number of time points between start_time and end_time (must be >=2)
               time_index - specific points for which positions should be calculated
    """
    positions = defaultdict(list)  # container for planets positions 
                                   # in this example we create multiple positions to 
                                   # show the orbits
    ret = system.get_xyzs(time_index)
    for name, pos in ret.items():
        positions[name].append(pos)

    for name, pos in positions.items():
        positions[name] = pd.DataFrame(np.atleast_2d(np.array(pos).squeeze()), index = [time_index], columns=['x','y','z'])

    return positions
