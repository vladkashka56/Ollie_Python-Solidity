# CORBIT SIMULATOR
# software by Grzegorz Wiktorowicz (gwiktoro@gmail.com)
# version: 2 (211211)
#
# see notebook for exemplary usage
#
import numpy as np
from scipy.optimize import fsolve  # numerical equation solver
from datetime import datetime, timedelta

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

    return np.array([x, y, r, v]).reshape(-1, 1)


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
    # print('matrix')
    # print(self.Euler_angle_transformation_matrix)
    p1 = self.Euler_angle_transformation_matrix @ p0

    # x, y, z = p1

    return p1

# Main class for orbits
class Orbit:

  def __init__(self, a=1, e=0, M=0, T=1, Omega=0, omega=0, I=0, barycenter = [], roc_funcs = {}, verbose=False):
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

    self.a0 = a
    self.e0 = e
    self.M0 = M
    self.T0 = T
    self.Omega0 = Omega
    self.omega0 = omega
    self.I0 = I

    self.barycenter = np.array(barycenter).reshape(-1,1)

    self.roc_funcs = roc_funcs
    
    self.verbose = verbose

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
      updates[param] = getattr(self, param+'0') + roc_func(t)
    
    if len(updates)>0:
      self.orbit3d.update(**updates)

    return self.orbit3d.get_xyz(t)


  def get_xy(self, t):

    # updating params
    # for param in self.params  

    orbit2d = self.orbit3d.orbit2d
    return orbit2d.get_xy(t)





# Convenience class to create a system of orbits
class System:

  def __init__(self, verbose = False, observer = None):

    self.verbose = verbose
    self.observer = observer

    self.orbits = {}

  def add_orbit(self, name, **params):
    """
    Arguments: 
      name [hashable] - uniq identifier
      [param = value] - Orbit class arguments
    """
    print("orbit param")
    print(name, self.verbose, params)               
    self.orbits[name] = Orbit(verbose=self.verbose, **params)

  def get_xyzs(self, t, observer=None):
    """
    Argument: t [sec] - time since time zero [default to J2000 epoch]
              observer = list-like of length three or str; if list-like these are
                         positions of the observer in kartesian coordinates
                         if string, it's a body from the added orbits list
    Returns: dictionary with key: (x, y, z)  # see Orbit3D.get_xyz()
    """

    ret = {}
    for name, orbit in self.orbits.items():
      if self.verbose:
        print('name:', name)
      ret[name] = orbit.get_xyz(t)

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

  def get_xys(self, t, observer=None):
    """
    Argument: t [sec] - time since time zero [default to J2000 epoch]
              observer = list-like of length three or str; if list-like these are
                         positions of the observer in kartesian coordinates
                         if string, it's a body from the added orbits list
    Returns: dictionary with key: (x, y)  # see Orbit2D.get_xy()
    """

    ret = {}
    for name, orbit in self.orbits.items():
      if self.verbose:
        print('name:', name)
      ret[name] = orbit.get_xy(t)

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
  """

  x, y, z = p.T
  alpha = np.arctan2(y, x) % (2 * np.pi)
  delta = np.arcsin(z / np.linalg.norm(p, axis=1))

  return alpha, delta

def cart_ecl2cart_eq(p, epsilon=0.4091):
  """
  conversion from ecliptic cartesian coordinates to equatorial ones
  Arguments: p - array-like of shape [n,3] where n>0 is the number of data points
             epsilon - axial tilt (default value for the Earth)
  """

  rot = np.array([[1, 0, 0], [0, np.cos(epsilon), -np.sin(epsilon)], [0, np.sin(epsilon), np.cos(epsilon)]])

  return p @ rot.T




#######################################################################
#######################################################################

####   Convenience functions used druing the development process
from collections import defaultdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def create_system(orbits_dict, rates_dict={}, observer = None):
    """ Creates a system for positions calculations
    Arguments: orbits_dict - orbital parameters and names
               rates_dict - change rates of the orbital parameters for orbits
               observer - observer location
    """
    system = System()

    system.observer = observer  

    for name, params in orbits_dict.items():
        system.add_orbit(name, **params)
        if name in rates_dict.keys():
            roc_funcs={}
            for rate_name, rate_value in rates_dict[name].items():
                roc_funcs[rate_name] = lambda t: rate_value * t
    
            system.orbits[name].update(roc_funcs = roc_funcs)
    
    return system

def get_positions(system, start_time=None, end_time=None, n_points=2, time_index=None):
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
    if time_index is None and start_time and end_time:
        time_index = pd.date_range(start_time, end_time, periods=n_points)
    elif time_index is None:
        raise AttributeError("Either time_stamp or start_time and end_time has to be provided")

    J2000 = datetime(2000, 1, 1, 12)
    time_points = (time_index - J2000).view(np.int64) // 1e9

    for t in time_points:
      ret = system.get_xyzs(t)
      for name, pos in ret.items():
        positions[name].append(pos)


    for name, pos in positions.items():
      positions[name] = pd.DataFrame(np.array(pos).squeeze(), index = time_index, columns=['x','y','z'])
    
    return positions

# convenience ploting function
def plot_positions(positions, size=2):
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


def get_sky_positions(positions):
    """ Transforms positions in space to positions on the sky relative to an observer (currently only Earth-based observer is supported
    Arguments: positions - dictionary with orbit names and positions in space relative to the observer
    """
    # valid only for the Earth Observer
    sky_positions = {}
    
    for name, data_cart_ecl in positions.items():
        data_cart_eq = cart_ecl2cart_eq(data_cart_ecl.values)
        RA, DEC = cart2sph(data_cart_eq)
        
        sky_positions[name] = pd.DataFrame({"RA":RA, "DEC":DEC}, index=data_cart_ecl.index)
    
    return sky_positions

# convenience plotting function for the Mercator projection of the sky
def plot_sky_positions(sky_positions, size=2):
    # Mercator projection
    
    fig = go.Figure()
    for name, data in sky_positions.items():
        fig.add_trace(go.Scatter(
            x=data.RA,
            y=data.DEC,
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



def get_ts_3d_positions(system, time_index):
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
    
    return positions

def get_ts_2d_positions(system, time_index):
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

    ret = system.get_xys(time_index)
    for name, pos in ret.items():
      positions[name].append(pos)
    
    return positions