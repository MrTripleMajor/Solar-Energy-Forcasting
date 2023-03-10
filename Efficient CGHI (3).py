
# The ~clearsky~ module contains several methods to calculate clear sky GHI, DNI, and DHI.
from __future__ import division
import math
import matplotlib.pyplot as plt
import pytz
import pvlib
import os
from collections import OrderedDict
import calendar
from matplotlib import docstring
import numpy as np
import pandas as pd
from pvlib import tools

# what does this do?


def ineichen(apparent_zenith, airmass_absolute,  linke_turbidity, altitude, dni_extra):
    ...


# Reading actual data
data = pd.read_csv('Full_SRRL_measurement_timeseries.csv', index_col=0)
apparent_zenith = data['Zenith']
airmass_relative = data['Airmass']

altitude = 30
# the above value may be MOOT


def lookup_linke_turbidity(time, latitude=74.45, longitude=40.52, filepath=None, interp_turbidity=True):
    ...


def DatetimeIndex(timedata, freq, tz, normalize=False, closed=None, ambiguous='raise', dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None):
    ...


timedata = data['DateTime']
freq = 'infer'

tz = pytz.timezone('US/Mountain')
#tz = pytz.timezone('America/New_York')

time = pd.DatetimeIndex(timedata, freq, tz, normalize=False, closed=None, ambiguous='raise',
                        dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None)

linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(
    time, latitude=74.45, longitude=40.52, filepath=None, interp_turbidity=True)

#pressure = pvlib.atmosphere.alt2pres(altitude)
linke_turbidity = 2.54
dni_extra = data['Dni']

#Radians = apparent_zenith * (math.pi/180)
#airmass_relative = 1/(math.cos(Radians)+0.50572*(96.07995-Radians)**(-1.6364))
altitude = 30
# richard weeks
print('airmass_Relative')


def alt2pres(altitude):
    ...


pressure = pvlib.atmosphere.alt2pres(altitude)


# This is the function that calculates the clear sky GHI
def get_absolute_airmass(airmass_relative, pressure):
    ...


airmass_absolute = pvlib.atmosphere.get_absolute_airmass(
    airmass_relative, pressure)

data_clearsky = pvlib.clearsky.ineichen(
    apparent_zenith, airmass_absolute, linke_turbidity, altitude, dni_extra)


data_actualghi = data['GHI']
data_csghi = data_clearsky['ghi']
#Grad_CSGHI = data['CSGHI']
#difference = data_csghi - Grad_CSGHI

# Saving csghi and actual ghi the data to a csv file
data_csghi.to_csv('csghi.csv')
# difference.to_csv('difference.csv')

# summary statistics GRAD CSGHI
#Average = np.mean(Grad_CSGHI)
#Standard_Deviation = np.std(Grad_CSGHI)
#Maximum = np.max(Grad_CSGHI)
#Minimum = np.min(Grad_CSGHI)
#Median = np.median(Grad_CSGHI)

#print("Summary Statistics Undergrad CSGHI")
# print(Average)
# print(Standard_Deviation)
# print(Maximum)
# print(Minimum)
# print(Median)

# summary statistics Undergrad CSGHI
Average = np.mean(data_csghi)
Standard_Deviation = np.std(data_csghi)
Maximum = np.max(data_csghi)
Minimum = np.min(data_csghi)
Median = np.median(data_csghi)

print("Summary Statistics Undergrad CSGHI")
print(Average)
print(Standard_Deviation)
print(Maximum)
print(Minimum)
print(Median)


# Summary Statistics difference
#Average = np.mean(difference)
#Standard_Deviation = np.std(difference)
#Maximum = np.max(difference)
#Minimum = np.min(difference)
#Median = np.median(difference)

print("Summary Statistics difference")
print(Average)
print(Standard_Deviation)
print(Maximum)
print(Minimum)
print(Median)

print("Summary Statistics Undergrad CSGHI")
print(Average)
print(Standard_Deviation)
print(Maximum)
print(Minimum)


# Plotting the data
plt.plot(data_actualghi, color='r', label='actual')
plt.plot(data_csghi, color='g', label='UGrad_CSGHI')
#plt.plot(Grad_CSGHI, color='y', label='Grad_CSGHI')
#plt.plot(difference, color='b', label='difference')
plt.show()
