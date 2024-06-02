# https://lasp.colorado.edu/lisird/data/nrl2_tsi_P1Y
# https://climate.nasa.gov/vital-signs/global-temperature/?intent=121
# https://gml.noaa.gov/ccgg/trends/data.html
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import pickle
import math
import statistics

################################################################################
# DATA DECOMPOSITION
# File data for Total solar income
nc_file = nc.Dataset('total solar energy income data')
time_data = nc_file.variables['time']
time_values = time_data[:]
time_units = time_data.units
tsi_data = nc_file.variables['TSI']


tsi = tsi_data[:]
year_tsi = []
for i in range(len(time_values)):
    year_tsi.append(1610+i)


# File data for Glabal annual average tempreture
filename_temp = "Tempreture changes data.txt"
with open(filename_temp, "r") as file:
    data = file.readlines()
data = data[5:]


years_temp = []
temp = []
for line in data:
    parts = line.strip().split()
    year = int(parts[0])
    annual_mean = float(parts[1])
    years_temp.append(year)
    temp.append(annual_mean)
    
    

# File data for CO2 concentration level on the atmosphere
filename_co2 = "co2_mm_mlo.txt"
with open(filename_co2, "r") as file:
    data_co2 = file.readlines()


years_decimal_co2 = []
co2_ppm = []
for line in data_co2:
    if line.startswith("#") or not line.strip():
        continue  
    parts = line.strip().split() 
    time = float(parts[2])
    co2 = float(parts[3])
    years_decimal_co2.append(time)
    co2_ppm.append(co2)

################################################################################

minor_tsi = np.polyval(np.polyfit(year_tsi, tsi, 1),year_tsi) # Do not need to use it, just there

plt.plot(year_tsi, tsi,label = "actual data")
plt.plot(year_tsi, minor_tsi, label="best linear fit")
plt.xlabel("year")
plt.ylabel("Solar energy income in watt")
plt.title("energy income vs year")
plt.legend()
plt.show()
detrend_tsi = tsi - minor_tsi


minor_temp = np.polyval(np.polyfit(years_temp, temp, 0),years_temp)

plt.plot(years_temp, temp,label = "actual data")
plt.plot(years_temp, minor_temp, label="best linear fit")
plt.xlabel("year")
plt.ylabel("tempreture compared to avergae tempreture in degree")
plt.title("average tempreture in the worldvs year")
plt.legend()
plt.show()
detrend_temp = temp - minor_temp


minor_co2 = np.polyval(np.polyfit(years_decimal_co2, co2_ppm, 1),years_decimal_co2)

plt.plot(years_decimal_co2, co2_ppm,label = "actual data")
plt.plot(years_decimal_co2, minor_co2, label="best linear fit")
plt.xlabel("year")
plt.ylabel("CO2 concentration in ppm")
plt.title("CO2 concentration change over the year")
plt.legend()
plt.show()
detrend_co2 = co2_ppm - minor_co2

################################################################################
"""
FFT of tsi to find the peaks
"""
fft_tsi_axis = np.fft.fftshift(np.fft.fftfreq(len(year_tsi), 1))
fft_tsi = np.fft.fftshift(np.fft.fft(detrend_tsi))

index_cutoff_1_1 = np.where((0.08 < fft_tsi_axis) & (fft_tsi_axis < 0.105))
index_cutoff_1_2 = np.where((-0.08 > fft_tsi_axis) & (fft_tsi_axis > -0.105))
fft_tsi_1 = fft_tsi.copy()
fft_tsi_1[index_cutoff_1_1] = 0
fft_tsi_1[index_cutoff_1_2] = 0

tsi_1 = np.fft.ifft(np.fft.ifftshift(fft_tsi_1))
tsi_1 = tsi_1 + minor_tsi

#riu
plt.plot(year_tsi,tsi_1)
plt.xlabel("year")
plt.ylabel("Solar energy income in watt")
plt.title("tempreture vs year after cutting second biggest frequency")
plt.show()

index_cutoff_2_1 = np.where((0.0 < fft_tsi_axis) & (fft_tsi_axis < 0.035))
index_cutoff_2_2 = np.where((-0.0 > fft_tsi_axis) & (fft_tsi_axis > -0.035))
fft_tsi_1[index_cutoff_2_1] = 0
fft_tsi_1[index_cutoff_2_2] = 0

index_cutoff_3_1 = np.where((0.12 < fft_tsi_axis) & (fft_tsi_axis < 0.13))
index_cutoff_3_2 = np.where((-0.12 > fft_tsi_axis) & (fft_tsi_axis > -0.13))
fft_tsi_1[index_cutoff_3_1] = 0
fft_tsi_1[index_cutoff_3_2] = 0

plt.plot(fft_tsi_axis,np.abs(fft_tsi_1))
plt.xlabel("frequency in year")
plt.ylabel("Solar energy income power spectrum")
plt.title("Solar energy income power spectrum vs frequency after cuttoff")
plt.show()

"""
FFT of temp to delete the maximum frequency of tsi
"""
fft_temp_axis = np.fft.fftshift(np.fft.fftfreq(len(years_temp), 1))
fft_detrend_temp = np.fft.fftshift(np.fft.fft(detrend_temp))

index_cutoff_tsi_1 = np.where(((0.08 < fft_temp_axis) & (fft_temp_axis < 0.105)) | ((-0.08 > fft_temp_axis) & (fft_temp_axis > -0.105)))
index_cutoff_tsi_2 = np.where(((0.0 < fft_temp_axis) & (fft_temp_axis < 0.035)) | ((0 > fft_temp_axis) & (fft_temp_axis > -0.035)))
index_cutoff_tsi_3 = np.where(((0.12 < fft_temp_axis) & (fft_temp_axis < 0.13)) | ((-0.12 > fft_temp_axis) & (fft_temp_axis > -0.13)))
fft_temp_tsi = fft_detrend_temp.copy()
fft_temp_tsi[index_cutoff_tsi_1] = 0
fft_temp_tsi[index_cutoff_tsi_2] = 0
fft_temp_tsi[index_cutoff_tsi_3] = 0

temp_tsi = np.fft.ifft(np.fft.ifftshift(fft_temp_tsi))
temp_tsi = temp_tsi + minor_temp

plt.plot(fft_temp_axis,np.abs(fft_temp_tsi),)
plt.xlabel("year")
plt.ylabel("tempreture power spectrum")
plt.title("tempreture power sepctrum vs frequency for linear")
plt.show()

plt.plot(years_temp,temp_tsi, label = "tempreture increse without solar increase")
plt.plot(years_temp,temp, label = "original data")
plt.xlabel("year")
plt.ylabel("tempreture in degree")
plt.title("Estimated tempreture without solar income vs year for constant")
plt.legend()
plt.show()

"""
FFT of CO2 to find the maximum peak
"""

fft_co2_axis = np.fft.fftshift(np.fft.fftfreq(len(years_decimal_co2), 1/12))
fft_co2 = np.fft.fftshift(np.fft.fft(detrend_co2))

index_cutoff_co2_1_1 = np.where((0.975 < fft_co2_axis) & (fft_co2_axis < 1.025))
index_cutoff_co2_1_2 = np.where((-0.975 > fft_co2_axis) & (fft_co2_axis > -1.025))
fft_co2_1 = fft_co2.copy()
fft_co2_1[index_cutoff_co2_1_1] = 0
fft_co2_1[index_cutoff_co2_1_2] = 0

index_cutoff_co2_2_1 = np.where((0.0 < fft_co2_axis) & (fft_co2_axis < 0.075))
index_cutoff_co2_2_2 = np.where((-0.0 > fft_co2_axis) & (fft_co2_axis > -0.075))
fft_co2_1[index_cutoff_co2_2_1] = 0
fft_co2_1[index_cutoff_co2_2_2] = 0

index_cutoff_co2_3_1 = np.where((1.975 < fft_co2_axis) & (fft_co2_axis < 2.025))
index_cutoff_co2_3_2 = np.where((-1.975 > fft_co2_axis) & (fft_co2_axis > -2.025))
fft_co2_1[index_cutoff_co2_3_1] = 0
fft_co2_1[index_cutoff_co2_3_2] = 0

plt.plot(fft_co2_axis,np.abs(fft_co2_1), color = "red")
plt.plot(fft_co2_axis,np.abs(fft_co2_1),)
plt.xlabel("year")
plt.ylabel("CO2 power spectrum")
plt.title("CO2 concentration power sepctrum vs frequency after cutoff")
plt.show()

"""
FFT of temp to delete the maximum frequency of CO2
"""
fft_temp_axis = np.fft.fftshift(np.fft.fftfreq(len(years_temp), 1))
fft_detrend_temp = np.fft.fftshift(np.fft.fft(detrend_temp))

index_cutoff_3_1 = np.where(((0.975 < fft_temp_axis) & (fft_temp_axis < 1.025)) | ((-0.975 > fft_temp_axis) & (fft_temp_axis > -1.025)))
index_cutoff_3_2 = np.where(((0.0 < fft_temp_axis) & (fft_temp_axis < 0.075)) | ((0 > fft_temp_axis) & (fft_temp_axis > -0.075)))
index_cutoff_3_3 = np.where(((1.975 < fft_temp_axis) & (fft_temp_axis < 2.025)) | ((-1.975 > fft_temp_axis) & (fft_temp_axis > -2.025)))
fft_temp_co2 = fft_detrend_temp.copy()
fft_temp_co2[index_cutoff_3_1] = 0
fft_temp_co2[index_cutoff_3_2] = 0
fft_temp_co2[index_cutoff_3_3] = 0

temp_co2 = np.fft.ifft(np.fft.ifftshift(fft_temp_co2))
temp_co2 = temp_co2 + minor_temp

plt.plot(years_temp,temp_co2, label = "tempreture increse without solar increase")
plt.plot(years_temp,temp, label = "original data")
plt.xlabel("year")
plt.ylabel("tempreture in degree")
plt.title("Estimated tempreture without CO2 vs year for constant")
plt.legend()
plt.show()

###############################################################################

N = len(temp_co2)
CR_axis = np.arange(N)  
CR_axis = CR_axis - N/2

CR_tsi = np.fft.ifft(np.fft.fft(temp_tsi) * np.conj(np.fft.fft(temp)))
CR_tsi = np.fft.fftshift(CR_tsi)
CR_co2 = np.fft.ifft(np.fft.fft(temp_co2) * np.conj(np.fft.fft(temp)))
CR_co2 = np.fft.fftshift(CR_co2)

plt.plot(CR_axis, CR_tsi,label = "without solarincome")
plt.plot(CR_axis, CR_co2, label ="without CO2")
plt.xlabel("time lag")
plt.ylabel("cross correlation")
plt.legend()
plt.title("cross correlation comparison for constant")
plt.show()




















