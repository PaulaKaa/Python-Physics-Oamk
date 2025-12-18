"""
Määrittele havainnoista kussilla oppimasi perusteella seuraavat asiat ja esitä ne numeroina visualisoinnissasi:
- Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta
- Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella
- Keskinopeus (GPS-datasta)
- Kuljettu matka (GPS-datasta)
- Askelpituus (lasketun askelmäärän ja matkan perusteella)
"""
#Import necessary libraries
#python -m streamlit run FY-projekti.py

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

#Get linear acceleration and location data 
path_Linear_Acceleration = './FYDataProjekti/LinearAcceleration.csv'
df_Linear = pd.read_csv(path_Linear_Acceleration)

path_Location = './FYDataProjekti/Location.csv'
df_Location = pd.read_csv(path_Location)

#Read data
df_Linear.head() #Check data what kind of it is 

#-----------------------------------
#Left sidebar
with st.sidebar:
    st.header("Info")
    st.markdown(":rainbow[Tämä työ on Soveltavan matematiikan ja fysiikan kurssin fysiikan osion loppuprojekti.]")
    st.write("Oamk 12/2025")

#-----------------------------------
#Main section 
     
#Header
st.title("My walking trail")
st.divider()

#----------------------------COUNTINGS


#-----------Filter the linear acceleration data ---
#Count steps from filtered data

#Bring a low pass filter function
from scipy.signal import butter, filtfilt
def butter_lowpass_filter(data, cutoff, fs, nyq, order): 
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter (order, normal_cutoff, btype='low', analog=False) 
    y = filtfilt(b, a, data)
    return y

#Define variables and filter the data
data = df_Linear['Linear Acceleration z (m/s^2)'] #Chose z, help of plots
T_total = df_Linear['Time (s)'].max() #The length of the data
n = len(df_Linear['Time (s)']) #The count datapiste
fs = n/T_total #Assume vakio näytteenottotaahyys
nyq = fs/2 #Nyqvistin taajuus
order = 3
cutoff = 1/0.3
filteredData = butter_lowpass_filter(data, cutoff, fs, nyq, order)



#-----------Count steps from the filtered data ---
stepsFilteredData = 0
for i in range(n-1):
    if filteredData[i]/filteredData[i+1] < 0:
        stepsFilteredData = stepsFilteredData + 1/2


#-----------Fourier and psd---
#Select z
signal = df_Linear['Linear Acceleration z (m/s^2)']
t = df_Linear['Time (s)']
N = len(signal)
dt = np.max(t)/N # Näytteenottoväli

fourier = np.fft.fft(signal, N) 
psd = fourier*np.conj(fourier)/N
freq = np.fft.fftfreq(N,dt)
Limit = np.arange(1,int(N/2))


#-----------Count steps from the fourier ---
f_max = freq[Limit][psd[Limit] == np.max(psd[Limit])][0] #DOminoiva taajus
T = 1/f_max
stepsFourier = f_max*np.max(t)




from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

#Count walked
path_Location = './FYDataProjekti/Location.csv'
df_Location = pd.read_csv(path_Location)

#New empty sarake
df_Location['Distance calculated'] = np.zeros(len(df_Location))

#Count distance 
for i in range(len(df_Location)-1):
    lon1 = df_Location['Longitude (°)'][i]
    lon2 = df_Location['Longitude (°)'][i+1]
    lat1 = df_Location['Latitude (°)'][i]
    lat2 = df_Location['Latitude (°)'][i+1]
    df_Location.loc[i+1, 'Distance calculated'] = haversine(lon1, lat1, lon2, lat2)

df_Location['Total distance'] = df_Location['Distance calculated'].cumsum()
#plt.plot(df_Location['Time (s)'],df_Location['Distance calculated'])
plt.plot(df_Location['Time (s)'],df_Location['Total distance'])
plt.grid()

plt.ylabel('Kokonaismatka')
plt.xlabel('Aika')
plt.show()

distanceKm = df_Location['Total distance'][(len(df_Location['Total distance'])-1)]
distanceM =  distanceKm * 1000
distanceCM =  distanceM * 100

#-----------The length of step ---
lengthOfStep = (distanceCM/stepsFilteredData)

#-----------The speed ---
df_Location['Time calculated'] = np.zeros(len(df_Location))

#Count total time 
for i in range(len(df_Location)-1):
    time1 = df_Location['Time (s)'][i]
    time2 = df_Location['Time (s)'][i+1]
    df_Location.loc[i+1, 'Time calculated'] = time2 - time1

df_Location['Total time passed'] = df_Location['Time calculated'].cumsum()
totalTime = df_Location['Total time passed'][len(df_Location['Total time passed'])-1]

speedMS = distanceM/totalTime
speedKMH = speedMS *3.6

st.markdown("🚶Askelmäärä **kiihtyvyysdatasta Fourier-analyysin** perusteella: ")
st.write(round(stepsFourier, 0), "askelta")

#Columns
left_column, right_column = st.columns(2)

with left_column:
    st.markdown("🚶Askelmäärä **suodatetusta kiihtyvyysdatasta** ")
    st.markdown("🏃Keskinopeus:")
    st.markdown("📏Kuljettu matka:")
    st.markdown("📏Askelpituus: ")

with right_column:        
    st.write(stepsFilteredData, "askelta")
    st.write(round(speedKMH,2), "km/h")
    st.write(round(distanceKm, 3), "km")
    st.write(round(lengthOfStep, 2), "cm")
st.divider()


#-----------------------------------    
#Show data as a table if wanted
st.header("Mitattu data")
if st.checkbox('Näytä lineaarisen kiihtyvyyden mitattu data'):
    st.write("Raaka data - Lineaarinen kiihtyvyys")
    st.dataframe(df_Linear)
if st.checkbox('Näytä sijainnin mitattu data'):
    st.write("Raaka data - Sijanti")
    st.dataframe(df_Location)
if st.checkbox('Näytä sijainnin mitattu data'):
    st.write("Raaka data - Sijanti")
    st.dataframe(df_Location)
st.divider()
#-----------------------------------


#-----------------------------------
#Tabs
st.header("Datasta luotuja kuvaajia")
tab1, tab2 = st.tabs(["Suodatettu kiihtyvyysdata", "Tehospektritiheys"])

with tab1:
    st.subheader('Suodatettu kiihtyvyysdata')
    st.write("Suodatettu kiihtyvyysdataa on käytetty askelmäärän määrittelemiseen. Askeldataa on 6 minuuttia, joten tässä on leikattu osio siitä, josta näkee askeleet.")

    #Draw line plot
    fig, ax = plt.subplots(figsize=(12,4))
    plt.plot(df_Linear['Time (s)'],filteredData,label='Suodatettu data')
    plt.axis([200,215,-7,7])
    plt.grid()
    plt.legend()
    st.pyplot(fig)

    #st.line_chart(filteredData, x='Time (s)', y='Linear Acceleration z (m/s^2)', y_label="Suodatettu ", x_label="Aika")
with tab2:
    st.subheader('Tehospektritiheys')
    st.write("Analyysiin valitun kiihtyvyysdatan z-komponentin tehospektritiheys.")

    #Draw line plot

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(freq[Limit],psd[Limit].real)
    #ax2.xlabel('Taajuus Hz = [Hz] = [1/s]')
    #ax2.ylabel('Teho')
    ax2.axis([0,14,0,11000])
    ax2.grid()
    ax2.legend()
    st.pyplot(fig2)

    #st.line_chart(df, x='Time (s)', y='Linear Acceleration x (m/s^2)', y_label="Teho", x_label="Taajuus")

#-----------------------------------


#-----------------------------------
st.divider()

#Map
st.header('Reitti kartalla')
st.write("Datan pohjalta piirrettyy kuljettu reitti kartalla.")

#Define center of the map
df = df_Location[df_Location['Horizontal Accuracy (m)'] < 10]
#df = df[df['Satellites'] > 30]
lat1 = df_Location['Latitude (°)'].mean()
long1 = df_Location['Longitude (°)'].mean() 

#Create map
my_map = folium.Map(location = [lat1,long1], zoom_start = 15) 

#Draw the trail
folium.PolyLine(df[['Latitude (°)','Longitude (°)']], color = 'red', weight = 3).add_to(my_map) 
st_map = st_folium(my_map, width= 900, height= 650 ) 
st.divider()

st.write("*Apuna on käytetty Soveltavan matematiikan ja fysiikan kurssin kurssimateriaalia sekä Streamlitin dokumentaatiota.*")


st.markdown("*Streamlit* is **really** ***cool***.")
st.markdown('''
    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
    :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')