#python -m streamlit run FY-projekti.py
#Import necessary libraries

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

#Get linear acceleration and location data 
path_Linear_Acceleration = "https://raw.githubusercontent.com/PaulaKaa/Python-Physics-Oamk/refs/heads/main/LinearAcceleration.csv"
df_Linear = pd.read_csv(path_Linear_Acceleration)

path_Location = "https://raw.githubusercontent.com/PaulaKaa/Python-Physics-Oamk/refs/heads/main/Location.csv"
df_Location = pd.read_csv(path_Location)

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
n = len(df_Linear['Time (s)']) #The count data point
fs = n/T_total #Sampling frequency (constant)
nyq = fs/2 #Nyqvistin frequency
order = 3
cutoff = 1/0.3
filteredData = butter_lowpass_filter(data, cutoff, fs, nyq, order)


#Count steps from the filtered data
stepsFilteredData = 0
for i in range(n-1):
    if filteredData[i]/filteredData[i+1] < 0:
        stepsFilteredData = stepsFilteredData + 1/2


#Fourier and psd
#Select z
signal = df_Linear['Linear Acceleration z (m/s^2)']
t = df_Linear['Time (s)']
N = len(signal)
dt = np.max(t)/N # Sampling interval

fourier = np.fft.fft(signal, N) 
psd = fourier*np.conj(fourier)/N
freq = np.fft.fftfreq(N,dt)
Limit = np.arange(1,int(N/2))


#Count steps from the fourier
f_max = freq[Limit][psd[Limit] == np.max(psd[Limit])][0] #Dominant frequency
T = 1/f_max
stepsFourier = f_max*np.max(t)


#The Haversine formula
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

#Count walked distance
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

distanceKm = df_Location['Total distance'][(len(df_Location['Total distance'])-1)]
distanceM =  distanceKm * 1000
distanceCM =  distanceM * 100

#The length of step
lengthOfStep = (distanceCM/stepsFilteredData)

#The speed
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


#-----------------------------------
#Data visualization

#Left sidebar
with st.sidebar:
    st.header("Info")
    st.markdown(":rainbow[Tämä työ on Soveltavan matematiikan ja fysiikan kurssin fysiikan osion loppuprojekti.]")
    st.write("Oamk 12/2025")

#Main section      
#Header
st.title("Minun kävelyni")
st.divider()

#Paragh
st.write("Datan keruuta varten kävelin kuusi (6) minuuttia eri tavoin. Ensimmäisessä osiossa kävelin hitaasti, toisessa osiossa lävelin rivakasti ja viimeissä pätkässä juoksin. Datan keruu tapahtui suhteellisen tasaisella alueella lukuunottamatta alkumatkan jyrkkää alamäkeä.")
st.divider()

st.markdown("🚶Askelmäärä **kiihtyvyysdatasta Fourier-analyysin** perusteella: :green[868] askelta")
#Askeleet on kova koodattu lasketusta arvosta, koska ne eivät näyttäneet visualisesti hyviltä. Parempi tapa olisi käyttää muuttujia.
#st.write(round(stepsFourier, 0), "askelta")

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
#Tabs
st.header("Datasta luotuja kuvaajia")
tab1, tab2, tab3, tab4 = st.tabs(["Suodatettu kiihtyvyysdata", "Tehospektritiheys", "Mitattu data", "Kuljettu matka"])

with tab1:
    st.subheader('Suodatettu kiihtyvyysdata')
    st.write("Suodatettu kiihtyvyysdataa on käytetty askelmäärän määrittelemiseen. Askeldataa on 6 minuuttia, joten tässä on leikattu osio siitä, josta näkee askeleet.")

    #Draw line plot
    fig, ax = plt.subplots(figsize=(10,5))
    plt.plot(df_Linear['Time (s)'],filteredData,)
    ax.set(xlabel='Aika', ylabel='Kiihtyvyys z', title='Suodatettu data')
    plt.axis([200,215,-7,7])
    plt.grid()
    st.pyplot(fig)

with tab2:
    st.subheader('Tehospektritiheys')
    st.write("Analyysiin valitun kiihtyvyysdatan z-komponentin tehospektritiheys.")

    #Draw line plot
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(freq[Limit],psd[Limit].real)
    ax.set(xlabel='Taajuus Hz', ylabel='Teho', title='Tehospektri')
    ax.axis([0,14,0,11000])
    ax.grid()
    st.pyplot(fig)

    #st.line_chart(df, x='Time (s)', y='Linear Acceleration x (m/s^2)', y_label="Teho", x_label="Taajuus")

with tab3:
    st.subheader('Mitattu data')
    st.write("Tähän kuvaajaan on piirretty data alkutilanteessa ennen kuin sitä on lähdetty operoimaan. Kuvaajasta huomaa hyvin, kuinka nopeus nousee hitaasta kävelystä reippaaseen kävelyyn ja reippaasta kävelystä juoksuun. Noissa kohdissa kiihtyvyys kasvaa.")
    
    #Draw line plot
    fig, ax = plt.subplots(figsize=(12,5))
    plt.plot(df_Linear['Time (s)'],data)
    ax.set(xlabel='Aika', ylabel='Kiihtyvyys z', title='Alkuperäinen data')
    plt.grid()
    st.pyplot(fig)

with tab4:
    st.subheader('Kuljettu matka')
    st.write("Kuljettu matka on laskettu datapisteiden välisestä matkasta Haversine-funktion avulla. Kuvaajaan on nuolilla piirretty kohdat, joissa etenemisnopeus kasvaa ja tässä tapauksessa matkan kertymä kasvaa nopeammin (eli jyrkempi viiva).")

    #Draw line plot
    fig, ax = plt.subplots(figsize=(10,5))
    plt.plot(df_Location['Time (s)'],df_Location['Total distance'])
    ax.set(xlabel='Aika', ylabel='Matka', title='Kuljettu matka yhteensä')
    ax.annotate('Hidas kävely --> reipas kävely', xy=(150, 0.177), xytext=(3, 0.25),
            arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('Reipas kävelys --> juoksu', xy=(290, 0.412), xytext=(140, 0.55),
            arrowprops=dict(facecolor='black', shrink=0.05))
    plt.grid()
    st.pyplot(fig)

st.divider()

#-----------------------------------
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

st.write("*Apuna on käytetty Soveltavan matematiikan ja fysiikan kurssin kurssimateriaalia sekä Streamlitin ja Matplotlibin dokumentaatiota.*")
