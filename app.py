# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

#import numpy.random as npr
import pandas as pd
# data visualisation and manipulation
import numpy as np


# sets matplotlib to inline and displays graphs below the corressponding cell.
import matplotlib.pyplot as plt
from matplotlib import style

#%matplotlib inline
#import matplotlib.cm as cm

style.use('fivethirtyeight')

import seaborn as sns
sns.set_style('white')

import streamlit as st
import folium
from streamlit_folium import folium_static

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>©℗Accredian. All rights reserved <a style='display: block; color: #00008B ;text-align: center;' href="https://accredian.com/" target="_blank">Accredian</a></p>
</div>
"""


# Creating a helper function for reading any csv file
def get_data(path):
    # Reading from the path
    # For now we will make it for only csv files
    # Later on as complexcity arises we can elaborate the same approach for different files
    my_data = pd.read_csv(path)

    # getting the shape of the data
    #print('The shape of the dataset is ', my_data.shape)
    # returning the dataframe
    return my_data

# """This path needs to be mapped with the input data"""
# #NOTE This path itself needs to be mapped with the Form

def separate_categorical_and_numerical(data):
    categorical_columns, numerical_columns = [], []
    for i in new_data.columns:
        if new_data[i].dtypes == 'object':
            categorical_columns.append(i)
        else:
            numerical_columns.append(i)

    return categorical_columns, numerical_columns



def data_cleaning(categorical_columns,numerical_columns,data):
    from tqdm.notebook import tqdm
    # Replacing the null values with the mode for categorcial variables
    for i in tqdm(categorical_columns):
        data[i].fillna(data[i].mode()[0], inplace = True)
        #print('Categorical Data Cleaned')

    #Replacing the null values with the mean for contnious variables
    for j in tqdm(numerical_columns):
        data[j].fillna(np.mean(data[j]), inplace = True)
        #print('Continious Data Cleaned')

    # occurrence based on all columns
    duplicate_data = data[data.duplicated()]
    #print("Duplicate Rows for data:")
    #duplicate_data

    # Return Statement
    data_cleaned = data
    return data_cleaned


def plot_map_with_folium(data):
    data_columns  = data.columns
    if 'Latitude' in data_columns:
        if 'Longitude' in data_columns:

            bike_station_locations = new_data[["Latitude", "Longitude"]]
            map = folium.Map(location=[bike_station_locations.Latitude.mean(),
                                       bike_station_locations.Longitude.mean()])
            for index, location_info in bike_station_locations.iterrows():
                #print(location_info)
                folium.Marker([location_info["Latitude"], location_info["Longitude"]]).add_to(map)
    elif 'latitude' in data_columns:
        if 'longitude' in data_columns:

            bike_station_locations = new_data[["latitude", "longitude"]]
            map = folium.Map(location=[bike_station_locations.latitude.mean(),
                                       bike_station_locations.longitude.mean()])
            for index, location_info in bike_station_locations.iterrows():
                #print(location_info)
                folium.Marker([location_info["latitude"], location_info["longitude"]]).add_to(map)

    elif 'lat' in data_columns:
        if 'long' in data_columns:
            bike_station_locations = new_data[["lat", "long"]]
            map = folium.Map(location=[bike_station_locations.lat.mean(),
                                       bike_station_locations.Long.mean()])
            for index, location_info in bike_station_locations.iterrows():
                # print(location_info)
                folium.Marker([location_info["lat"], location_info["long"]]).add_to(map)

    elif 'Lat' in data_columns:
        if 'Long' in data_columns:

            bike_station_locations = new_data[["Lat", "Long"]]
            map = folium.Map(location=[bike_station_locations.Lat.mean(),
                                       bike_station_locations.Long.mean()])
            for index, location_info in bike_station_locations.iterrows():
                # print(location_info)
                folium.Marker([location_info["Lat"], location_info["Long"]]).add_to(map)

    return map




# The following function plots 2 respective subplots for continious features
def plot_continious_data(data_cleaned):
    # Importing the dependencies
    from tqdm.notebook import tqdm
    import seaborn as sns
    import matplotlib.pyplot as pltCollab

    # Separating numerical and categrical columns
    categorical_cols, numerical_cols = separate_categorical_and_numerical(data_cleaned)

    for i in tqdm(numerical_cols):
        plt.figure(figsize=[15, 3])
        sns.set(style="darkgrid")
        plt.subplot(1, 2, 1)
        sns.boxplot(data=data_cleaned, x=i, color='orange')
        plt.subplot(1, 2, 2)
        sns.distplot(a=data_cleaned[i], color='darkblue')
        plt.xticks(rotation=45)
        plt.grid()
        st.pyplot(plt)




def plot_categorical_data(data_cleaned):
    # Importing the dependencies
    from tqdm.notebook import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Separating numerical and categrical columns
    categorical_cols, numerical_cols = separate_categorical_and_numerical(data_cleaned)

    for i in tqdm(categorical_cols):
        values = data_cleaned[i].value_counts().head(10).values
        labels = data_cleaned[i].value_counts().head(10).index
        plt.figure(figsize=[20, 10])
        sns.set(style="darkgrid")
        plt.subplot(1, 2, 1)
        plt.barh(width=values, y=labels, color='orange')
        plt.xlabel('Count')
        plt.ylabel(i)
        plt.title('Top 10 values of {}'.format(i))
        plt.subplot(1, 2, 2)
        plt.pie(values, labels=labels, autopct='%1.0f%%')
        plt.title('Percentage Distribution for the Top 10 values of {}'.format(i))
        plt.xticks(rotation=45)
        plt.grid()
        st.pyplot(plt)

def get_info(data_cleaned):
    null_count = []
    datatypes = []
    for i in data_cleaned.columns:
        #print(data_cleaned[i].isnull().sum())
        #print(data_cleaned[i].dtypes)
        null_count.append(data_cleaned[i].isnull().sum())
        datatypes.append(data_cleaned[i].dtypes)
    df = pd.DataFrame({'ID':data_cleaned.columns,
                    'Null_Counts':null_count,
                    'DataTypes':datatypes})
    return df.astype(str)


def convert_df(df):
    return df.to_csv().encode('utf-8')

def get_report(df):
    csv = convert_df(df)
    st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )


col1,col2,col3 = st.columns(3)
# with col2:
#     st.image('./Accredian Form Header.png' , width=300)

buff1,col= st.columns([1,5])
with col:
    st.header('CDF Student Capstone Dashboard')



def get_observations(categorical_columns_after_data_cleaning, numerical_columns_after_data_cleaning, data_cleaned):
    numerical_observations, categorical_observations = [], []
    numerical_means = []
    numerical_stds = []
    numerical_skew_type = []
    numerical_ids, categorical_ids = [], []

    for i in numerical_columns_after_data_cleaning:
        if np.mean(data_cleaned[i]) > np.std(data_cleaned[i]):
            numerical_observations.append('The feature {0} has a right skewed distribution'.format(i))
            numerical_means.append(np.mean(data_cleaned[i]))
            numerical_stds.append(np.std(data_cleaned[i]))
            numerical_skew_type.append('Right Skewed')
            numerical_ids.append(i)

        elif np.mean(data_cleaned[i]) < np.std(data_cleaned[i]):
            numerical_observations.append('The feature {0} has a left skewed distribution'.format(i))
            numerical_means.append(np.mean(data_cleaned[i]))
            numerical_stds.append(np.std(data_cleaned[i]))
            numerical_skew_type.append('Left Skewed')
            numerical_ids.append(i)

    for j in categorical_columns_after_data_cleaning:
        categorical_observations.append('The highest freq recorded is {0}'.format(data_cleaned[j].mode()[0]))
        categorical_ids.append(j)

    df1 = pd.DataFrame({'Feature Name': numerical_ids,
                        'Observations': numerical_observations,
                        'Skewness': numerical_skew_type,
                        'Reported Std': numerical_stds,
                        'Reported Mean': numerical_means})

    df2 = pd.DataFrame({'Feature Name': categorical_ids,
                        'Observations': categorical_observations})

    return df1, df2

def groupby_agregation(data_cleaned, column_name_categorical, column_name, aggregartion_function):
    categorical_cols, numerical_cols = separate_categorical_and_numerical(data_cleaned)
    # print(column_name)
    # print(column_name_categorical)
    if column_name in categorical_cols:
        # aggregartion_function = ['count']
        if column_name != column_name_categorical:
            data_grouped = data_cleaned.groupby([column_name_categorical, column_name])[column_name].count()
            return data_grouped
        else:
            st.error('Both the Column names can not be same')
    else:
        if aggregartion_function == 'count':
            if column_name != column_name_categorical:
                data_grouped = data_cleaned.groupby([column_name_categorical, column_name])[column_name].count()
                return data_grouped
            else:
                st.error('Both the Column names can not be same')
        elif aggregartion_function == 'sum':
            if column_name != column_name_categorical:
                data_grouped = data_cleaned.groupby([column_name_categorical, column_name])[column_name].sum()
                return data_grouped
            else:
                st.error('Both the Column names can not be same')



def get_grouped_plot(data_cleaned,columns_name,coslumn_name_categorical):
    if column_name != column_name_categorical:
        plt.figure(figsize = [20,10])
        plt.title('Countplot for {0} grouped by {1}'.format(columns_name,column_name_categorical))
        fig =  sns.countplot(data = data_cleaned, y = columns_name , hue = column_name_categorical,orient = 'h')
        st.pyplot(plt)
        #return fig
    else:
        st.error('Both the Column names can not be same')

path = st.file_uploader("Choose a file")
flag =0
if path is not None:
    new_data = get_data(path)
    flag =1
else:
    st.info('Please give a valid csv data')



if flag ==1:
    categorical_columns, numerical_columns = separate_categorical_and_numerical(new_data)
    data_cleaned = data_cleaning(categorical_columns, numerical_columns, new_data)
    col1, col2, col3,col4,col5 , col6 = st.columns(6)
    button_flag = 0
    with col1:
        if st.button('Top 5 rows of the data'):
            button_flag = 1

    if button_flag == 1:
        st.dataframe(new_data.head())
        get_report(new_data.head())

    with col2:
        if st.button('Continuous Plots'):
            button_flag = 2

    if button_flag == 2:
        st.subheader('Plot for the numerical variables')
        plot_continious_data(data_cleaned)

    with col3:
        if st.button('Categorical Plots'):
            button_flag = 3

    if button_flag == 3:
            st.subheader('Get Plot for top 10 categories countwise')
            plot_categorical_data(data_cleaned)

    with col4:
        if st.button('Data Description'):
            button_flag = 4

    if button_flag == 4:
            st.subheader('Get Data Description')
            st.dataframe(new_data.describe())
            get_report(new_data.describe())


    with col5:
        if st.button('Data Information'):
            button_flag = 5

    if button_flag == 5:
            st.subheader('Get Data Information')
            var1 = get_info(new_data)
            st.dataframe(var1)
            get_report(var1)

    with col6:
        if st.button('Data Correlation'):
            button_flag = 6

    if button_flag == 6:
            st.subheader('Get Data Correlation')
            #st.pyplot(plt.matshow(new_data.corr()))
            fig, ax = plt.subplots()
            sns.set_style('white')
            sns.heatmap(new_data.corr(),annot = True, cmap= None, ax=ax)
            st.write(fig)

    #numerical_obs, categorical_obs = get_observations(categorical_columns, numerical_columns, new_data)


    # Calling the categorical and the numerical columns
    categorical_columns_after_data_cleaning = categorical_columns
    numerical_columns_after_data_cleaning = numerical_columns
    numerical_obs, categorical_obs = get_observations(categorical_columns_after_data_cleaning,
                                                      numerical_columns_after_data_cleaning, data_cleaned)

    buff_obs_1,col_get_obs_1, col_get_obs_2,buff_obs_2 = st.columns([8,6,6,8])
    flag_col_get_obs_1= 0
    flag_col_get_obs_2 = 0

    with col_get_obs_1:
        if st.button('Get Numerical Observations'):
            flag_col_get_obs_1 = 1
    if flag_col_get_obs_1 == 1:
            st.table(numerical_obs)
            get_report(numerical_obs)

    with col_get_obs_2:
        if st.button('Get Categorical Observations'):
            flag_col_get_obs_2 = 1
    if flag_col_get_obs_2 == 1:
            st.table(categorical_obs)
            get_report(categorical_obs)




    button_flag_for_group = 0
    #col_grouped_1 = st.columns(1)
    # with col_grouped_1:
    #     if st.button('Group Your Data'):
    #         button_flag_for_group = 1
    # if button_flag_for_group == 1:
    st.subheader('Group Your Data')
    column_name = st.selectbox('Please select the features you want to group the data with',
                               tuple(data_cleaned.columns.tolist()))
    #print(column_name)

    column_name_categorical = st.selectbox('Please select the features you want to apply aggregation on',
                                           tuple(data_cleaned.columns.tolist()))
    #print(column_name_categorical)
    aggregartion_function = st.selectbox('Select Aggregation',('count','sum'))

    #print(aggregartion_function)

    buff_gr_1 ,col_group_1 , col_group_2 , buff_gr_2  = st.columns([8,6,6,8])

    #col_group_1 = st.columns([1, 1])
    grouped_flag_1 = 0
    with col_group_1:
        if st.button('Apply Grpoupby'):
            grouped_flag_1 = 1
    if grouped_flag_1 ==1:
        grouped_data = groupby_agregation(data_cleaned, column_name_categorical, column_name, aggregartion_function)


        #print(grouped_data)
        if grouped_data is not None:
            index1 = [i[0] for i in pd.DataFrame(grouped_data).index]
            index2 = [i[1] for i in pd.DataFrame(grouped_data).index]
            values_df = pd.DataFrame(grouped_data).values.reshape(1, -1)[0]
            new_data_frame = pd.DataFrame({column_name_categorical: index1,
                                           column_name: index2,
                                           aggregartion_function: values_df})

            st.table(new_data_frame)
            get_report(new_data_frame)

    button_flag_grouped_plot = 0
    #buff1, col = st.columns([1, 5])
    #col_group_2 = st.columns([1, 2])
    with col_group_2:
        if st.button('Get Stacked Plot'):
            button_flag_grouped_plot = 1
    if button_flag_grouped_plot == 1:
        get_grouped_plot(data_cleaned,column_name,column_name_categorical)


    if (('Latitude' in new_data.columns) and ('Longitude' in new_data.columns)):
        # Map of the data
        buff_map_1, col_map_1, buff_map_2 = st.columns([10, 6, 8])
        button_flag_for_map = 0
        with col_map_1:
            if st.button('Get Your Map'):
                button_flag_for_map = 1
        if button_flag_for_map ==1:
            folium_static(plot_map_with_folium(new_data))
            #st.map(new_data)

    elif (('latitude' in new_data.columns) and ('longitude' in new_data.columns)):
        # Map of the data
        buff_map_1, col_map_1, buff_map_2 = st.columns([10, 6, 8])
        button_flag_for_map = 0
        with col_map_1:
            if st.button('Get Your Map'):
                button_flag_for_map = 1
        if button_flag_for_map ==1:
            folium_static(plot_map_with_folium(new_data))
            #st.map(new_data)

    elif (('lat' in new_data.columns) and ('long' in new_data.columns)):
        # Map of the data
        buff_map_1, col_map_1, buff_map_2 = st.columns([10, 6, 8])
        button_flag_for_map = 0
        with col_map_1:
            if st.button('Get Your Map'):
                button_flag_for_map = 1
        if button_flag_for_map ==1:
            folium_static(plot_map_with_folium(new_data))
            #st.map(new_data)

    elif (('Lat' in new_data.columns) and ('Long' in new_data.columns)):
        # Map of the data
        buff_map_1, col_map_1, buff_map_2 = st.columns([10, 6, 8])
        button_flag_for_map = 0
        with col_map_1:
            if st.button('Get Your Map'):
                button_flag_for_map = 1
        if button_flag_for_map ==1:
            folium_static(plot_map_with_folium(new_data))
            #st.map(new_data)

st.markdown(footer,unsafe_allow_html=True)


