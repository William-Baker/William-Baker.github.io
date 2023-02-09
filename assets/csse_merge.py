
# This is an interactive python file, it can be run by installing vscode and python, then running the cells

#%%
!python -m pip install pandas
!python -m pip install tqdm

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

#%%


path = """/home/w/Documents/MSDV/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/"""

from distutils.command.clean import clean
from os import listdir
from os.path import isfile, join

# Get all the file names for all the CSV files
files = [f for f in listdir(path) if isfile(join(path, f))]
csvs = [f for f in files if ".csv" in f]

# Read all the CSV files and merge them into a single large pandas data frame
dfs = [pd.read_csv(path + c) for c in csvs]
df = pd.concat(dfs)



#%%

# There are duplicated columns, in this section, we 'unify' these duplicates taking whichever attribute is not NA
def unify(e):
    ret = pd.NA
    for col in e.index:
        if e[col] != "<NA>" and not pd.isna(e[col]):
            return e[col]
    return pd.NA

clean_df = pd.DataFrame()
clean_df['Province'] = df[['Province/State', 'Province_State']].progress_apply(unify, axis=1)
clean_df['Country'] = df[['Country/Region', 'Country_Region']].progress_apply(unify, axis=1)
clean_df['lat'] = df[['Lat', 'Latitude']].progress_apply(unify, axis=1)
clean_df['lng'] = df[['Long_', 'Longitude']].progress_apply(unify, axis=1)
clean_df['Last Update'] = df[['Last Update', 'Last_Update']].progress_apply(unify, axis=1)
clean_df['Case Fatality Ratio'] = df[['Case-Fatality_Ratio', 'Case_Fatality_Ratio']].progress_apply(unify, axis=1)
clean_df['Incidence Rate'] = df[['Incidence_Rate','Incident_Rate']].progress_apply(unify, axis=1)

common = ['Confirmed', 'Deaths', 'Recovered', 'FIPS', 'Admin2',  'Active', 'Combined_Key']
clean_df[common] = df[common]

#%%


clean_df.dropna(subset=['lat', 'lng'], inplace=True)
# To increase performance and normalise the density of the heatmap, round coordinates to the nearest integer
clean_df['lat'] = clean_df['lat'].astype(int)
clean_df['lng'] = clean_df['lng'].astype(int)

# Fill any NA values with 0, so all frequencies are numeric
clean_df[['Confirmed', 'Deaths', 'Recovered', 'Active']] = clean_df[['Confirmed', 'Deaths', 'Recovered', 'Active']].fillna(0)

#%%

# function to de-accumulate the cumulative sum over the whole dataset, so we can view each datapoint in isolation
def df_sort_decumulate(df):
    df = pd.DataFrame(df.sort_values('Last Update', ascending=True))
    for col in ['Confirmed', 'Deaths', 'Recovered', 'Active']:
        l = df[col].to_list()
        n = [0] * len(l)
        for i in range(1, len(l)):
            n[i] = l[i] - l[i-1]
        df[col] = n
    return df

# The data has been accumulated over time, this isnt ideal for the time series graph
grouped = clean_df.groupby('Combined_Key').progress_apply(lambda df: df_sort_decumulate(df)) # The function is applied to each table of grouped values, 
        # then pandas automatically concatenates the result, back into one big table like what we started with

# ensure the date attribute is correctly interpreted
grouped['Last Update'] = pd.to_datetime(grouped['Last Update'])

for col in ['Confirmed', 'Deaths', 'Recovered', 'Active']:
    grouped[col] = grouped[col].apply(lambda x: 0 if x < 0 else x)

#%%

# Group by location and date, on a weekly frequency
monthly_freq = grouped.groupby(['lat', 'lng', pd.Grouper(key='Last Update', freq='7D')]).sum()
monthly_freq = monthly_freq.reset_index()

#%%

# Define functions to export the data into a JSON, using correct ISO date formatting
def series_to_json(series):
    if series.dtype == '<M8[ns]':
        lst = str(series.apply(lambda x: x.isoformat()).to_list())
        lst = lst.replace('\'', '\"')
    else:
        lst = str(series.to_list())
    return f"\"{series.name}\": {lst}"

def df_to_json(df):
    json_str = "{"
    for col in df.columns:
        json_str += "\n" + series_to_json(df[col])
        json_str += ","
    json_str = json_str.removesuffix(',')  
    json_str += "\n}"
    return json_str

# Index the data by location and export each location to JSON
json_by_lat_lng = monthly_freq.groupby(['lat', 'lng']).apply(lambda df: df_to_json(df[['Last Update', 'Confirmed', 'Deaths', 'Recovered', 'Active']]))

#%%

# Reset the index so we can access it
json_by_lat_lng_df = pd.DataFrame(json_by_lat_lng, columns=['data'])
json_by_lat_lng_df = json_by_lat_lng_df.reset_index()

# Iterate through each location and add the location data to the json - unify each location into a single JSON file
json_str = "["
for index, row in json_by_lat_lng_df.iterrows():
    json_str += "{\n" f"\"lat\": {row['lat']},\n\"lng\": {row['lng']},\n\"data\": {row['data']}" + "},\n"
json_str = json_str.removesuffix(",\n")
json_str += "\n]"

#%%

with open("global_covid.json", "w") as text_file:
    text_file.write(json_str)
text_file.close()

# %%
