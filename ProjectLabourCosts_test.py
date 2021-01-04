import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from pivottablejs import pivot_ui
import streamlit.components.v1 as components
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
import warnings 

# Define all data
input = pd.read_csv('./tsInput.csv')
ref1 = pd.read_csv('./project_Ls.csv')
ref2 = pd.read_csv('./staff_Lvl_Gp.csv')
ref3 = pd.read_csv('./salary_Lvl.csv')
data = './data.csv'

# Merging master-input file with reference-file
input_ref1 = pd.merge(input, ref1, on='PROJECT', how='left')
input_ref1_2 = pd.merge(input_ref1, ref2, on='STAFF', how='left')
input_ref1_2_3 = pd.merge(input_ref1_2, ref3, on='LEVEL', how='left')

# Adding new calculated column calculated to the merge-table
input_ref1_2_3['SalaryCost'] = input_ref1_2_3['HOUR'] * input_ref1_2_3['AVER HOURLY SALARY']

# Output the .csv data file to a specific location
outputTable = pd.DataFrame(input_ref1_2_3)
outputTable.to_csv(data)

# Load the processed output data into streamlit as data source
DATA_URL = ("./data.csv")
@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data
data = load_data()

# Setting up streamlit by giving titles and description
st.write('###### REMARK: The analysis on this web app is based on mock datasets. This site is used by the owner as a means to practice and illustrate the skills in python for data analysis and visualization.')
st.title('Project labour costs')
st.subheader('Scenario')
st.write('A company named Color-Veg Pte. Ltd. would like to study the labour cost on projects. A timesheet system was created to collect time spent enter by each employee, grouped by functional group, for each project.')
st.markdown('Employee number: 84')
st.markdown('Functional group: 19')
st.markdown('Project number: 33')
st.markdown('Data collection period: 1st Aug 2019 - 3rd May 2020')
st.subheader('View of data for selected month')

st.sidebar.title('Interactive control sidebar')
st.sidebar.subheader('View of data for selected month')

# Graph 1: labour costs in S$ and Hr by Project/Functional Group
selectYr = st.sidebar.selectbox('Year', ['2019', '2020'], key='1')
if selectYr == '2019':
    selectMth = st.sidebar.selectbox('Month', ['8', '9', '10', '11', '12'], key='1')
else: 
    selectMth = st.sidebar.selectbox('Month', ['1', '2', '3', '4', '5'], key='1')

selectGraph = data.query('(YEAR == @selectYr) & (MONTH == @selectMth)') 
selectGraph
select = st.sidebar.selectbox('Sort by:', ['Functional Group', 'Project'], key='1')
if select == 'Functional Group':
    trace0 = go.Bar(x=selectGraph["GROUP"], y=selectGraph["SalaryCost"], name='S$', xaxis='x', yaxis='y', offsetgroup=1)
    trace1 = go.Bar(x=selectGraph["GROUP"], y=selectGraph["HOUR"], name='Hr', yaxis='y2', offsetgroup=2)
    dataTrace = [trace0, trace1]
    layoutTrace={'xaxis': {'title': 'Functional Group'},'yaxis': {'title': 'Salary Cost (S$)'}, 'yaxis2': {'title': 'Time Spent (Hr)', 'overlaying': 'y', 'side': 'right'}, 'height':550}
    fig = go.Figure(data=dataTrace, layout=layoutTrace)
    st.plotly_chart(fig)
    
    # Graph 2b: Ranking of labour cost for project/Functional Group by S$ & Hr
    selectGraph2 = px.bar(selectGraph, x='SalaryCost', y='GROUP', color=None, facet_row=None, category_orders={}, labels={})
    selectGraph2.update_yaxes(categoryorder='sum ascending')
    st.plotly_chart(selectGraph2)
   
    selectGraph2 = px.bar(selectGraph, x='HOUR', y='GROUP', color=None, facet_row=None, category_orders={}, labels={})
    selectGraph2.update_yaxes(categoryorder='sum ascending')
    st.plotly_chart(selectGraph2)
else:
    trace0 = go.Bar(x=[selectGraph["FIELD"],selectGraph["PROJECT"]], y=selectGraph["SalaryCost"], name='S$', xaxis='x', yaxis='y', offsetgroup=1)
    trace1 = go.Bar(x=[selectGraph["FIELD"],selectGraph["PROJECT"]], y=selectGraph["HOUR"], name='Hr', yaxis='y2', offsetgroup=2)
    dataTrace = [trace0, trace1]
    layoutTrace={'xaxis': {'title': 'Project'},'yaxis': {'title': 'Salary Cost (S$)'}, 'yaxis2': {'title': 'Time Spent (Hr)', 'overlaying': 'y', 'side': 'right'}, 'height':550}
    fig = go.Figure(data=dataTrace, layout=layoutTrace)
    st.plotly_chart(fig)

    # Graph 2a: Ranking of labour cost for project/Functional Group by S$ & Hr
    selectGraph2 = px.bar(selectGraph, x='SalaryCost', y='PROJECT', color='GROUP', facet_row=None, category_orders={}, labels={})
    selectGraph2.update_yaxes(categoryorder='sum ascending')
    st.plotly_chart(selectGraph2)
   
    selectGraph2 = px.bar(selectGraph, x='HOUR', y='PROJECT', color='GROUP', facet_row=None, category_orders={}, labels={})
    selectGraph2.update_yaxes(categoryorder='sum ascending')
    st.plotly_chart(selectGraph2)


# Hide/Show data table for the selected month
if st.sidebar.checkbox("Show data table", False):
    st.markdown("#### Data table for the selected time period")
    selectGraph = data.query('(YEAR == @selectYr) & (MONTH == @selectMth)')
    selectGraph = selectGraph.drop(['Unnamed: 0', 'WEEKNUMBER', 'CREATED_AT', 'LEVEL', 'AVER HOURLY SALARY'], axis=1)
    selectGraph


# Appendices: pivot table and forecasting data
st.subheader('Appendices')

# Hide/Show pivot table for raw data
st.sidebar.subheader("Appendices")
if st.sidebar.checkbox("Show pivot table for the raw data", True):
    st.markdown("#### Pivot table for the raw data")
    dataPv = data.drop(['AVER HOURLY SALARY'], axis=1)
    pvTable = pivot_ui(dataPv)
    with open(pvTable.src) as t:
        components.html(t.read(), height=400, scrolling=True)


# Below is the machine learning time series forecasting for time enteries
dataPrd = data
dataPrd = pd.pivot_table(dataPrd, index=['CREATED_AT'], values='HOUR', aggfunc='sum', margins=True)
dataPrd = dataPrd[:-1]
dataPrd = pd.DataFrame(dataPrd.to_records())

dates = list(dataPrd['CREATED_AT'])
dates = list(pd.to_datetime(dates))
Hr = list(dataPrd['HOUR'])

dataset = pd.DataFrame(columns=['ds', 'y'])
dataset['ds'] = dates
dataset['y'] = Hr
dataset = dataset.set_index('ds')

index = pd.date_range(start=dataset.index.min(), end=dataset.index.max(), freq='D')
dataset = dataset.reindex(index)
dataset = dataset.loc['2019-08-01':'2020-05-03']
dataset['y'] = dataset['y'].fillna(0)

start_date ='2020-02-15'
train = dataset.loc[dataset.index < pd.to_datetime(start_date)]
test = dataset.loc[dataset.index >= pd.to_datetime(start_date)]
model = SARIMAX(train, order=(3, 0, 7))
results = model.fit(disp=True)

sarimax_prediction = results.predict(start='2020-02-15', end='2020-05-03', dynamic=False)
sarimax_prediction = pd.DataFrame(sarimax_prediction)

trace1 = {
    "name": "Observation",
    'mode': 'lines',
    'type': 'scatter',
    'x': dataset.index,
    'y': dataset['y']
}
trace2 = {
    'name': 'Prediction',
    'mode': 'lines',
    'type': 'scatter',
    'x': sarimax_prediction.index,
    'y': sarimax_prediction['predicted_mean']
}
data = [trace1, trace2]
layout={
    "title": 'Method: SARIMAX',
    "xaxis": {'type': 'date', "title": "Dates", 'autorange': True},
    "yaxis": {"title": "Time entered (Hr)"},
    'autosize': True
}
fig = Figure(data=data, layout=layout)

# Hide/show forecasting chart
if st.sidebar.checkbox("Show forecasting for time entries", True):
    st.markdown("#### Time series prediction using SARIMAX")
    st.plotly_chart(fig)

# Print the SARIMA MAE and model summary
st.write('SARIMA MAE = ', mean_absolute_error(sarimax_prediction, test))

@st.cache(allow_output_mutation=True)
def load_model():
    model = pm.auto_arima(train, start_p=0, start_q=0, test='adf', max_p=7, max_q=7, m=1, d=None, seasonal=False, start_P=0, D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    return model
model = load_model()
st.text(model.summary())
