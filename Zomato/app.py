import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load and preprocess the data
dataframe = pd.read_csv("Zomato data .csv")

# Clean and preprocess the 'rate' column
def handleRate(value):
    value = str(value).split('/')[0]
    try:
        return float(value)
    except ValueError:
        return None

dataframe['rate'] = dataframe['rate'].apply(handleRate)

# Grouped data for listed_in(type) analysis
grouped_data = dataframe.groupby('listed_in(type)')['votes'].sum().reset_index()

# Heatmap pivot table
pivot_table = dataframe.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)

# Maximum votes restaurant
max_votes = dataframe['votes'].max()
restaurant_with_max_vote = dataframe.loc[dataframe['votes'] == max_votes, 'name'].values[0]

# Streamlit Dashboard
st.title("Zomato Data Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a section", [
    "Overview",
    "Online Orders Analysis",
    "Ratings Distribution",
    "Cost for Two Distribution",
    "Votes by Restaurant Type",
    "Heatmap",
    "Max Votes Restaurant",
    "Online vs Offline Ratings"
])

# Overview
if options == "Overview":
    st.header("Dataset Overview")
    st.write(dataframe.head(10))
    
   

# Online Orders Analysis
elif options == "Online Orders Analysis":
    st.header("Online Orders Distribution")
    fig = px.histogram(dataframe, x='online_order', title="Online Orders Distribution", color='online_order')
    st.plotly_chart(fig)

# Ratings Distribution
elif options == "Ratings Distribution":
    st.header("Ratings Distribution")
    fig = px.histogram(dataframe, x='rate', nbins=10, title="Ratings Distribution")
    st.plotly_chart(fig)

# Cost for Two Distribution
elif options == "Cost for Two Distribution":
    st.header("Cost for Two People Distribution")
    fig = px.histogram(dataframe, x='approx_cost(for two people)', title="Cost for Two Distribution", color_discrete_sequence=['blue'])
    st.plotly_chart(fig)

# Votes by Restaurant Type
elif options == "Votes by Restaurant Type":
    st.header("Votes by Restaurant Type")
    fig = px.bar(grouped_data, x='listed_in(type)', y='votes', title="Votes by Restaurant Type", color='votes')
    st.plotly_chart(fig)

# Heatmap
elif options == "Heatmap":
    st.header("Heatmap of Online Orders by Restaurant Type")
    fig = px.imshow(pivot_table, text_auto=True, title="Online Orders Heatmap", color_continuous_scale="YlGnBu")
    st.plotly_chart(fig)

# Max Votes Restaurant
elif options == "Max Votes Restaurant":
    st.header("Restaurant with Maximum Votes")
    st.write(f"The restaurant with the highest votes is: **{restaurant_with_max_vote}**")

# Online vs Offline Ratings
elif options == "Online vs Offline Ratings":
    st.header("Ratings for Online vs Offline Orders")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='online_order', y='rate', data=dataframe, ax=ax)
    ax.set_title("Ratings Comparison (Online vs Offline)")
    st.pyplot(fig)