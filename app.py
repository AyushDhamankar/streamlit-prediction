import streamlit as st 
import numpy as np
from graphqlclient import GraphQLClient
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math

client = GraphQLClient('https://api.studio.thegraph.com/query/51089/real-estate-polygon/v0.1')
# Define your GraphQL query
query = '''
query {
    registeredProperties {
        location
        total_sqft
        bath
        price
        bhk
    }
}
'''

# Execute the query
result = client.execute(query)

# Parse the JSON response
data = json.loads(result)

# Extract the "tokens" array
tokens_array = data["data"]["registeredProperties"]

#load data into a DataFrame object:
df = pd.DataFrame(tokens_array)

df10 = pd.read_csv('bhp-final.csv')

result = pd.concat([df, df10])

st.title("streamlit Forms & Submit Demo")
location = st.selectbox('Select location', result['location'].unique(), key=0)
sqft = st.slider("Enter your age", min_value=500, max_value=5000)
bhk =  st.selectbox('Select flavor', ['1', '2', '3', '4', '5'], key=1)
bath =  st.selectbox('Select bath', ['1', '2', '3', '4', '5'], key=2)


dummies = pd.get_dummies(result.location)
df11 = pd.concat([result, dummies.drop('Other', axis='columns')], axis='columns')
df12 = df11.drop('location', axis='columns')

X = df12.drop(['price'], axis='columns')
y = df12.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)

# Check if the location column exists
if location in X.columns:
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
        result = str(math.floor(lr_clf.predict([x])[0]))
else:
    result =  "Location not found in the dataset"

# Define a function to be executed when the button is clicked
def on_button_click():
    st.title(result+ " Lakhs")

# Create a button with the label "Click me!"
button_clicked = st.button("Click me!", type="primary")

# Check if the button is clicked
if button_clicked:
    # If the button is clicked, execute the on_button_click function
    on_button_click()