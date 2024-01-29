import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('xt_model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    Type= int(request.form['Type'])
    Category_Name= int(request.form['Category Name'])
    Customer_Country= int(request.form['Customer Country'])
    Customer_Segment= int(request.form['Customer Segment'])
    Customer_State= int(request.form['Customer State'])
    Department_Name= int(request.form['Department Name'])
    Market= int(request.form['Market'])
    Order_Country= int(request.form['Order Country'])
    Order_Item_Quantity= int(request.form['Order Item Quantity'])
    Order_Region= int(request.form['Order Region'])
    Product_Price= int(request.form['Product Price'])
    Shipping_Mode= int(request.form['Shipping Mode'])

    # Date_of_Order
    date_order = request.form["date_order"]
    Order_date= int(pd.to_datetime(date_order, format="%Y-%m-%dT%H:%M").day)
    order_month= int(pd.to_datetime(date_order, format ="%Y-%m-%dT%H:%M").month)
    order_year= int(pd.to_datetime(date_order, format ="%Y-%m-%dT%H:%M").year)
    # print("Order Date : ",Date, Month,Year)

    # Time
    order_hour = int(pd.to_datetime(date_order, format ="%Y-%m-%dT%H:%M").hour)
    order_day_of_the_week = int(pd.to_datetime(date_order, format ="%Y-%m-%dT%H:%M").dayofweek)
        # print("Time : ",Hour)
    
    
    is_weekend= int(request.form['is weekend'])
    
    # Create a DataFrame with the user's input
    input_data = pd.DataFrame({

        'Type':[Type],
        'Category Name':[Category_Name],
        'Customer Country':[Customer_Country],
        'Customer Segment':[Customer_Segment],
        'Customer State':[Customer_State],
        'Department Name':[Department_Name],
        'Market':[Market],
        'Order Country':[Order_Country],
        'Order Item Quantity':[Order_Item_Quantity],
        'Order Region':[Order_Region],
        'Product Price':[Product_Price],
        'Shipping Mode':[Shipping_Mode],
        'order year':[order_year],
        'order month':[order_month],
        'Order date':[Order_date],
        'order day of the week':[order_day_of_the_week],
        'is weekend':[is_weekend],
        'order hour':[order_hour] 
    })
        
    # Make the prediction
    prediction = model.predict(input_data)[0]

    return render_template('home.html',prediction=prediction)


    
if __name__ == '__main__':
    app.run(debug=True)