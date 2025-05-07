from pprint import PrettyPrinter

from pymongo import MongoClient
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pytz
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

pp = PrettyPrinter(indent=2)

#Connecting database to localhost by creating client
client = MongoClient(host="localhost", port=27017)

#Printing list of database available on client
pp.pprint(list(client.list_databases()))

#Assigning "air-quality" database
db = client["air-quality"]

#Printing collection available on db
for c in db.list_collections():
    print(c["name"])

#Assigning the "nairobi" collection in db
nairobi = db["nairobi"]
nairobi.count_documents({})

result = nairobi.find_one({})
pp.pprint(result)

#Finding sensor sites 
nairobi.distinct("metadata.site")

print("Documents from site 6:", nairobi.count_documents({"metadata.site" : 6}))
print("Documents from site 29:", nairobi.count_documents({"metadata.site" : 29}))

#Finding readings in each site
result = nairobi.aggregate(
    [
        {
             {"$group" : {"_id": "$metadata.site", "count":{"$count":{}}}}
        }
    ]
)

#Finding types of measurements
nairobi.distinct("metadata.measurement")

#Retriving PM2.5 reading from all sites
result = nairobi.find({"metada.measurement":"P2"}).limit(3)
pp.pprint(list(result))

#Calculating how many readings are for each type
result = nairobi.aggregate(
    [
        {"$match" :{"metadata.site":6}},
        {"$group" : {"_id": "$metadata.measurement", "count":{"$count":{}}}}
    ]

)
pp.pprint(list(result))

#Calculating how many readings are in site 29
result = nairobi.aggregate(
    [
        {"$match" :{"metadata.site":29}},
        {"$group" : {"_id": "$metadata.measurement", "count":{"$count":{}}}}
    ]

)
pp.pprint(list(result))

#Retriving the PM2.5 reading from site 29
result = nairobi.find(
    {"metadata.site":29, "metadata.measurement":"P2"},
    projection={"P2":1, "timestamp":1, "_id":0}
)

#Reading records from result into dataframe
df = pd.DataFrame(result).set_index("timestamp")
df.head()

#Creating wrangle function
def wrangle(collection):
    results = collection.find(
        {"metadata.site":29,"metadata.measurement":"P2"},
        projection={"P2":1,"timestamp":1,"_id":0}
    )
    df = pd.DataFrame(results).set_index("timestamp")

    #Localizing timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    #Removing outliers
    df = df[df["P2"]<500]

    #Resampling to 1hr window ffill missing values
    df=df["P2"].resample("1H").mean().fillna(method="ffill").to_frame()

    #Adding lag features and removing columns
    df["P2.L1"]=df["P2"].shift(1)
    df.dropna(inplace=True)

    return df

#Reading data collection in DataFrame
df = wrangle(nairobi)
print(df.shape)
df.head()

#Creating box-plot of P2
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(kind="box", vert = False, title="Distribution of PM2.5 Readings", ax=ax)

#Creating time-series plot of P2
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(xlabel="Time", ylabel="PM2.5",title="PM2.5 timeseries",ax=ax);

#Creating lineplot of P2
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].rolling(168).mean().plot(ax=ax,ylabel="PM2.5",title="Weekly Rolling Average")

#Correlation matrix
df.corr()

#Creating Scatter plot for P2
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x=df["P2.L1"], y=df["P2"])
ax.plot([0,120],[0,120], linestyle="--", color="red")
plt.xlabel("P2.L1")
plt.ylabel("P2")
plt.title("PM2.5 AutoCorrelation");

#Splitting dataframe into feature matrix and target vector
target = "P2"
y = df [target]
X = df.drop(columns=target)

#Splitting into training and test set
cutoff = int(len(X) * 0.8)
X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]

#Calculating  baseline mean absolute error
y_pred_baseline = [(y_train.mean())] * len(y_train)
mae_baseline = mean_absolute_error(y_train,y_pred_baseline)
print("Mean P2 Reading:", round(y_train.mean(), 2))
print("Baseline MAE:", round(mae_baseline, 2))

#Institaing a LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Calculating training and test mean absolute error
training_mae = mean_absolute_error(y_train, model.predict(X_train))
test_mae =  mean_absolute_error(y_test, model.predict(X_test))
print("Training MAE:", round(training_mae, 2))
print("Test MAE:", round(test_mae, 2))

#Extracting intercept and coefficient
intercept = round(model.intercept_,2)
coefficient = round(model.coef_[0],2)
print(f"P2 = {intercept} + ({coefficient} * P2.L1)")

#Creating Dataframe
df_pred_test = pd.DataFrame(
    {
        "y_test":y_test ,
        "y_pred":model.predict(X_test)
    }
)
df_pred_test.head()

#Creating time-series line plot for prediction
fig = px.line(df_pred_test, labels={"value":"P2"})
fig.show()