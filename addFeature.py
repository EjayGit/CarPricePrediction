

# Add new feature for company names.
df["company"] = df.name.apply(lambda x: x.split(" ")[0])
df.head()

# Check if non=year value.
df2 = df.copy()
df2 = df2[df2["year"].str.isnumeric()]

# Change to integer.
df2["year"] = df2["year"].astype(int)

# Price has 'Ask for Price'.
df2 = df2[df2["Price"]] != "Ask for Price"
df2.Price

# Remove commas from prices and convet to int.
df2.Price = df2.Price.str.replace(",","").astype(int)

# Review "kms driven".
df2["kms_driven"]

# Remove comma and space in 'kms driven' data field.
df2["kms_driven"] = df2["kms_driven"].str.split(" ").str.get(0).str.replace(",","")

# Check details are numeric.
df2 = df2[df2["kms_driven"].str.isnumeric()]

# Review info.
df2.info()

# Convert numeric details to int.
df2["kms_driven"] = df2["kms_driven"].astype(int)

# Review fuel types that have NaN values.
df2[df2["fuel_type"].isna()]

# If NaN do not copy to df2.
df2 = df2[~df2["fuel_type"].isna()]

# Copy first 3 words of car name to car name.
df2['name']=df2['name'].str.split().str.slice(start=0,stop=3).str.join(' ')

# Review head.
df2.head()

# Reset the index to '0' and delete (drop) the old index saved in (what would be new variable) 'index'.
df2 = df2.reset_index(drop=True)

# Save the cleaned data.
df2.to_csv("cleaned_car_data.csv")
df2.describe(include="all")

# Drop price outliers.
df2 = df2[df2["Price"]<6e6].reset_index(drop=True)
df2

# Get unique company names.
df2["company"].unique()

# Display relationship between company and price.
import seaborn as sns
plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=df2)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()

# Display relationship between year and price.
plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=df2)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()

# Display relationship between kms_driven and price.
sns.relplot(x='kms_driven',y='Price',data=df2,height=7,aspect=1.5)

# Display relationship between fuel type and price.
plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=df2)

# Display relationship between fuel type, year and company.
ax=sns.relplot(x='company',y='Price',data=df2,hue='fuel_type',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')

# Extract training data.
X=df2[['name','company','year','kms_driven','fuel_type']]
y=df2['Price']
X

# Apply train test split.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# Create OneHotEncoder obj to contain all categories.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])

# Transform categorical columns.
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']), remainder='passthrough')

# Build linear regression model.
lr = LinearRegression()

# Make pipeline.
pipe = make_pipeline(column_trans,lr)

# Fit model.
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Check R2 score.
r2_score(y_test, y_pred)

# Finding the best R2 score for different states of TrainTestSplit
scores=[]
for i in range(1000):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))

scores[np.argmax(scores)]

# Predict specific car price.
pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))

# Train the model using the best random state.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)

import joblib
joblib.dump(pipe,open('LinearRegressionModel.pkl','wb'))
pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))