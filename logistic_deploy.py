import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from PIL  import Image

#img = Image.open(r"C:\Users\Admin\Downloads\images.jpg")
#st.image(img, width=200,align="Center")

st.title("Logistic Regression")
st.sidebar.header("User input Parameters")


def user_inputs():
	pclass = st.sidebar.selectbox("Pclass",('1','2','3'))
	age = st.sidebar.number_input("Enter the Age")
	sibsp = st.sidebar.number_input("Type the number of siblings")
	parch = st.sidebar.number_input("Type the number of Parents/Children")
	fare = st.sidebar.number_input("Type the Fare")
	sex = st.sidebar.selectbox("Sex",('1','0'))
	st.sidebar.subheader("Embarked")
	embarked_q = st.sidebar.selectbox("Queenstown",('1','0'))
	embarked_s = st.sidebar.selectbox("Southampton",('1','0'))
	
	
	data = {
		
		"Pclass" : pclass,
		"Age" : age,
		"SibSp" : sibsp,
		"Parch" : parch,
		"Fare" : fare,
		"Sex_male" : sex,
		"Embarked_Q" : embarked_q,
		"Embarked_S" : embarked_s
		}
	features = pd.DataFrame(data,index=[0])
	return features



df = user_inputs()
st.subheader("Selected Options")
st.write(df)


test = pd.read_csv(r"C:\Users\Admin\Downloads\Logistic Regression (1)\Logistic Regression\Titanic_test.csv")
train = pd.read_csv(r"C:\Users\Admin\Downloads\Logistic Regression (1)\Logistic Regression\Titanic_train.csv")

### Removing the unwanted columns ### 

cols = ["PassengerId","Name","Cabin","Ticket"]
train.drop(cols, axis=1,inplace=True)
test.drop(cols, axis=1, inplace=True)

### Imputation of missing data ###

train['Age'] = train['Age'].fillna(train['Age'].median())
test["Age"] = test['Age'].fillna(train['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

### Encoding ###

train = pd.get_dummies(train,dtype=int,drop_first=True)
test = pd.get_dummies(test,dtype=int,drop_first=True)

x = train.iloc[:,1:]
y = train.iloc[:,0]
X_train,X_val,Y_train,Y_val = train_test_split(x,y,test_size=0.2,random_state=7)
log_reg = LogisticRegression(max_iter=1000)
model = log_reg.fit(X_train,Y_train)


yhat = model.predict(df)
prediction_probabilities = model.predict_proba(df)

st.subheader("Predicted Results")
st.write("The person **survived**" if prediction_probabilities[0][1] > 0.5 else "The person **did not survive**")

st.subheader("Prediction Probability")
st.write(prediction_probabilities)



