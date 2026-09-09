import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival chance in the Titanic Journey")
pclass = st.slider('Enter the passenger class for the user', 1, 3)
sex = st.selectbox('Enter the Passenger gender', ['male', 'female'])
sibsp = st.slider('Enter the passenger sibling and spouse', 1, 8)
parch = st.slider('Enter the passenger total no of parents and child for the user', 0, 6)
fare=st.number_input("Enter the Fare of The passenger")
embarked=st.selectbox('Enter the Passenger station from where they started the journey', ['Southampton', 'Chebourg', 'Queenstown'])

data = pd.DataFrame([{'Pclass':pclass, 'Sex':sex, 'SibSp':sibsp, 'Parch':parch, 'Fare':fare, 'Embarked':embarked}])

model = load_model('model.h5')

with open('label_encoder.pkl', 'rb') as file:
    label = pickle.load(file)

with open('onehotencoder.pkl', 'rb') as file:
    onehot = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

data['Sex'] = label.transform(data['Sex'])
embarked = onehot.transform(data[['Embarked']])
embarked = pd.DataFrame(embarked, columns=onehot.get_feature_names_out())
data = pd.concat([data.drop(columns=['Embarked']), embarked], axis=1)
data[['Pclass', 'SibSp', 'Parch', 'Fare']] = scaler.transform(data[['Pclass', 'SibSp', 'Parch', 'Fare']])

y = model.predict(data)
y[0][0]

def chance(y):
    if y>0.5:
        st.write("The passenger will survive the journey")
    else:
        st.write("The passenger will survive the journey")

if st.button('predict survival chance'):
    st.write('Probability of Passenger Survival chance', y)
    st.write(chance(y))