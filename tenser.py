import pandas as pd
from tensorflow.keras.models import Sequential # Importing the Sequential model
from tensorflow.keras.layers import Dense # Importing the Dense layer
from tensorflow.keras.models import load_model # Importing the load_model function 
# from io import StringIO
# import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('Churn.csv')


X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def train_and_evaluate_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=10):
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    eval = model.evaluate(X_test, y_test);
    print('The model\'s loss is: ', eval[0])


    y_hat = model.predict(X_test)
    y_hat = [1 if y > 0.5 else 0 for y in y_hat]

    print('The model\'s accuracy is: ', accuracy_score(y_test, y_hat))

    number_of_churns = sum(y_hat)
    print(f'Number of customers predicted to churn: {number_of_churns}')

    return model

model = train_and_evaluate_model(X_train, y_train, X_test, y_test)

model.save('model.h5')

del model

model = load_model('model.h5')

model = train_and_evaluate_model(X_train, y_train, X_test, y_test, epochs=5, batch_size=10)
