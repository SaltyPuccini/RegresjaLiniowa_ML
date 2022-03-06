import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

auto = pd.read_csv(r'Q:\MachineLearning\Dane\auto-mpg\auto-mpg.csv')
print(auto.head())

X = auto.iloc[:, 1:-1]
X = X.drop(columns=['horsepower'])
y = auto.loc[:, 'mpg']

print(X.head())
print(y.head())

lr = LinearRegression()

lr.fit(X, y)

print(lr.score(X, y))

my_car1 = [4, 160, 1906, 12, 90, 1]
my_car2 = [4, 200, 2609, 15, 83, 1]

cars = [my_car1, my_car2]
mpg_predict = lr.predict(cars)
print(mpg_predict)