import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

#load data from csv file
data = pd.read_csv('life_expectancy_ph.csv')

#get target X and Y columns
X = data['Year'].values
Y = data['Life expectancy'].values

mean_X = np.nanmean(X)
mean_Y = np.nanmean(Y)

X_minus_mean = X - mean_X
Y_minus_mean = Y - mean_Y

#compute for slope
m = np.sum((X_minus_mean * Y_minus_mean)) / np.sum(np.square(X_minus_mean))

#compute for c
c = mean_Y - (m*mean_X)

print(f"Mean of X = {mean_X}")
print(f"Mean of Y = {mean_Y}")
print(f"Slope = {m}")
print(f"C = {c}")

p_X = np.array([])
p_Y = np.array([])

#get predicted y value
for x in X:
    p_X = np.append(p_X, [x])
    y = (m*x) + c
    p_Y = np.append(p_Y, y)

#check fit, compute R squared
p_Y_minus_mean = p_Y - mean_Y
R = np.sum(np.square(p_Y_minus_mean))/np.sum(np.square(Y_minus_mean))
print(f"R squared value = {R}")

#plot data
plt.plot(X, Y, 'bo', color='red')
plt.plot(p_X, p_Y, color='green')
plt.show()