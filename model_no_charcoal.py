import pandas as pd
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from random import random, seed
import pickle
import statsmodels.api as sm

df = pd.read_csv('titration_data.csv')

x = df[['vol_acid', 'vol_water']]
y = df[['no_charcoal','with_charcoal']]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

#model = svm.SVR(kernel='linear')
model = LinearRegression()
model.fit(x_train, y_train)


model_filename = 'titration.model'
pickle.dump(model, open(model_filename, 'wb'))


# Plot

X = df[['vol_acid', 'vol_water']]
y = df['no_charcoal']

X = sm.add_constant(X)
est = sm.OLS(y,X).fit()
xx1, xx2 = np.meshgrid(np.linspace(X.vol_acid.min(), X.vol_acid.max(), 100), 
                       np.linspace(X.vol_water.min(), X.vol_water.max(), 100))
Z = est.params[0] + est.params[1] * xx1 + est.params[2] * xx2
fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, azim=-115, elev=15)
surf = ax.plot_surface(xx1, xx2, Z, alpha=0.6, linewidth=0)
resid = y - est.predict(X)
ax.scatter(X[resid >= 0].vol_acid, X[resid >= 0].vol_water, y[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X[resid < 0].vol_acid, X[resid < 0].vol_water, y[resid < 0], color='black', alpha=1.0)

# set axis labels
ax.set_xlabel('Vol Acid')
ax.set_ylabel('Vol Water')
ax.set_zlabel('Titrant (No Charcoal)')

#print(est.summary())
#plt.show()

print(model.intercept_[0])
print(model.coef_[0])
print(model.coef_[1])