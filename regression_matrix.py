import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
data = np.array([
	[0.05, 0.12],
	[0.18, 0.22],
	[0.31, 0.35],
	[0.42, 0.38],
	[0.5, 0.49],
	])
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
# linear least squares
#Formula: B=(X.T*X)inv * X.Ty
b = inv(X.T.dot(X)).dot(X.T).dot(y)
print(b)
# predict using coefficients
yhat = X.dot(b)
# plot data and predictions
plt.scatter(X, y)
plt.plot(X, yhat, 'red')
plt.show()