import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# Generate 'random' data
np.random.seed(0)
X = 2.5 * np.random.randn(100) + 1.5   # Array of 100 values with mean = 1.5, stddev = 2.5
eq = 0.5 * np.random.randn(100)       # Generate 100 residual terms
y = 2 + 0.3 * X + eq                  # Actual values of Y
# Create pandas dataframe of the random data to store our X and y values
df = pd.DataFrame(
    {'X': X,
     'y': y}
)

# Show the first five rows of our dataframe
df.head()


# Calculate the mean of X and y
xmean = np.mean(X)
ymean = np.mean(y)

# Calculate the terms needed for the numator and denominator of beta
df['xycov'] = (df['X'] - xmean) * (df['y'] - ymean)
df['xvar'] = (df['X'] - xmean)**2

# Calculate beta and alpha
B = df['xycov'].sum() / df['xvar'].sum()
A = ymean - (B * xmean)
print(f'alpha(A) = {A}')
print(f'beta(B) = {B}')

#Y = 2.003 + 0.323 X- From answer

regLine = A + B*X

#Visulaizations
plt.plot(X, regLine)     # regression line
plt.scatter(X, y, 'red')   # scatter plot showing actual data
plt.xlabel('X')
plt.ylabel('y')
plt.grid('True')
plt.show()
