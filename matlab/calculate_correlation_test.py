import numpy

import numpy as np
from matplotlib import pyplot as plt


def calculate_corr(f, g):
    # Calculate the means of f and g
    mean_f = np.mean(f)
    mean_g = np.mean(g)

    # Calculate the variances of f and g
    var_f = np.var(f)
    var_g = np.var(g)

    # Calculate the covariance between f and g
    cov_fg = np.mean((f - mean_f) * (g - mean_g))

    # Calculate the correlation coefficient
    corr = cov_fg / np.sqrt(var_f * var_g)

    return corr

# Example usage
# Create two sets of data as tensors
f = np.random.random((1000,))
g = np.random.random((1000,))

# Calculate the correlation coefficient
correlation = calculate_corr(f, g)
print("Correlation coefficient:", correlation)

# Define the functions k1, k2, dk1_1, dk1_2, dk2_1, dk2_2, dk2_11, and dk1_22
def k1(X, Y):
    return X/2 + Y/2 + (((X + Y)**2)/4 - X*Y)**(1/2)



def k2(X, Y):
    return X/2 + Y/2 - (((X + Y)**2)/4 - X*Y)**(1/2)



def dk1_1(X, Y):
    return X*0
    # return (X*Y)/2 + (X*Y*(X + Y))/(4*((X + Y)**2/4 - X*Y)**(1/2))


def dk1_2(X, Y):
    return X*0


def dk2_1(X, Y):
    return X*0
    # return (X*Y)/2 - (X*Y*(X + Y))/(4*((X + Y)**2/4 - X*Y)**(1/2))

def dk2_2(X, Y):
    return X*0

def dk1_22(X, Y):
    return X*Y**2 - (3*Y**2*(X + Y))/2 + (4*X*Y**3 - (3*Y**2*(X + Y)**2)/2 + X*Y**2*(X + Y))/(2*((X + Y)**2/4 - X*Y)**(1/2))

def dk2_11(X, Y):
    return X**2*Y - (3*X**2*(X + Y))/2 - (4*X**3*Y - (3*X**2*(X + Y)**2)/2 + X**2*Y*(X + Y))/(2*((X + Y)**2/4 - X*Y)**(1/2))



# Put the functions in a list
functions = [k1, k2, dk1_1, dk1_2, dk2_1, dk2_2, dk2_11, dk1_22]

# Define the values of X and Y
X = np.random.random((1000,))
Y = np.random.random((1000,))

# Calculate correlations and create the correlation matrix
num_functions = len(functions)
# correlation_matrix = np.zeros((num_functions, num_functions))
# for i in range(num_functions):
#     for j in range(i, num_functions):
#         corr = calculate_corr(functions[i](X, Y), functions[j](X, Y))
#         correlation_matrix[i, j] = corr
#         correlation_matrix[j, i] = corr


functions_applyed_on_samples = np.array([functions[i](X,Y) for i in range(num_functions)])
correlation_matrix = np.corrcoef(functions_applyed_on_samples)

# Plot the correlation matrix
plt.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(num_functions), ['k1', 'k2', 'dk1_1', 'dk1_2', 'dk2_1', 'dk2_2', 'dk2_11', 'dk1_22'], rotation=45)
plt.yticks(range(num_functions), ['k1', 'k2', 'dk1_1', 'dk1_2', 'dk2_1', 'dk2_2', 'dk2_11', 'dk1_22'])
plt.title('Correlation Matrix')
plt.show()
