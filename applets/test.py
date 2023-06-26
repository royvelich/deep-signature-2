# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
#
# def generate_surface(k1, k2, n_points=100):
#     x = np.linspace(-1, 1, n_points)
#     y = np.linspace(-1, 1, n_points)
#
#     x, y = np.meshgrid(x, y)
#     z = k1 * x ** 2 / 2 + k2 * y ** 2 / 2
#
#     return x, y, z
#
#
# # Test the function
# k1 = -1.0
# k2 = -1.0
# x, y, z = generate_surface(k1, k2)
#
# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z, cmap='viridis')
# plt.show()


import numpy as np

# Define original tensor of shape (K, 3) with K=12
K = 9
tensor = np.arange(K * 3).reshape(K, 3)

# Reshape to a 3D tensor with shape (2, 2, 6)
reshaped_tensor = tensor.reshape(3, 3, 3)

# Say you have an index in the original tensor
original_index = (4, 1)

# Convert this to the index in the reshaped tensor
# We only consider the first two dimensions
reshaped_index = np.unravel_index(original_index[0], reshaped_tensor.shape[:-1])

print(f"The index {original_index} in the original tensor corresponds to the index {reshaped_index} in the reshaped tensor")

# And you can do the reverse as well
reshaped_index = (1, 1)

# Convert this to the index in the original tensor
# We only consider the first two dimensions
original_index = np.ravel_multi_index(reshaped_index, reshaped_tensor.shape[:-1])

print(f"The index {reshaped_index} in the reshaped tensor corresponds to the index {original_index} in the original tensor")
