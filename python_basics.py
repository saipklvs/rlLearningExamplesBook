#%%
val_1 = list(range(20))
print(val_1[:-1])
# %%
print(val_1[5:7])
# %%
import numpy as np
# %%
rank_1 = np.array([1, 2, 3])
print(rank_1.shape)
# %%
b = np.array([[1, 2, 3], [4, 6, 9]])
print(b.shape)
# %%
b[0], b[1]
# %%
b[-1]
# %%
a = np.zeros((2, 2))
a
# %%
b = np.ones((1, 2))
# %%
b
# %%
c  =np.random.random((2, 2))
# %%
print(c)
# %%
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# %%
c  =np.random.random((2, 2, 2))

# %%
print(c)
# %%
np.full((2, 2), 7)
# %%
np.eye(5)
# %%
a
# %%
a[:2, 1:3]
# %%
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
row_1 = a[1, :]
row_2 = a[1:2, :]
# %%
print(row_1, row_1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_2, row_2.shape)
# %%
col_1 = a[:, 1]
col_2 = a[:, 1:2]
print(col_1, col_1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_2, col_2.shape)
# %%
import numpy as np
a = np.array([[1,2], [3, 4], [5, 6]])
# %%
a[[0, 1, 2], [0, 1, 1]][2]
# %%
import numpy as np

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
# %%
b = np.array([0, 2, 0, 1])
# %%
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# %%
a[np.arange(4), b] += 10

# %%
a
# %%
a = np.array([[1,2], [3, 4], [5, 6]])
b_i = a[a > 1]
# %%
b_i
# %%
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
# %%
