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
x + y
# %%
v = np.array([9,10])
w = np.array([11, 12])
# %%
np.dot(w, v)
# %%
np.dot(x, v)
# %%
## Broadcasting 
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)
# %%
for i in range(4):
    y[i, :] = x[i, :] + v
# %%
y
# %%
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1)) 
y = x + vv  # Add x and vv elementwise

# %%
print(y) 
# %%
np.tile(v, (4, 1))
# %%
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)

# %%
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5]) 
# %%
print(np.reshape(v, (3, 1)) * w)

# %%
# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)
# %%# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(x + v)

# %%
# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((x.T + w).T)
# %%
print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)
# %%
