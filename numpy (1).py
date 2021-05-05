#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[7]:


get_ipython().run_line_magic('pinfo', 'np')


# In[16]:


import array
l = list(range(10))
a = array.array('i',l)    #'i'indicate the content is integer
a


# In[20]:


import numpy as np
np.array([1,2,3,4])


# In[21]:


np.array([1.25,2,3,4])


# In[22]:


np.array([1.25,2,3,4],dtype='float32')


# In[23]:


np.array([range(i, i + 3) for i in [2, 4, 6]])   #The inner lists are treated as 
                                                #rows of the resulting two-dimensional array


# In[24]:


np.zeros(10, dtype=int)


# In[33]:


#np.ones((3, 5) , dtype = int)
#np.ones((3, 5) , dtype = float)
np.ones((3, 5) , dtype = str)


# In[30]:


np.full((3, 5), 3.14)


# In[36]:


np.arange(0, 20, 2)


# In[37]:


np.linspace(0, 1, 5)


# In[44]:


np.random.random((3, 3))


# In[47]:


np.random.normal(0, 1, (3, 3))


# In[49]:


np.random.randint(0, 10, (3, 3))


# In[53]:


np.eye(3)


# In[54]:


np.empty(3)


# In[59]:


np.random.seed(0)
x1 = np.random.randint(10, size=6) # One-dimensional array
x2 = np.random.randint(10, size=(3, 4)) # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5)) # Three-di


# In[60]:


print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)


# In[61]:


print("dtype:", x3.dtype)
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")


# In[62]:


x1#1D


# In[63]:


x2   #2D


# In[75]:


print(x2[0,0] )
print(x2[0,-1] )


# In[77]:


x2[0,0]=25
x2


# In[96]:


print(x2[:, 0]) #first colum of x2 or print(x2[0])


# In[97]:


print(x2[0, :]) #first row of x2


# In[98]:


x2


# In[99]:


x2_sub = x2[:2, :2]
print(x2_sub)


# In[101]:


x2_sub[0,0] = 99
x2_sub


# In[102]:


x2


# In[104]:


x2_sub_copy = x2[:2,:2].copy()
print(x2_sub_copy)


# In[105]:


x2_sub_copy[0,0] = 42  #42 is not updated in copy func
x2_sub_copy


# In[106]:


x2


# In[112]:


grid = np.arange(1, 13).reshape((4, 3))
print(grid)


# In[113]:


x = np.array([1, 2, 3])
 # row vector via reshape
x.reshape((1, 3))


# In[114]:


x[np.newaxis, :]


# In[115]:


x.reshape((3, 1))#column


# In[116]:


x[:, np.newaxis]


# In[117]:


x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])


# In[121]:


z = [99, 99, 99]
np.concatenate([x, y, z])


# In[125]:



grid = np.array([[1, 2, 3],
                  [4, 5, 6]])
np.concatenate([grid, grid])


# In[126]:


np.concatenate([grid, grid], axis=1)


# In[128]:


x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
 # vertically stack the arrays
np.vstack([x, grid])


# In[129]:


y = np.array([[99],
             [99]])
np.hstack([grid, y])


# In[132]:


x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)


# In[133]:


grid = np.arange(16).reshape((4, 4))
grid


# In[137]:


upper, lower = np.vsplit(grid, [2])  #np.vsplit
print(upper)
print(lower)


# In[135]:


left, right = np.hsplit(grid, [2])    #np.hsplit
print(left)
print(right)


# In[138]:


x = np.arange(9).reshape((3, 3))
x


# In[139]:


2**x


# In[140]:


x = np.arange(4)
print("x =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2) 


# In[141]:


print("-x = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2 = ", x % 2)


# In[142]:


-(0.5*x + 1) ** 2


# In[143]:


np.add(x, 2)


# In[145]:


x = np.array([-2, -1, 0, 1, 2]) #abs = absolute
abs(x)


# In[146]:


x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
np.abs(x)


# In[148]:


theta = np.linspace(0, np.pi, 3)


# In[149]:


print("theta = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))


# In[150]:


x = [-1, 0, 1]
print("x = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))


# In[151]:


x = [1, 2, 3]
print("x =", x)
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("3^x =", np.power(3, x))


# In[152]:


x = [1, 2, 4, 10]
print("x =", x)
print("ln(x) =", np.log(x))
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))


# In[154]:


from scipy import special
# Gamma functions (generalized factorials) and related functions
x = [1, 5, 10]
print("gamma(x) =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2) =", special.beta(x, 2))


# In[155]:


# Error function (integral of Gaussian)
# its complement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x) =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))


# In[4]:


x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)


# In[5]:


y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)


# In[11]:


x = np.arange(1, 6)
np.add.reduce(x)


# In[12]:


np.multiply.reduce(x)


# In[13]:


x = np.arange(1, 6)
np.add.accumulate(x)


# In[14]:


np.multiply.accumulate(x)


# In[15]:


x = np.arange(1, 6)
np.multiply.outer(x, x)


# In[24]:


L = np.random.random(100)
sum(L)


# In[25]:


np.sum(L)


# In[26]:


big_array = np.random.rand(1000000)
get_ipython().run_line_magic('timeit', 'sum(big_array)')
get_ipython().run_line_magic('timeit', 'np.sum(big_array)')


# In[27]:


min(big_array), max(big_array)


# In[28]:


np.min(big_array), np.max(big_array)


# In[29]:


print(big_array.min(), big_array.max(), big_array.sum())


# In[30]:


M = np.random.random((3, 4))
print(M)


# In[31]:


M.sum()


# In[34]:


M.min(axis = 0)


# In[35]:


M.max(axis=1)


# In[42]:


heights =np.arange(30,71)
print(heights)
print("Mean height: ", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height: ", heights.min())
print("Maximum height: ", heights.max())


# In[43]:


print("25th percentile: ", np.percentile(heights, 25))
print("Median: ", np.median(heights))
print("75th percentile: ", np.percentile(heights, 75))


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # set plot style


# In[48]:


plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number')


# In[51]:


a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
b


# In[52]:


a+b


# In[54]:


M = np.ones((2, 3)) #2d
a = np.arange(3)#1d
print(M)
print(a)


# In[56]:


M+a


# In[62]:


X = np.random.random((10, 3))
X


# In[63]:


Xmean = X.mean(0)
Xmean


# In[65]:


X_centered = X - Xmean
X_centered


# In[66]:


X_centered.mean(0)


# In[69]:


# x and y have 50 steps from 0 to 5
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


# In[70]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[71]:


plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],cmap='viridis')
plt.colorbar();


# In[79]:


x = np.random.randint(10, size=(3, 4))
x


# In[82]:


# how many values less than 6?
np.count_nonzero(x < 6)


# In[83]:


np.sum(x < 6)


# In[84]:


# how many values less than 6 in each row?
np.sum(x < 6, axis=1)


# In[85]:


# are there any values greater than 8?
np.any(x > 8)


# In[86]:


# are there any values less than zero?
np.any(x < 0)


# In[87]:


# are all values less than 10?
np.all(x < 10)


# In[88]:


# are all values equal to 6?
np.all(x == 6)


# In[89]:


# are all values in each row less than 8?
np.all(x < 8, axis=1)


# In[93]:


import numpy as np  #sorting of array
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x


# In[94]:


x = np.array([2, 1, 4, 3, 5])
selection_sort(x)


# In[95]:


def bogosort(x):
    while np.any(x[:-1] > x[1:]):
        np.random.shuffle(x)
    return x


# In[96]:


x = np.array([2, 1, 4, 3, 5])
bogosort(x)


# In[97]:


x = np.array([2, 1, 4, 3, 5])
np.sort(x)


# In[100]:


x.sort()
x


# In[101]:


x = np.array([2, 1, 4, 3, 5])  #argsort sort the index of the elem
i = np.argsort(x)
print(i)


# In[102]:


x = np.array([7, 2, 3, 1, 6, 5, 4])   #partial sort
np.partition(x, 3) 


# In[103]:


name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
x = np.zeros(4, dtype=int)
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
 'formats':('U10', 'i4', 'f8')})
print(data.dtype)


# In[105]:


data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)


# In[ ]:




