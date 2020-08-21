#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data[:, 1:3]
y = iris.target

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.5, random_state= 42)


# In[14]:


neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(train_X, train_y)
    train_accuracy[i] = knn.score(train_X, train_y)
    test_accuracy[i] = knn.score(test_X, test_y)


# In[15]:


plt.plot(neighbors, test_accuracy * 100, label= "Test Accuracy")
plt.style.use("ggplot")
plt.xlabel("k Value")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy and k Value Relation\nfor Iris Type Detection")
plt.legend()
#plt.savefig("Accuracy and k Value Relation\nfor Iris Type Detection.pdf", bbox_inches= "tight")


# In[5]:


cmap_light = ListedColormap(['orange', 'Turquoise', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

x_min, x_max = test_X[:, 0].min() - .5, test_X[:, 0].max() + .5
y_min, y_max = test_X[:, 1].min() - .5, test_X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

plt.scatter(test_X[:, 0], test_X[:, 1], c= test_y, cmap= cmap_bold)
plt.style.use("ggplot")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Types (Setosa, Versicolour, and Virginica): Sepal Width and Length")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
#plt.savefig("Iris Types (Setosa, Versicolour, and Virginica): Sepal Width and Length.pdf", bbox_inches= "tight")


# In[6]:


h = 0.2

for i in neighbors:
    aknn = KNeighborsClassifier(n_neighbors = i)
    aknn.fit(train_X, train_y)
    predicted = aknn.predict(test_X)
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = aknn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.style.use("seaborn-deep")
    plt.pcolormesh(xx, yy, Z, cmap= cmap_light, alpha= 0.4)
    
    plt.scatter(test_X[:, 0], test_X[:, 1], c=predicted, cmap= cmap_bold, edgecolor= "k", s= 20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Nearest Neighbors Classifier:\nIris Type Detection (k =" + str(i) + ")")
    #plt.savefig("Nearest Neighbors Classifier:\nIris Type Detection (k =" + str(i) + ").pdf", bbox_inches= "tight")

