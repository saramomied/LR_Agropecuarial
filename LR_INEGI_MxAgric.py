import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
from array import *
import statistics

__errors__ = []  # global variable to store the errors/loss for visualisation
__yhyp__ = []
__tyhyp__ = []
__testyhyp__ = []
__errorsT__ = []
__avg__ = 0.0
__maxValue__ = 0.0
__avgT__ = 0.0
__maxValueT__ = 0.0


def h(tetha, data_instances_train): # it calculates a singles value for y(hypothesis)
    acum = 0
    for i in range(len(tetha)):
        acum = acum + tetha[i] * data_instances_train[i]
    return acum

def show_errors(params, samples, y): # Using a cost function technique the global variable errors include model behavior and update
    global __errors__
    error_acum = 0
    y_float = np.array(y, dtype=np.float32)
    for i in range(len(samples)):
        hyp = h(params, samples[i])
        error = hyp - y[i]
        error_acum = +error ** 2
    mean_error_param = error_acum / len(samples)
    __errors__.append(mean_error_param)

def show_errors_Testing(params, samples, y): # Same cost fuction but for testing part
    global __errorsT__
    error_acum = 0
    for i in range(len(samples)):
        hyp = h(params, samples[i])
        error = hyp - y[i]
        error_acum = +error ** 2
    mean_error_param = error_acum / len(samples)
    __errorsT__.append(mean_error_param)

def GD(tetha, data_instances_train, data_target_train, alfa):#Gradient descent implementarion
    temp = list(tetha)
    data_target_train_float = np.array(data_target_train, dtype=np.float32)
    for j in range(len(tetha)):
        acum = 0
        for i in range(len(data_instances_train)):
            error = h(tetha, data_instances_train[i]) - data_target_train_float[i] #Error between real train value and hypothesis
            acum = acum + error * data_instances_train[i][j]  # Sumatory fron GD
        temp[j] = tetha[j] - alfa * (1 / len(data_instances_train)) * acum  # Application of learning rate to change parameters theta
    return temp

def scaling_Data(dataSamples):
    global __avg__ # Calling global variabe avg to use in R2 Score determination
    global __maxValue__# Calling global variabe maxValue to use in R2 Score determination
    acum = 0
    samples = numpy.asarray(dataSamples).T.tolist()
    for i in range(0, len(samples)):
        for j in range(len(samples[i])):
            acum = + samples[i][j]  # Sumatory of al sample values
        __avg__ = acum / (len(samples[i]))  # Updating avg value
        __maxValue__ = max(samples[i]) # Updating maxValue
        for j in range(len(samples[i])):
            samples[i][j] = (samples[i][j] - __avg__) / __maxValue__ # Defining new scaled sample value
    return numpy.asarray(samples).T.tolist()

def Final_y_app(tetha, data_instances_train):
    global __tyhyp__ # Variable to get final target variable hypotesis values using training
    for i in range(len(data_instances_train)):
        __tyhyp__.append(h(tetha, data_instances_train[i]))

def Final_y_app_Test(tetha, data_instances_test):
    global __testyhyp__ # Variable to get final target variable hypotesis values of the test DB using final theta parameters
    for i in range(len(data_instances_test)):
        __testyhyp__.append(h(tetha, data_instances_test[i]))

def mean(data): # Implementation of a handing mean fuction, this bc I had a problem with statistics lib
    sum = 0.0
    data = np.array(data, dtype=np.float32)
    for i in range(len(data)):
        sum = sum + data[i]
    avg = sum / len(data)
    return avg


# importing the data
df = pd.read_csv('agro.data')

# Assigning columns to represent the information
# Important highlites for this Linear Regression:
# UP => Units of production (tons) [target variable]
# STH => Total Surface of the region (hectares)
# CAAH => Surface that has agricultural production (hectares)
# SAAH => Surface without agricultural production but it has the potencial to have it (hectares)
df.columns = ["entidad y municipio", "UP", "STH", "CAA", "CAAH", "SAA", "SAAH"]

# Delimitating the amount of data for this LR (1000 instances - 80% training & 20% testing)
rdf = df[:1000]
instances = rdf.drop(['entidad y municipio', 'UP', 'CAA', 'SAA'], axis=1)
target = rdf.drop(['entidad y municipio', 'STH', 'CAA', 'CAAH', 'SAA', 'SAAH'], axis=1)
# instances['bias'] = 1 ############ BIAS was considered 0 ###############

# Scaling my data
instances = scaling_Data(instances)
target = scaling_Data(target)

# Separating data in training & testing sets
# "X" values
data_instances_train = instances[:-200]
data_instances_test = instances[-200:]
# "Y" values
data_target_train = target[:-200]
data_target_test = target[-200:]

# Defining parameters for each independent variable
tetha = [1, 1, 1]
alfa = 0.3
ie = 0

while True:
    oldparams = list(tetha)
    print(data_instances_train)
    print(tetha)
    tetha = GD(tetha, data_instances_train, data_target_train, alfa) #Updates values for theta
    show_errors(tetha, data_instances_train, data_target_train) #Gets the error from new values
    ie = ie + 1
    if (oldparams == tetha or ie == 1000):# No. of epochs 1000
        break

# Plotting the evolution of the error in the Training Set
plt.plot(__errors__)
plt.show()

# Showing the final values for tetha (1, 2, 3) of the training DataSet
print("Final tetha values:")
tethaValues = np.array(tetha)
print(tethaValues)

# //////////////////////////////////////////////////////////////////////////////
# Calculating R2 Score of Training
ymean = mean(data_target_train)
SST = (((data_target_train - ymean) ** 2).sum()) / 1000000
error_float = np.array(__errors__, dtype=np.float32)
SSE = (((error_float) ** 2).sum()) * 1000000
R2_T = (SST - SSE) / SST
print("////////////////////////////")
print("R2_Training:")
print(R2_T)

ie2 = 0
while True:
    show_errors_Testing(tetha, data_instances_test, data_target_test) # Getting error from theta parameters in testing DB
    ie2 = ie2 + 1
    if (ie2 == 1000):
        break
# Calculating R2 Score of Testing
ymean2 = mean(data_target_test)
SST2 = (((data_target_test - ymean2) ** 2).sum()) / 100000
error_float2 = np.array(__errorsT__, dtype=np.float32)
SSE2 = (((error_float2) ** 2).sum()) * 100000
R2_T2 = (SST2 - SSE2) / SST2
print("////////////////////////////")
print("R2_Testing:")
print(R2_T2)

# ///////////////////////////////////////////////////////////////////////////////////
# Getting final array of target variable ("UP") using final tetha values for training
Final_y_app(tetha, data_instances_train)
x = list(range(1, 801))
plt.plot(x, __tyhyp__, alpha=0.5)
plt.plot(x, data_target_train, alpha=0.5)
plt.legend(["Hypotesis Approach", "Real Training Values"])
plt.show()

# Getting final array of target variable ("UP") using final tetha values for testing
Final_y_app_Test(tetha, data_instances_test)
x2 = list(range(1, 201))
plt.plot(x2, __testyhyp__, alpha=0.5)
plt.plot(x2, data_target_test, alpha=0.5)
plt.legend(["Hypotesis Approach", "Real Testing Values"])
plt.show()

while True:
# ////////////////////////////////////////////////////////////////////////////////////
# USER Input
    U_Information = [0.0, 0.0, 0.0]
    print("////////////////////////////")
    print("Write 3 physical characteristics of the agricultural field in municipality")
    U_Information[0] = float(input("Total Surface (Hectares)= "))
    U_Information[1] = float(input("Surface with Agricultural Activities (Hectares)= "))
    U_Information[2] = float(input("Surface WITHOUT Agricultural Activities (Hectares)= "))
    if U_Information[0]==0 and U_Information[1]==0 and U_Information[2]==0:
        break
    Y_User = (tethaValues[0] * U_Information[0]) + (tethaValues[1] * U_Information[1]) + (tethaValues[2] * U_Information[2])
    print("Estimated Units of Production in Tones are:")
    print(float(Y_User))
