import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# the file has no names for columns.
df = pd.read_csv('agro.data')
# print(df.head())

# we check the names of the columns on the on name file.
df.columns = ["entidad y municipio", "UP", "STH", "CAA", "CAAH", "SAA", "SAAH"]
# print(df.head())

# use variable name to change properties of the plot
x=df["CAAH"]
y=df["SAAH"]
z=df["UP"]
# print(x.head())

# imprimir en 2D
# plt.scatter(x='UP', y='CAA',  data=df)
# plt.scatter(x, z)
# plt.show()

# imprimir en 3D
# ax = plt.axes(projection='3d')
# ax.scatter3D(x, y, z)
# plt.show()

# imprimir en 3D interactivo
fig = plt.figure()
# ///ax = plt.axes(projection='3d')
ax = fig.add_subplot(projection='3d')
ax.view_init(0, 90)
ax.scatter(x, y, z)
plt.show()

# tools is no longer in pandas, now you can call pd.plotting directly
#plt.ma(df)

