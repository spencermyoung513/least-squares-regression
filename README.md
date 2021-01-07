# Least-Squares-Regression
Short project modeling velocity/displacement data from a rocket launch with linear regression techniques.

### Notes:
- Data for this project was pulled from a school assignment.
- For more information about using normal equations to solve linear regression problems, [this video](https://www.youtube.com/results?search_query=linear+regression+with+normal+equation) provides a great explanation/visualization.

### Overview:
How can we model a dataset that appears to follow a mathematical pattern, but doesn't rigidly adhere to any one equation? This is a question that has long been explored by mathematicians and scientists, and there are some pretty cool methods that have been developed to achieve this outcome. The specific technique I have chosen to apply in this project is Least Squares Regression, using the normal equation of a system.

We will need the following libraries to properly solve this problem:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Goals: 
1. Using a dataset of time stamps along with **velocity** measurements, we want to find `v_0` and `a` that satisfy the well-known kinematics equation `v(t) = v_0 + at`, where our line v(t) is the best approximation of the actual data.
2. Using a separate dataset of time stamps and **displacement** measurements, we want to find `v_0` and `A` so that `d(t) = (v_0)(t) + At^2`, where `A = a/2`, or one half the acceleration, such that the curve `d(t)` is the best approximation of the actual data.

#### Velocity:

To find `v_0` and `a` such that `v(t) = v_0 + at`, we can take each separate time and velocity measurement from the dataset to form a matrix equation: 

![Initial Matrix Equation](https://github.com/spencermyoung513/Least-Squares-Regression/blob/main/Matrix%20Images/Matrices1.PNG)

With a little rewriting, this gives us

![Reworked Matrix Equation](https://github.com/spencermyoung513/Least-Squares-Regression/blob/main/Matrix%20Images/Matrices2.PNG)

Let `Y`, `X`, and `β` be defined as below:

![Matrix Equation Definitions](https://github.com/spencermyoung513/Least-Squares-Regression/blob/main/Matrix%20Images/Matrices3.PNG)

Then we have an equation to solve of the form `Y = Xβ`. Note that because of the nature of our raw data, there is not a single vector β that will satisfy this equation. To absolve this issue, we will instead solve the *normal equation of the system* to find a least squares regression line:

![Normal Equation](https://github.com/spencermyoung513/Least-Squares-Regression/blob/main/Matrix%20Images/NormalEq1.PNG)

We first need to process our data and load it into some numpy arrays:

```
# Building X1 (coefficient matrix for velocity data)

timeData = pd.read_csv("velocityData.txt", usecols=["time"]).values

X1 = np.zeros((timeData.size,2))
for i in range(X1.shape[0]):
	for j in range(X1.shape[1]):
		if j == 0:
			X1[i][j] = 1
		else:	
			X1[i][j] = timeData[i]

# Y1: observation vector. Contains all velocity measurements (km/s)

Y1 = pd.read_csv("velocityData.txt", usecols=["velocity"]).values
```

The following function solves for `β` using `xArr` (X1) and `yArr` (Y1), and then returns an approximated velocity given `t`, the time in seconds:

```
def linApprox(t,xArr,yArr): 
	normal_coef = np.transpose(xArr) @ xArr
	normal_vect = np.transpose(xArr) @ yArr
	beta = np.linalg.solve(normal_coef, normal_vect) 
	v_0 = beta[0]
	a = beta[1]
	v = v_0 + a*t
	return v
```

Now, all that is left to do is plot the data along with our regression line using the function call `createPlot("velocity",X1,Y1)`:

```
def createPlot(type,xArr,yArr):
	fig = plt.figure()
	graph = fig.add_subplot()
	
	if type == "velocity":	
		t = [row[1] for row in xArr]
		fit = linApprox(t,xArr,yArr)
		plt.title("Velocity vs. Time")
		plt.ylabel("Velocity (km/s)")

	elif type == "displacement":
		t = [row[0] for row in xArr]
		fit = curveApprox(t,xArr,yArr)
		plt.title("Displacement vs. Time")
		plt.ylabel("Displacement (km)")

	else:
		print("Invalid graph type specified")
		return
	
	graph.scatter(t, yArr)
	graph.plot(t,fit)
	plt.xlabel("Time (seconds)")
	fig.savefig(type+"Graph.png")
 ```
This yields the following figure, which seems to be a pretty good fit:

![Velocity Plot](https://github.com/spencermyoung513/Least-Squares-Regression/blob/main/velocityGraph.png)

#### Displacement:

The procedure for solving for displacement is similar to velocity, except we are searching for different weights. We need `v_0` and `A` so that `d(t) = (v_0)(t) + At^2`, where `A = a/2`.

We first form a matrix equation:

![Initial Matrix Equation](https://github.com/spencermyoung513/Least-Squares-Regression/blob/main/Matrix%20Images/Matrices4.PNG)

We can rewrite this as

![Reworked Matrix Equation](https://github.com/spencermyoung513/Least-Squares-Regression/blob/main/Matrix%20Images/Matrices5.PNG)

Defining our matrices Y, X, and β yields

![Reworked Matrix Equation](https://github.com/spencermyoung513/Least-Squares-Regression/blob/main/Matrix%20Images/Matrices6.PNG)

We cannot find a perfect vector `β` to satisfy `Y = Xβ`, so we must again use the normal equation 

![Normal Equation](https://github.com/spencermyoung513/Least-Squares-Regression/blob/main/Matrix%20Images/NormalEq1.PNG)

We build our coefficient matrix (X) and observation vector (Y):

```
# Building X2 (coefficient matrix for displacement data)

X2 = np.zeros((timeData.size,2))
for i in range(X2.shape[0]):
	for j in range(X2.shape[1]):
		if j == 0:
			X2[i][j] = timeData[i]
		else:
			X2[i][j] = timeData[i] ** 2

# Y2: observation vector. Contains all displacement measurements (km)

Y2 = pd.read_csv("displacementData.txt", usecols=["displacement"]).values
```
The following function solves for `β` using `xArr` (X2) and `yArr` (Y2), and then returns an approximate displacement given `t`, the time in seconds:

```
def curveApprox(t, xArr, yArr):
	normal_coef = np.transpose(xArr) @ xArr  
	normal_vect = np.transpose(xArr) @ yArr
	beta = np.linalg.solve(normal_coef, normal_vect)
	v_0 = beta[0]
	a = beta[1]	
	d = v_0*t + a*t*t
	return d
```

Finally, we can plot our displacement data with the regression curve, this time calling `createPlot("displacement",X2,Y2)`. Our result:

![Displacement Plot](https://github.com/spencermyoung513/Least-Squares-Regression/blob/main/displacementGraph.png)
