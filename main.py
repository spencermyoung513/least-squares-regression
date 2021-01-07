import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
linApprox takes an input value / input array, t, representing time(s) in seconds, along with a coefficient matrix xArr and an observation matrix yArr
Returns v, the expected velocity (km/s) based on a least squares line of best fit (v can also be an array of values)
'''
def linApprox(t,xArr,yArr): 
	normal_coef = np.transpose(xArr) @ xArr	# X^T * X
	normal_vect = np.transpose(xArr) @ yArr	# X^T * Y
	beta = np.linalg.solve(normal_coef, normal_vect) 
	v_0 = beta[0]
	a = beta[1]
	v = v_0 + a*t
	return v

'''
curveApprox takes an input value / input array, t, representing time(s) in seconds, along with a coefficient matrix xArr and an observation matrix yArr
Returns d, the expected displacement (km) based on a least squares parabola of best fit (d can also be an array of values)
'''
def curveApprox(t, xArr, yArr):
	normal_coef = np.transpose(xArr) @ xArr  
	normal_vect = np.transpose(xArr) @ yArr
	beta = np.linalg.solve(normal_coef, normal_vect)
	v_0 = beta[0]
	a = beta[1]	
	d = v_0*t + a*t*t
	return d

'''
createPlot takes in a string 'type', indicating if it should create a plot for velocity or displacement, along with two arrays xArr and yArr with the respective data
Creates a .png file with the requested figure
'''
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

'''

BEGIN MAIN CODE HERE

'''

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

# Plot our results

createPlot("velocity",X1,Y1)
createPlot("displacement",X2,Y2)



