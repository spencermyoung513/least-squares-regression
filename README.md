# Least-Squares-Regression
Short project modeling velocity/displacement data from a rocket launch with linear regression techniques.

### Notes:
- Data for this project was pulled from a school assignment.
- For more information about using normal equations to solve linear regression problems, [this video](https://www.youtube.com/results?search_query=linear+regression+with+normal+equation) provides a great explanation/visualization.

### Overview:
How can we model a dataset that appears to follow a mathematical pattern, but doesn't rigidly adhere to any one equation? This is a question that has long been explored by mathematicians and scientists, and there are some pretty cool methods that have been developed to achieve this outcome. The specific technique I have chosen to apply in this project is Least Squares Regression, using the normal equation of a system.

### Goals: 
1. Using a dataset of time stamps along with velocity measurements, we want to find `v_0` and `a` that satisfy the well-known kinematics equation `v(t) = v_0 + at`, where our line v(t) is the best approximation of the actual data.
2. Using a separate dataset of time stamps and displacement measurements, we want to find `v_0` and `A` so that `d(t) = (v_0)(t) + At^2`, where `A = a/2`, or one half the acceleration, such that the curve `d(t)` is the best approximation of the actual data.
