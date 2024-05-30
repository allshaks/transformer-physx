# U_avg_transpose * data matrix = sigma* V_avg_transpose

# Steps:
## 1. Load artificially created data set or original data trials if available
## 2. Calculate SVD (U matrix, s vector and V matrix) for averaged data
## 3. Take first 3 elements of transposed U matrix and multiply with each trial
##  -> this results in a 3x369 dimensional matrix for each trial
## 4. Calculate standard deviation in x, y and z direction 
##  -> how exactly?
##  -> idea: for each time points (369) we want to have 4 values, each has 3 coordinates (?)
##      -> one average value
##      -> std in x direction
##      -> std in y direction
##      -> std in z direction
# 5. Plot final result