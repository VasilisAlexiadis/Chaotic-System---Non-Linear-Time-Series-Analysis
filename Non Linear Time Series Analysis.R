#################################################################################################
# Load the required libraries
#################################################################################################

install.packages("entropy")
library(entropy)
install.packages("nonlinearTseries")
current_directory <- getwd()
library(nonlinearTseries)
install.packages("pracma")
install.packages("ReaCTran")
library(pracma)  
library(ReacTran)  
install.packages("FNN")
library(FNN)
install.packages("tseriesChaos")
library(tseriesChaos)
install.packages("DChaos")
library(DChaos)
install.packages("tsfknn")
library(tsfknn)
install.packages("rgl")
library(rgl)
library(plot3Drgl)

#################################################################################################
# Loading data
#################################################################################################

# Load the data w8.txt file
x <- scan("C:/Users/User/Documents/w8.txt")
# Convert the data to a time series object
x <- ts(x)
# Plot the time series
plot(x, type = "l")

#################################################################################################
# PART 1: Calculate the time delay for properly embedding the corresponding system attractor.
#################################################################################################


# Estimate time delay employing the ACF method
timeLag1<-timeLag(
  x,
  technique = c("acf"),
  selection.method = c("first.e.decay"), 
  value = 1/exp(1),
  lag.max = NULL,
  do.plot = TRUE,
  main = NULL,
  
)

timeLag1


# Estimate time delay employing the first local minimum of Mutual Information
timeLag2<-timeLag(
  x,
  technique = c("ami"),
  selection.method = c("first.minimum"), 
  value = 1/exp(1),
  lag.max = NULL,
  do.plot = TRUE,
  main = NULL,
  
)

timeLag2

#################################################################################################
# PART 2: Calculate the correlation integrals and estimate the minimum embedding dimension
# and the fractal dimension for this system. 
#################################################################################################


# Estimate the correlation integrals
cd <- corrDim(x,
              min.embedding.dim = 1,
              max.embedding.dim = 5,
              time.lag = timeLag2,
              min.radius = 0.001, max.radius = 50,
              n.points.radius = 50,
              theiler.window=100,number.boxes=100,
              do.plot = FALSE)

# Plot the correlation dimension
plot(cd)
# Second method to Estimate the correlation integrals using tseriesChaos package
m=6
d=8
t=4
eps.min=2
d2(x, m, d, t, eps.min, neps=100)
# Estimate embedding (essential) dimension using time delay 
emb_dim <- estimateEmbeddingDim(x, time.lag = timeLag2, max.embedding.dim = 15)
emb_dim

# Fractal dimension calculation
cd.est <- estimate(cd, regression.range = c(0.72, 3 ), use.embeddings = (1):(5))
cat("Estimate: ", cd.est, "\n")

#################################################################################################
# PART 3: Calculate local Lyapunov spectrum and write the values of the Lyapunov exponents.
#################################################################################################

# Computing the maximal Lyapunov Exponent (using nonlinearseries routines)

ml = maxLyapunov(x, 
                 sampling.period=1,
                 min.embedding.dim = 2,
                 max.embedding.dim = 5 ,
                 time.lag = timeLag2,
                 theiler.window = 4,
                 radius=1,
                 max.time.steps=1000,
                 do.plot=T)
plot(ml,type="l", xlim = c(0,8))
ml.est = estimate(ml, regression.range = c(0,15),use.embeddings=2:5,
                  do.plot = T,type="l")
cat("estimate: ", ml.est,"\n")



#################################################################################################


# Estimate the Lyapunov exponent spectrum using the DChaos R package

LyapSpectrum<-lyapunov(
  x,
  m = 2:5,
  lag = 8,
  timelapse = c("FIXED"),
  h = 2:10,
  w0maxit = 100,
  wtsmaxit = 1e+06,
  pre.white = TRUE,
  lyapmethod = c("SLE"),
  blocking = c("EQS"),
  B = 1000,
  trace = 1,
  seed.t = TRUE,
  seed = 56666459,
  doplot = TRUE
)
summary(LyapSpectrum)

# Estimate the Maximum Lyapunov exponent using the DChaos R package

LyapMax<-lyapunov(
  x,
  m = 2:5,
  lag = 8,
  timelapse = c("FIXED"),
  h = 2:10,
  w0maxit = 100,
  wtsmaxit = 1e+06,
  pre.white = TRUE,
  lyapmethod = c("LLE"),
  blocking = c("EQS"),
  B = 1000,
  trace = 1,
  seed.t = TRUE,
  seed = 56666459,
  doplot = TRUE
)
summary(LyapMax)


#################################################################################################

# Second method of estimating the Maximum Lyapunov Exponent
data <- x  # Your time-series data
m <- 2:5   # Embedding dimension
lag <- 8  # Time lag
timelapse <- "FIXED"  # Timelapse option
h <- 2:10  # Range of values for other parameters

# Compute the partial derivatives using jacobian.net
jacob <- jacobian.net(data = data, m = m, lag = lag, timelapse = timelapse, h = h)

LyapMax1<-lyapunov.max(
  data= jacob,
  blocking = c( "EQS"),
  B = 1000,
  doplot = TRUE
)

summary(LyapMax1)

#################################################################################################
# PART 4: What is the value of the Kolmogorov-Sinai entropy?
#################################################################################################

# Calculate the sample entropy
se <- sampleEntropy(cd, do.plot = TRUE)
se.est = estimate(se, do.plot = F,
                  regression.range = c(8,15))
cat("Sample entropy estimate: ", mean(se.est), "\n")


#################################################################################################
# Exclude the last value of your timeseries and predict it by utilizing nonlinear timeseries
# forecasting methods. Calculate the relative prediction error (%)
#################################################################################################


# Find the length of the time series
length_of_series <- length(x)

# Exclude the last point
x_1 <- x[-length_of_series]

pred <- knn_forecasting(x[1:length(x_1)], h = 1, lags = 8, k = 5, transform = "none")
knn_examples(pred)
plot(pred)

# Get the predicted value from the forecast result
predicted_value <- pred$prediction[1]

# Get the actual value from the original time series
actual_value <- x[length(x)]

# Calculate the relative error percentage
relative_error <- abs((predicted_value - actual_value) / actual_value) * 100

# Output the estimated relative error
estimated_error <- paste0("The estimated relative % error is ", round(relative_error, 2), "%.")
print(estimated_error)


#################################################################################################
# 3d representation of phase space with the attractor
#################################################################################################

# This function constructs the Takens embedding by creating a multidimensional representation of the time series.
tak = buildTakens(x,embedding.dim = 3, time.lag = 8)
plot3d(tak[,1], tak[,2], tak[,3],
          main = "system reconstructed phase space",
          col = 1, type="o",cex = 0.3)
install.packages("plot3Drgl")
# Assuming `tak` is a matrix containing the reconstructed phase space data
scatter3Drgl(x = tak[, 1], y = tak[, 2], z = tak[, 3],
             col = "blue", size = 2,
             main = "System Reconstructed Phase Space")



#################################################################################################
#  Phase Portrait
#################################################################################################

# Retrieve the time series x
x <- x  

# Set the time delay and embedding dimension
t <- 8
d <- 2

# Create the delayed coordinates
delayed_coords <- matrix(NA, nrow = length(x) - (d-1)*t, ncol = d)

for (i in 1:d) {
  delayed_coords[, i] <- x[(i-1)*t + 1:length(delayed_coords[, i])]
}

# Plot the phase portrait
plot(delayed_coords[, 1], delayed_coords[, 2], type = "l", col = 1,
     xlab = "X", ylab = "Y", main = "Phase Portrait")

#################################################################################################
#  Joyful Summer! Thanks for the teaching!
#################################################################################################