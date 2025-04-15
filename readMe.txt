ForecastExperiment.m performs data adjustments and out of sample forecast experiment as described in paper and generates mainly results for Table 2 in paper. FullSample.m generates the full sample results for Table 3. 

Remaining functions are explained within code or in the paper but quick overview:

OutOfSampleStats.m returns out of sample statistics (ALPL, MSE) by recursively updating the sample and performing the estimation in an expanding window scheme.

NeweyWest.m computes the estimated long-run variance of a (scaled) mean of autocorrelated samples. 

mvtfit.m fits data to a t-distribution by iterating between matching moments and maximum likelihood to estimate the degree of freedom.

IGSampler.m generates samples from the adaptive hierarchical Minnesota type global-local prior using inverse gamma priors for the shrinkage factors. I confirmed in simulations that the model can return the Minnesota structure if the data is generated in such a way (even without imposing it in the prior), yields consistent estimates and can pick out important coefficients with the local components so I'm reasonably confident in not doing any coding errors. 

gigrnd.m generates a draw from the generalized inverse gaussian distribution.

getLags.m generates a matrix of lagged observations of the same time series with the same number of observations for each lag.

GARCHLikelihood.m computes the log-likelihood of an error following a garch(1,1) process.

GammaSampling.m performs Monte Carlo simulations of sampling and approximating moments from a gamma distribution by using Metropolis algorithms sampling from the Level and from the log respectively (shows that sampling from the log is more efficient).

GammaSampler.m is the same as IGSampler just using gamma priors.  

FacMissing.m fills in missing values of a (large) data set by fitting factor models and iterating as described in Stock & Watson (2002).

FacIC.m computes the number of factors chosen by the Bai & NG information criterion.

AR1FilterLik.m computes the log-likelihood of the model described in A.3 to replace covid observations. Formulas written out slightly different than in paper but mathematically equivalent (also tested in simulations that the maximizing this likelihood can correctly identify the parameters with sufficient data).

AR1_MSFE.m computes the out of sample forecast statistics for the AR(1) benchmark. 

gigrnd.m has been copied from the code associated with Chan (2021) provided on the authors website and NeweyWest.m has been copied from code provided to us for the course in Bayesian Econometrics. I adopted the latter by choosing the lag length automatically applying a rule of thumb instead of pre-specifying it. The remaining code is self-written. 


 