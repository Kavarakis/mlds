You are given a dataset of 1000+ basketball shots from the Euroleague with shot success, angle (if facing the basket, the angle is 0, shooting from the side, it is 90 degrees; we don't distinguish between left and right) and distance from basket (meters) recorded for each shot.

The goal is to use Bayesian Logistic regression to infer the relationship between the input variables and shot success. You may use a third party library (for example, in R you can use rstanarm::stan_glm with the usual binomal family and logit link), as long as inference is based on MCMC and you will get the result in the form of samples from the posterior distribution of coefficients. Sample at least 10000 samples. You may use default priors.

Before looking at any of the results, what is your personal opinion about the coefficient beta for distance? State it in probabilistic terms (as a distribution). Discuss after observing the results.

Plot a scatterplot with contours of the posterior samples of the angle and distance coefficients. Repeat the process for a smaller subset of only 50 shots (sampled at random). Discuss the similarities and differences, if any.

Formulate each of the following as a probabilistic question and estimate an answer based on the posterior using all available data:

    Which is more important for shot success, angle or distance?
    Does shot success increase or decrease with increasing angle (the further on the sides we are)?

Submit a 1-page A4 pdf with the results. No need to describe the problem or dataset (you may assume that the person reading it is familiar with the task, so just submit your results and discussion). Note that in this homework you are allowed to skip two important parts without loss of points: diagnosing problems with MCMC and the quantification of MCMC approximation error; however, if you want to practice, you may still do it.
