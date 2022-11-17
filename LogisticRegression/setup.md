### Part 1 (Model implementation)

Implement multinomial logistic regression and ordinal logistic regression.

For both, implement the (log-)likelihood of the model and an algorithm that fits the model using maximum likelihood estimation and can make predictions for new observations. For optimization you may use any third-party optimization library that allows for box-constraints (for example, `fmin_l_bfgs_b` from `scipy`). Optimization with numerical gradients will suffice.

Implement models as classes:

- `MultinomialLogReg`
- `OrdinalLogReg`

that provide a method `build(X, y)`, which returns the fitted model as an object, whose `predict(X)` method returns the predicted probabilities of given input samples.

### Part 2 (Application)

The `dataset.csv` (in implementation using `dataset_encoded.csv`) file contains data from over 5024 basketball shots in real-world basketball games. For each shot you have the shot type, which will be our target variable. For a detailed description of all the variables, [see the Methods section of Jornal](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0128885).

Use categorical regression to provide insights into the relationship between shot type and the other variables. How you prepare the independent variables and if you include the intercept is up to you. Keep in mind that your data preparation will affect interpretation, so do not take it lightly.

Interpret the coefficients of the categorical logistic regression: which independent variables affect the response and how? Is there a practical explanation? Regression coefficients, like any estimate, contain uncertainty. Before interpreting them, include a measure of uncertainty, such as bootstrapped confidence intervals (less difficult) or intervals based on asymptotic normality of MLE (more difficult, refer to literature).

### Part 3 (DGP)

Come up with a data generating process where ordinal logistic regression has a better log score than multinomial logistic regression.
Implement a generator (as function `multinomial_bad_ordinal_good`) that generates IID observations from this data generating process and explain why should ordinal logistic regression perform better.
Draw a training data set (you may choose the size, save it under the `MBOG_TRAIN` constant) and test data set (of size `1000`) and demonstrate empirically that ordinal logistic regression is indeed better.
