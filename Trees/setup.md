For this homework, you will implement classification trees and random forests. Your implementations must support numeric input variables and a binary target variable.

You will implement these methods as classes (`Tree`, `RandomForest`) that provide a method `build`, which returns the model as an object, whose predict method returns the predicted target class of given input samples (see test_hw2.py for usage examples):

Tree - a flexible classification tree, with the following attributes: (1) rand, a random generator, for reproducibility, of type random.Random; (2) `get_candidate_columns`, a function that returns a list of column indices considered for a split (needed for the random forests); and (3) `min_samples`, the minimum number of samples, where a node is still split. Use the Gini impurity for selecting the best split.

RandomForest, with attributes: (1) `rand`, a random generator; (2) `n`: number of
bootstrap samples. The RandomForest should use an instance of Tree internally. Build full trees (`min_samples=2`). For each split, consider random (square root of the number of input variables) variables.

Also implement variable importance for random forests as described in section 10 of [Breiman (2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf); implement it as method importance() of the random forest model.

Apply the developed methods to the `tki-resistance.csv` FTIR spectral data set. Always use the first 130 rows of data as the training set and the remainder as the testing set. Do the following:

- In function `hw_tree_full`, build a tree with `min_samples=2`. Return misclassification rates and standard errors (to quantify uncertainty) on training and testing data.

- In function `hw_randomforest`, use random forests with `n=10`0 trees with `min_samples=2`. Return misclassification rates and standard errors (to quantify uncertainty) on training and testing data.

As a rough guideline, building the full tree on this data set should take less than 10 seconds - more shows inefficiencies in the implementation. Likewise, computing random forest variable importance for all variables should be faster than building the random forest.

Your code needs to be Python 3.8 compatible and needs to conform to the unit tests from test_hw_tree.py; see tests for the precise interface. In it, execute any code only under if **name** == "**main**". Your need to write the crux of the solution yourself, but feel free to use libraries for data reading and management (numpy, pandas) and drawing (matplotlib). Submit your code in a single file named hw_tree.py.

Submit a report in a single .pdf file (max one page). In the report:

Show misclassification rates from `hw_tree_full`. Explain how did you quantify the uncertainty of your estimates.

Show misclassification rates from `hw_randomforest`.

Plot misclassification rates versus the number of trees `n`.

Plot variable importance for the given data set for a RF with `n=100` trees. See Figure 4 in Breiman (2001). For comparison, also show variables from the roots of 100 non-random trees (sensibly sample data to produce different trees) on the same plot. Comment on results.
