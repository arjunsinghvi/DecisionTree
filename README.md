# DecisionTree
Implements an ID3-like decision-tree learner for classification

Command to run the classifier: ./dt-learn <train-set-file> <test-set-file> m

m specifies the minimum number of training instances that need to be present to continue building the tree. If the number is less than 'm', then the node is converted into a leaf. This is one of the stopping criteria implemented apart from the normal ID3 based stopping criteria
