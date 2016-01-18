# DecisionTree
Implements an ID3-like decision-tree learner for classification

##Requirements
Python 2.7

## Important Scripts
*<b>dt-learn</b>: Script that calls dt-learn.py 
*<b>dt-learn.py</b>: Contains the implementation of the decision-tree
*<b>*.arff</b>: Sample training and testing files 

## Usage
<b>Command to run the classifier</b>: ./dt-learn <train-set-file> <test-set-file> m

m specifies the minimum number of training instances that need to be present to continue building the tree. If the number is less than 'm', then the node is converted into a leaf. This is one of the stopping criteria implemented apart from the normal ID3 based stopping criteria
