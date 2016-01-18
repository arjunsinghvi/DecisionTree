import math
import sys

def entropy_calculator(training_data):

	classification_frequency = {}
	entropy = 0.0

	for training_instance in training_data:
		if training_instance[-1] in classification_frequency.keys():
			classification_frequency[training_instance[-1]] += 1
		else:
			classification_frequency[training_instance[-1]] = 1

	for count in classification_frequency:
		proportion = classification_frequency[count]/float(len(training_data))
		entropy += (-proportion) * math.log(proportion,2)

	return entropy

def discrete_informationGain_calculator(training_data,features,relevant_feature):

	frequency_subDataset = {}
	relevant_feature_index = features.index(relevant_feature)
	expected_entropy = 0
	for training_instance in training_data:
		if training_instance[relevant_feature_index] in frequency_subDataset.keys():
			frequency_subDataset[training_instance[relevant_feature_index]][0] += 1
			frequency_subDataset[training_instance[relevant_feature_index]][1].append(training_instance)
		else:
			frequency_subDataset[training_instance[relevant_feature_index]]=[1,[training_instance]]

	for feature_value in frequency_subDataset.keys():
		proportion = frequency_subDataset[feature_value][0]/float(len(training_data))
		training_data_subset = frequency_subDataset[feature_value][1]
		expected_entropy += proportion * entropy_calculator(training_data_subset)
	
	information_gain = entropy_calculator(training_data) - expected_entropy

	return (information_gain, None)

def candidate_thresholds_calculator(training_data, relevant_feature_index):
	relevant_features_value_list = []
	feature_classification_distribution = {}
	candidate_thresholds = []
	for training_instance in training_data:
		if float(training_instance[relevant_feature_index]) not in relevant_features_value_list:
			relevant_features_value_list.append(float(training_instance[relevant_feature_index]))
			feature_classification_distribution[float(training_instance[relevant_feature_index])] = {}
			feature_classification_distribution[float(training_instance[relevant_feature_index])][training_instance[-1]] = 1
			if training_instance[-1] == features_values['class'][0]:
				unclassified_class = features_values['class'][1]
			else:
				unclassified_class = features_values['class'][0]
			feature_classification_distribution[float(training_instance[relevant_feature_index])][unclassified_class] = 0
		else:
			feature_classification_distribution[float(training_instance[relevant_feature_index])][training_instance[-1]] += 1

	relevant_features_value_list.sort()
	classification_label_1 = features_values['class'][0]
	classification_label_2 = features_values['class'][1]

	for index in range(0,len(relevant_features_value_list)-1):
		current_value = relevant_features_value_list[index]
		next_value = relevant_features_value_list[index+1]

		#If current value and next value have different classifications
		if (feature_classification_distribution[current_value][classification_label_1] > 0 and feature_classification_distribution[next_value][classification_label_2] >0 ) or (feature_classification_distribution[current_value][classification_label_2] > 0 and feature_classification_distribution[next_value][classification_label_1] >0 ):
			candidate_thresholds.append((current_value+next_value)/2.0)

	return candidate_thresholds

def numeric_informationGain_calculator(training_data,features, relevant_feature):
	#print relevant_feature
	relevant_feature_index = features.index(relevant_feature)
	candidate_thresholds = candidate_thresholds_calculator(training_data, relevant_feature_index)
	candidate_informationGain = []

	for threshold in candidate_thresholds:
		frequency_subDataset ={'lt_eq':[0,[]],'gt':[0,[]]}
		expected_entropy = 0
		for training_instance in training_data:
			if float(training_instance[relevant_feature_index]) <= threshold:
				frequency_subDataset['lt_eq'][0] += 1
				frequency_subDataset['lt_eq'][1].append(training_instance)
			else:
				frequency_subDataset['gt'][0] += 1
				frequency_subDataset['gt'][1].append(training_instance)
		for relational_operator in frequency_subDataset.keys():
				proportion = frequency_subDataset[relational_operator][0]/float(len(training_data))
				training_data_subset = frequency_subDataset[relational_operator][1]
				expected_entropy += proportion * entropy_calculator(training_data_subset)
		candidate_informationGain.append(expected_entropy)
		#print str(frequency_subDataset['lt_eq'][0])+"  "+str(frequency_subDataset['gt'][0])+" "+str(expected_entropy)
	
	if (len(candidate_thresholds) != 0):
		initial_entropy = entropy_calculator(training_data)
		#print initial_entropy
		candidate_informationGain_list = [(initial_entropy - entropy) for entropy in candidate_informationGain]
		chosen_informationGain = max(candidate_informationGain_list)	
		chosen_threshold = candidate_thresholds[candidate_informationGain_list.index(chosen_informationGain)]
	else:
		chosen_informationGain = None
		chosen_threshold = None
	return (chosen_informationGain, chosen_threshold)

def best_feature_split_finder(training_data,features):
	
	maximum_informationGain = 0.0
	best_feature = None
	best_threshold = None
	for current_feature in features:
		if(type(features_values[current_feature]) is list):
			current_informationGain, current_threshold = discrete_informationGain_calculator(training_data, features, current_feature)
		else:
			current_informationGain, current_threshold = numeric_informationGain_calculator(training_data, features, current_feature)
		if(current_informationGain != None):
			if(current_informationGain > maximum_informationGain):
				maximum_informationGain = current_informationGain
				best_feature = current_feature
				best_threshold = current_threshold
	return (best_feature, best_threshold)

def subset_training_data_creator(training_data, features, relevant_feature, relevant_feature_output, discrete_flag):


	relevant_feature_index = features.index(relevant_feature)
	subset_training_data = []

	if(type(discrete_flag) is bool and discrete_flag == True):
		for training_instance in training_data:
			if(training_instance[relevant_feature_index] == relevant_feature_output):
				data_instance = []
				for index in range(0,len(training_instance)):
					if(index != relevant_feature_index):
						data_instance.append(training_instance[index])
				subset_training_data.append(data_instance)
	else:
		threshold = discrete_flag
		
		for training_instance in training_data:
			if(((relevant_feature_output == "<=") and (float(training_instance[relevant_feature_index]) <= threshold)) or ((relevant_feature_output == ">") and (float(training_instance[relevant_feature_index]) > threshold))):
				subset_training_data.append(training_instance)

	return subset_training_data


def decisionTree_builder(training_data, features, m,recursion_count=-1):
	global output_str
	recursion_count += 1
	training_data = training_data[:]
	classification_outputs = []
	for training_instance in training_data:
		classification_outputs.append(training_instance[-1])
	
	higher_priority_classification_count = classification_outputs.count(features_values['class'][0])
	lower_priority_classification_count = len(classification_outputs) - higher_priority_classification_count
	distribution_str = " ["+str(higher_priority_classification_count)+" "+str(lower_priority_classification_count)+"]"
	output_str = output_str[:-1]
	if recursion_count != 0:
		output_str += distribution_str + "\n"

	if (len(training_data) == 0):
		output_str = output_str[:-1]
		output_str += ": "+features_values['class'][0]+"\n"
		return features_values['class'][0]
	elif len(classification_outputs) == classification_outputs.count(classification_outputs[0]):
		output_str = output_str[:-1]
		output_str += ": "+classification_outputs[0]+"\n"
		return classification_outputs[0]

	elif (len(features) == 0) or (len(training_data) < m):
		if(higher_priority_classification_count >= lower_priority_classification_count):
			output_str = output_str[:-1]
			output_str += ": "+ features_values['class'][0]+"\n"
			return features_values['class'][0]
		elif(higher_priority_classification_count < lower_priority_classification_count):
			output_str = output_str[:-1]
			output_str += ": "+features_values['class'][1]+"\n"
			return features_values['class'][1]
	else:
		
		best_split,best_threshold = best_feature_split_finder(training_data, features)
		if(best_split is None): # No +ve information_gain
			if(higher_priority_classification_count >= lower_priority_classification_count):
				output_str = output_str[:-1]
				output_str += ": "+ features_values['class'][0]+"\n"
				return features_values['class'][0]
			elif(higher_priority_classification_count < lower_priority_classification_count):
				output_str = output_str[:-1]
				output_str += ": "+features_values['class'][1]+"\n"
				return features_values['class'][1]
		else:
			decision_tree = {best_split:{}}
			if best_threshold is None:
				best_feature_outputs = features_values[best_split]
				for feature_output in best_feature_outputs:
					for tab_count in range(0,recursion_count):
						output_str += "|       "
					output_str += best_split+" = "+feature_output+'\n'
					training_data_subset = subset_training_data_creator(training_data,features,best_split,feature_output,True)
					features_subset = features[:]
					features_subset.remove(best_split)
					sub_decisionTree = decisionTree_builder(training_data_subset,features_subset, m,recursion_count)
					decision_tree[best_split][feature_output] = sub_decisionTree				
			else:
				best_feature_outputs = ["<=", ">"]
				for feature_output in best_feature_outputs:
					for tab_count in range(0,recursion_count):
						output_str += "|       "
					output_str += best_split+" "+feature_output+" "+str("%.6f"%best_threshold)+'\n'
					training_data_subset = subset_training_data_creator(training_data,features,best_split,feature_output,best_threshold)
					features_subset = features[:]
					#features_subset.remove(best_split)
					sub_decisionTree = decisionTree_builder(training_data_subset, features_subset, m,recursion_count)
					feature_output = feature_output+" "+str(best_threshold)
					decision_tree[best_split][feature_output] = sub_decisionTree


	return decision_tree

def classification(decision_tree, test_data):
	pcount = 0
	ncount =0 
	print "<Predictions for the Test Set Instances>"
	for index in range(0,len(test_data)):
		test_instance = test_data[index]
		if type(decision_tree) is dict:
			dynamic_tree = decision_tree.copy()
			output_label = ""
			while (type(dynamic_tree) is dict):
				head_attribute = dynamic_tree.keys()[0]
				head_attribute_index = ordered_features.index(head_attribute)
				head_attribute_value = test_instance[head_attribute_index]
				dynamic_tree = dynamic_tree[head_attribute]
				next_level_keys = dynamic_tree.keys()
				if '<=' in next_level_keys[0] or '>' in next_level_keys[0]:
					if(eval(head_attribute_value+next_level_keys[0])):
						output_label = dynamic_tree[next_level_keys[0]]
						dynamic_tree = dynamic_tree[next_level_keys[0]]
					else:
						output_label = dynamic_tree[next_level_keys[1]]
						dynamic_tree = dynamic_tree[next_level_keys[1]]	
				elif (head_attribute_value in dynamic_tree.keys()):
					output_label = dynamic_tree[head_attribute_value]
					dynamic_tree = dynamic_tree[head_attribute_value]
				else:
					print "Invalid Input"
		else:
			output_label = decision_tree
		#print output_label+" "+test_instance[-1]
		print "%3d: Actual: "%(index+1)+test_instance[-1]+"  Predicted: "+output_label
		if output_label == test_instance[-1]:
			pcount += 1
		else:
			ncount += 1
	print "Number of correctly classified: "+str(pcount)+"  Total number of test instances: "+str(pcount+ncount)
	return (pcount,ncount)

def read_file(filename,train_flag = False):
	fp = open(filename,"r")
	lines = fp.read().split("\n")
	lines = lines[1:] #removing the relation
	lines = [line for line in lines if line!='']
	data = []
	if(train_flag):
		ordered_features = []
		features_values = {}
		for line in lines:
			if line.startswith("@attribute"):
				line_parts =line.split(' ')
				ordered_features.append(eval(line_parts[1]))
				if len(line_parts) <4:
					features_values[eval(line_parts[1])] = line_parts[-1]	
				else:
					features_values[eval(line_parts[1])] = []
					for index in range(3,len(line_parts)):
						features_values[eval(line_parts[1])].append(line_parts[index][:-1])
			elif not line.startswith("@"):
				data.append(line.split(','))
		ordered_features.remove('class')
		return_data = (ordered_features,features_values,data)
	else:
		for line in lines:
			if not line.startswith("@"):
				data.append(line.split(','))
		return_data = data

	return return_data 

'''
def pick_random_training_data(training_data,percentage):
	training_data_size = int(math.ceil((len(training_data) * percentage *1.0)/100.0))
	training_indices = []
	import random
	while len(training_indices)!=training_data_size:
		random_index = random.randrange(0,len(training_data))
		if random_index not in training_indices:
			training_indices.append(random_index)
	random_training_data = []
	for index in training_indices:
		random_training_data.append(training_data[index])

	return random_training_data
'''

if __name__ == "__main__" :
	
	training_filename = str(sys.argv[1])
	test_filename = str(sys.argv[2])
	m = int(sys.argv[3])
	output_str=""
	ordered_features,features_values,training_data = read_file(training_filename,True) 
	tree = decisionTree_builder(training_data,ordered_features,m)
	if type(tree) is str:
		output_str = output_str[2:]
	test_data = read_file(test_filename)
	print ""
	print output_str[:-1]
	classification(tree, test_data)
	

