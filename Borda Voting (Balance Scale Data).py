# Majority Voting. Balance Scale Data.
#-------------------------------------------------------------------------------------
# Attribute Information:
# 1. Class Name: 3 (L, B, R) 
# 2. Left-Weight: 5 (1, 2, 3, 4, 5) 
# 3. Left-Distance: 5 (1, 2, 3, 4, 5) 
# 4. Right-Weight: 5 (1, 2, 3, 4, 5) 
# 5. Right-Distance: 5 (1, 2, 3, 4, 5)
#-------------------------------------------------------------------------------------
import timeit
start = timeit.default_timer()

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import sklearn
import numpy as np
import scipy.stats as ss



#Open/load training data for CoverType. Training & testing are both same. 
#Will need to split into training/test sets (CrossValidation/Holdout)
train_file = open('balance-scale.data', 'r').read().split("\n")
train_file.pop() #remove last element in list, the extra ' '


# Organize the training data. Convert the 1D list of strings (individual records) into a 
# 2D list of integer attributes, where each row is a single record. Final format of 
# training data will look like: [[Class, LW, LD, RW, RD], [record2],...[record625]]
trainset = []
for single_record in train_file:
	single_record = single_record.split(',')
	# Turn the list of string attributes into appropriate integers and floats. 
	single_recordAtrib = [str(single_record[0]), int(single_record[1]), int(single_record[2]), int(single_record[3]), int(single_record[4])]
	trainset.append(single_recordAtrib)



# Extract training class labels for each record/row and add it to a separate list. 
# It will always be the first attribute, so pop the first value in each sublist. This is a 
# 1D list of class labels. 
trainset_labelsY = []
for single_record in trainset:
	label = single_record.pop(0)
	trainset_labelsY.append(label)
	
	
# print(trainset)
# print(trainset_labelsY)
# print(len(trainset))
# print(len(trainset_labelsY))


# Set aside some of the training set, "trainset" for Cross Validation/Holdout. We will call
# this the "testset". We will have approximately a 70-30 split. Where 70% of data is 
# training and 30% is the test set.
original_trainsetlen = len(trainset)
original_trainsetlnY = len(trainset_labelsY)
N = int(len(trainset)*.70)

testset = trainset[N:]
trainset = trainset[:N]

testset_labelsY = trainset_labelsY[N:]
trainset_labelsY = trainset_labelsY[:N]

#------------------------------------------
# Normalize the training data: 
# scaler = Normalizer().fit(trainset)
# trainset = scaler.transform(trainset)
# 
# 
# # Normalize the test data: Don't need if using KFold Validation.
# scaler_test = Normalizer().fit(testset)
# testset = scaler_test.transform(testset)
# 
# 
# # Instantiate the PCA tool to transform the data using Principal Component Analysis.
# pca = PCA()
# 
# # Transform the training data so you have a PCA-transformed data set.
# trainset = pca.fit_transform(trainset)
# 
# # transform test data using the previously fitted PCA object. 
# testset = pca.transform(testset)




# ------------------ BORDA VOTING IMPlEMAENTATION?---------------------------
# Borda Conunting Method Function
def myborda(probs):
    probrank=[]
    
    for i in range(probs.shape[0]):
        probrank.append(ss.rankdata(probs[i,:],method='average')-1)
    
    probrank=np.array(probrank)
    #print(probrank)
    
    ranksums=np.sum(probrank,0)
    #print(ranksums)
    
    #return the column index of the class with the max number of points
    return np.argmax(ranksums)


list_ensemble_accuracy = []
for i in range(10):
	#Bagging Samples, each row of mega_index is a sample
	# each column of a row being the index of trainset to be included in the sample
	num_trees = 20

	mega_index=np.random.randint(len(trainset),size=(num_trees,int(len(trainset)*(2/3))))

	# print(mega_index)
# 	print(len(mega_index[0]))
# 	print(len(mega_index))

# grab the training data at that specific index for each subsample. Bagging Model.
	sub_samples_x  = []
	sub_samples_y = []
	for row in mega_index:
		list_samples = []
		list_samples_y = []
		for col in row:
			record = trainset[col]
			record_y = trainset_labelsY[col]
			list_samples.append(record)
			list_samples_y.append(record_y)
		sub_samples_x.append(list_samples)
		sub_samples_y.append(list_samples_y)
	


	# Create three decision trees. for ensemble
	master_DT = DecisionTreeClassifier(max_depth = 3, random_state = 42)

	fit0 = master_DT.fit(sub_samples_x[0], sub_samples_y[0])
	fit1 = master_DT.fit(sub_samples_x[1], sub_samples_y[1])
	fit2 = master_DT.fit(sub_samples_x[2], sub_samples_y[2])
	fit3 = master_DT.fit(sub_samples_x[3], sub_samples_y[3])
	fit4 = master_DT.fit(sub_samples_x[4], sub_samples_y[4])
	fit5 = master_DT.fit(sub_samples_x[5], sub_samples_y[5])
	fit6 = master_DT.fit(sub_samples_x[6], sub_samples_y[6])
	fit7 = master_DT.fit(sub_samples_x[7], sub_samples_y[7])
	fit8 = master_DT.fit(sub_samples_x[8], sub_samples_y[8])
	fit9 = master_DT.fit(sub_samples_x[9], sub_samples_y[9])
	fit10 = master_DT.fit(sub_samples_x[10], sub_samples_y[10])
	fit11 = master_DT.fit(sub_samples_x[11], sub_samples_y[11])
	fit12 = master_DT.fit(sub_samples_x[12], sub_samples_y[12])
	fit13 = master_DT.fit(sub_samples_x[13], sub_samples_y[13])
	fit14 = master_DT.fit(sub_samples_x[14], sub_samples_y[14])
	fit15 = master_DT.fit(sub_samples_x[15], sub_samples_y[15])
	fit16 = master_DT.fit(sub_samples_x[16], sub_samples_y[16])
	fit17 = master_DT.fit(sub_samples_x[17], sub_samples_y[17])
	fit18 = master_DT.fit(sub_samples_x[18], sub_samples_y[18])
	fit19 = master_DT.fit(sub_samples_x[19], sub_samples_y[19])



	# Posterior probabilities of each of the bagged decision trees. 
	probs0 = fit0.predict_proba(testset)
	probs1 = fit1.predict_proba(testset)
	probs2 = fit2.predict_proba(testset)
	probs3 = fit3.predict_proba(testset)
	probs4 = fit4.predict_proba(testset)
	probs5 = fit5.predict_proba(testset)
	probs6 = fit6.predict_proba(testset)
	probs7 = fit7.predict_proba(testset)
	probs8 = fit8.predict_proba(testset)
	probs9 = fit9.predict_proba(testset)
	probs10 = fit10.predict_proba(testset)
	probs11 = fit11.predict_proba(testset)
	probs12 = fit12.predict_proba(testset)
	probs13 = fit13.predict_proba(testset)
	probs14 = fit14.predict_proba(testset)
	probs15 = fit15.predict_proba(testset)
	probs16 = fit15.predict_proba(testset)
	probs17 = fit17.predict_proba(testset)
	probs18 = fit18.predict_proba(testset)
	probs19 = fit19.predict_proba(testset)


	predictions = []
	index_probs = 0 
	for record in testset:
		new_probs_list = []
		new_probs_list.append(probs0[index_probs])
		new_probs_list.append(probs1[index_probs])
		new_probs_list.append(probs2[index_probs])
		new_probs_list.append(probs3[index_probs])
		new_probs_list.append(probs4[index_probs])
		new_probs_list.append(probs5[index_probs])
		new_probs_list.append(probs6[index_probs])
		new_probs_list.append(probs7[index_probs])
		new_probs_list.append(probs8[index_probs])
		new_probs_list.append(probs9[index_probs])
		new_probs_list.append(probs10[index_probs])
		new_probs_list.append(probs11[index_probs])
		new_probs_list.append(probs12[index_probs])
		new_probs_list.append(probs13[index_probs])
		new_probs_list.append(probs14[index_probs])
		new_probs_list.append(probs15[index_probs])
		new_probs_list.append(probs16[index_probs])
		new_probs_list.append(probs17[index_probs])
		new_probs_list.append(probs18[index_probs])
		new_probs_list.append(probs19[index_probs])
	
		new_probs_list=np.array(new_probs_list)
		borda_predict = myborda(new_probs_list)
		predictions.append((master_DT.classes_[borda_predict]))
		index_probs += 1


	accuracy = sklearn.metrics.accuracy_score(testset_labelsY, predictions)
	list_ensemble_accuracy.append(accuracy)
	


list_ensemble_accuracy = np.array(list_ensemble_accuracy)




print("Not Normalized/PCA: ")
print(list_ensemble_accuracy)
print("Normalized/PCA")
print("MIN: ", list_ensemble_accuracy.min())
print("MAX: ", list_ensemble_accuracy.max())
print("MEAN: ", list_ensemble_accuracy.mean())


# print(sub_samples_x)
# print(sub_samples_x[0])
# print(sub_samples_y)
# print(sub_samples_y[0])
# print(predictions)
# print(len(predictions))
		



# Calculates average runtime of the code.
stop = timeit.default_timer()

print('Time: ', stop - start)  




# WITHOUT DATA NORMALIZATION/PCA: 0.7970302099334358
# WITH DATA NORMALIZATION/PCA:  0.8706605222734254