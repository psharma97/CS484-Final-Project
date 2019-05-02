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
# original_trainsetlen = len(trainset)
# original_trainsetlnY = len(trainset_labelsY)
# N = int(len(trainset)*.70)
# 
# testset = trainset[N:]
# trainset = trainset[:N]
# 
# testset_labelsY = trainset_labelsY[N:]
# trainset_labelsY = trainset_labelsY[:N]

#------------------------------------------
# Normalize the training data: 
scaler = Normalizer().fit(trainset)
trainset = scaler.transform(trainset)


# Normalize the test data: Don't need if using KFold Validation.
# scaler_test = Normalizer().fit(testset)
# testset = scaler_test.transform(testset)


# Instantiate the PCA tool to transform the data using Principal Component Analysis.
pca = PCA()

# Transform the training data so you have a PCA-transformed data set.
trainset = pca.fit_transform(trainset)

# transform test data using the previously fitted PCA object. 
# testset = pca.transform(testset)




# ------------------ MAJORITY VOTING IMPlEMAENTATION?---------------------------
kfold = model_selection.KFold(n_splits=10, random_state=42)
cart = DecisionTreeClassifier()


model = BaggingClassifier(base_estimator = DecisionTreeClassifier(), n_estimators= 100, random_state= 42)

results = model_selection.cross_val_score(model, trainset, trainset_labelsY, cv = kfold)

print(results)
print("MIN: ", results.min())
print("MAX: ", results.max())
print("MEAN: ", results.mean())






# Calculates average runtime of the code.
stop = timeit.default_timer()

print('Time: ', stop - start)  




# WITHOUT DATA NORMALIZATION/PCA: 0.7970302099334358
# WITH DATA NORMALIZATION/PCA:  0.8706605222734254



