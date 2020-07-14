import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import matplotlib.pyplot as plt
import time

# Xu ly du lieu
name = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','PokerHand']

Train = pd.read_csv("D://Bao_cao_NLMH//poker-hand-training-true.data", header = None)   		# Doc data vao train
Test = pd.read_csv("D://Bao_cao_NLMH//poker-hand-testing.data", header = None)     
Train.columns = name                                                                    # Gan ten cho cac thuoc tinh
Test.columns = name                                                                     # Gan ten cho cac thuoc tinh

# Tap du lieu goc
x_train_all = Train.drop('PokerHand', axis = 1)
y_train_all = Train['PokerHand']   
x_test_all = Test.drop('PokerHand', axis = 1)											# Phan dung de test
y_test_all = Test['PokerHand']

print(Train.shape)
print(Test.shape[0])
# print("So luong phan tu va thuoc tinh cua tap train:" ,x_train_all.shape)              # Hien thi so luong phan tu va thuoc tinh cua tap train
# print("So luong phan tu va thuoc tinh cua tap test:" ,x_test_all.shape)                # Hien thi so luong phan tu va thuoc tinh cua tap test

# Xay dung cay quyet dinh
def Decision_Tree():
	start = time.time()
	print("Decision Tree:")
	print()
	model = DecisionTreeClassifier(random_state = 0, max_depth = 10, min_samples_leaf =6)
	model = model.fit(x_train_all, y_train_all)
	y_pred = model.predict(x_test_all)
	result = accuracy_score(y_test_all, y_pred)
	print("Độ chính xác:", result*100)
	print("Timer", time.time() - start)



def Bayes():
	start = time.time()
	print("Bayes:")
	print()
	from sklearn.naive_bayes import GaussianNB, MultinomialNB
	model_2 = GaussianNB()
	model_2.fit(x_train_all, y_train_all)
	y_pred_2 = model_2.predict(x_test_all)
	result = accuracy_score(y_test_all, y_pred_2)
	print("Độ chính xác:", result*100)
	# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_2)))
	# print()
	# accuracyBayes = accuracy_score(y_test_all, y_pred_2)
	# print(accuracyBayes*100, "%")


	print("Timer", time.time() - start)

# region Hồi quy
def Linear_Regression(x_train, y_train, x_test, y_test):
	print("Linear Regression:")
	print()
	from sklearn import linear_model    # tien hanh huan luyen tap du lieu
	lm = linear_model.LinearRegression()
	lm.fit(x_train, y_train)
	y_pred_3 = lm.predict(x_test)     # tien hanh test-du bao cho tap du lieu
	from sklearn.metrics import mean_squared_error  # tien hanh danh gia
	acc = accuracy_score(y_test, y_pred_3)
	print("Độ chính xác:", acc*100)
	
	print()
# endregion

#------------------------------------ 	Dung tap du lieu day du	----------------------------------#

# Decision_Tree()
# Bayes()


#-----------------------	Chi dung tap train	---------------------------------#
def test_Model():
	total = 0
	total_2 = 0
	arrDT = []
	arrBayes = []
	for i in range(0,10):
		global Train
		print("Lan", i+1, ":")
		X_train,X_test,y_train,y_test = train_test_split(Train.iloc[:,0:10], Train.iloc[:,10], test_size=0.3, random_state=i)
		#DT
		DT = DecisionTreeClassifier(random_state = 0, max_depth = 10, min_samples_leaf =6)
		DT.fit(X_train, y_train)
		y_pred = DT.predict(X_test)
		result = accuracy_score(y_test, y_pred)*100
		print(result)
		total += result
		print()
		arrDT.append(result)
		#Bayes
		model_2 = GaussianNB()
		model_2.fit(X_train, y_train)
		y_pred_2 = model_2.predict(X_test)
		result_2 = accuracy_score(y_test, y_pred_2)*100
		print(result_2)
		total_2 += result_2
		print()
		arrBayes.append(result_2)
	print("Ket qua trung binh:")
	total = total/10	
	print("Decision Tree",total,"%")
	total_2 = total_2/10	
	print("Bayes",total_2,"%")
	

test_Model()


# region ve bieu do

# label = ["Lần 1", "Lần 2", "Lần 3", "Lần 4", "Lần 5", "Lần 6", "Lần 7", "Lần 8", "Lần 9", "Lần 10"]
# DT = test_DT()
# BS = test_Bayes()
# index = np.arange(10)
# width = 0.30
# plt.bar(index, DT, width, color='green', label='Decision Tree')
# plt.bar(index+width, BS, width, color='red', label='Bayes')
# plt.title("Biểu đồ thể hiện độ chính xác")
# plt.ylabel("Độ chính xác(%)")
# plt.xlabel("Lần lặp")
# plt.xticks(index+ width/2, label)
# plt.legend(loc=0)
# plt.show()

# #	7503 test
# #	17507 train
# phantu = ["Train: 17,507 phần tử", "Test: 7,503 phần tử"]
# market_share = [17507, 7503]
# plt.pie(market_share, labels=phantu)
# plt.axis('equal')
# plt.show()

#endregion

#region Tim max_depth và min_sample_leaf tốt nhất

def Max_depth_min_leaf():
	new_ar = []
	for i in range(1,11):
		print("Max_depth:",i, "	",end=' ')
		for j in range(1,11):
			X_train,X_test,y_train,y_test = train_test_split(Train.iloc[:,0:10], Train.iloc[:,10], test_size=0.3, random_state=0)
			DT = DecisionTreeClassifier(random_state = 0, max_depth = i, min_samples_leaf =j)
			DT.fit(X_train, y_train)
			y_pred = DT.predict(X_test)
			result = accuracy_score(y_test, y_pred)
			new_ar.append(result*100)
			print(result*100)
		print()
	# return new_ar


# Max_depth_min_leaf()

#endregion

