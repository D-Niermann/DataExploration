
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from prettytable import PrettyTable



def prettyPrint(label, arg):
    t = PrettyTable([label])
    t.add_row([arg])
    print(t)


def maximizeTPRFPR(tpr : np.array, fpr : np.array):
	s = tpr-fpr
	w = np.where(s==s.max())
	return {"tpr":tpr[w], "fpr":fpr[w]}

def fetchAndPrepareTitanicData():
	data : pd.DataFrame = seaborn.load_dataset("titanic")

	## remove nan and unecessary data
	dropCols = ["pclass", "deck", "who", "adult_male",
				"embarked", "embark_town" , "alive", "alone"]
	data = data.drop(dropCols, axis=1)
	data = data.replace(np.nan, 0)
	# print("#NaNs:\n", data.isna().sum())
	# print("---------------")

	## make types numeric
	def mapClass(className):
		a = 1
		if className == "Second":
			a = 2
		elif className == "Third":
			a = 3
		return a

	data["class"] = data["class"].map(lambda x: mapClass(x))
	data["isMale"] = data["sex"].map(lambda x: 1 if x=="male" else 0)
	data = data.drop("sex",axis=1)
	# print("Type:\n", data.dtypes)

	return data


def loadPreparedData():
	data = pd.read_csv("./exampleData.csv")
	return data


def splitData(data):
	y = data["survived"]
	x = data.drop("survived",axis=1)

	testTrainRatio = 2/10
	assert len(x)==len(y), "len x and len y not the same!"
	mask = np.random.rand(len(x)) > testTrainRatio
	x_train = x[mask]
	y_train = y[mask]
	x_test = x[~mask]
	y_test = y[~mask]
	print("Len train data:",len(x_train))
	print("Len test data:",len(x_test))
	return x_train,y_train,x_test,y_test

def linRegression(x_train, y_train, x_test, y_test):
	model = Ridge(alpha=0.2, fit_intercept=False)
	model.fit(x_train, y_train)

	w = model.coef_

	return model, pd.DataFrame(w,x_train.columns)


def score(pred, label):
	### get params
	u = ((label - pred) ** 2).sum()
	v = ((label - label.mean()) ** 2).sum()
	score:float = 1-(u/v)

	return score