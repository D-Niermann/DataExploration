import seaborn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression

def fetchAndPrepareTitanicData():
	data : pd.DataFrame = seaborn.load_dataset("titanic")

	## remove nan and unecessary data
	dropCols = ["pclass", "deck", "who", "adult_male", "embarked", "embark_town" , "alive", "alone"]
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

def linRegression(x_train, y_train):
	model = LinearRegression(fit_intercept=False)
	model.fit(x_train, y_train)

	w = model.coef_
	y_pred = model.predict(x_test)
	### get params
	u = ((y_test - y_pred) ** 2).sum()
	v = ((y_test - y_test.mean()) ** 2).sum()
	score = 1-(u/v)

	print("Score:" , np.round(score,3))

	return model, pd.DataFrame(x_train.columns, w)






data = loadPreparedData()
x_train,y_train,x_test,y_test = splitData(data)
model, w = linRegression(x_train, y_train)


## shap
shap.initjs()
explainer = shap.LinearExplainer(model,x_train,feature_dependence="independent")
shap_vals = explainer.shap_values(x_test)
## summary plot
shap.summary_plot(shap_vals,x_test.values,x_train.columns)
## one sample person
i=1
shap.force_plot(explainer.expected_value,shap_vals[i,:],x_test.values[i,:],x_test.columns,matplotlib=True)
