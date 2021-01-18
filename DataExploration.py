import seaborn
import numpy as np
import numpy.random as rnd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import shap
import importlib
import functions as F
import NNFunctions as NN
importlib.reload(F)
importlib.reload(NN)

"""
TODO: Was genau ist das expected value vom explainer -> mittelwert über die gemachten predictions
TODO: 	- x test daten in N subsets splitten und shap untersuchen  -  wenn verteilung der feature wenig unterschiedlich, shap identisch? 
			-> ja
		- wenn aber feature stark anders (zB sortiere feature1 und splitte so dass werte komplett anders in den subsets) dann shap values auch anders?
			-> Ja! bei 0,1 verteilung zerfällt shap prediction 
				komplett auf 0, wenn bei verteilung etnweder nur
				die nullen oder nur die einsen genommen werden
		- ist das so und will man das so? 
			"Ich denke ja. Die gewichte des modells sind ja immer gleich - heißt in denen sieht
			man den effekt der veränderten features nicht. Durch shap sieht man welche features
			mit der verteilung der vorliegenden daten welchen einfluss haben."
TODO: Kann ein feature welches immer 0 ist hohe gewichte im lin fit kriegen?  -> Ja! sehr einfach sogar
		Ändert die Ridge regression das? -> Ja! w[i] ist dann 0
		Dazu entweder einfach ein feature = 0 setzen und lernen oder ein gelerntes modell nehmen und feature X runter und gewicht zu X sehr hoch setzen - sollte die 
		performance wenig belasten wenn das feature X ca. 0 ist.
"""

### data
data = F.loadPreparedData()
### norm each feature to 1 (makes the weights comparable - not necessary with shap, but nice to have)
data = data/data.max(axis=0)
## append sqrt(fare) to better the distribution
data["sqrt(fare)"] = np.sqrt(data["fare"])
## append 0 data colum
data["testZero"] = rnd.random(len(data))*0.00001
## test train split
x_train,y_train,x_test,y_test = F.splitData(data)
### correlation
corr = data.corr()["survived"]
F.prettyPrint("Correlation", corr.sort_values(key=lambda x: abs(x), ascending=False))


DO_LIN_FIT   = 0
DO_NN_FIT    = 1
DO_NNLIN_FIT = 0



if DO_LIN_FIT:
	print("-----------------------------------------")
	print(" ######## LIN MODEL ########")
	print("-----------------------------------------")
	### Linear Model
	linModel, w = F.linRegression(x_train, y_train, x_test, y_test)
	w.plot.barh()
	plt.title("Weights")
	linPred = linModel.predict(x_test)
	scoreLin = F.score(linPred, y_test)
	print("Score:" , np.round(scoreLin,3))

	## linear shap
	explainer = shap.LinearExplainer(linModel,x_train,feature_dependence="independent")
	shap_valsLin = explainer.shap_values(x_test)
	## summary plot
	plt.figure()
	shap.summary_plot(shap_valsLin,x_test.values,x_train.columns,alpha=0.5)
	## one sample person
	# i=1
	# shap.force_plot(explainer.expected_value,shap_valsLin[i,:],x_test.values[i,:],x_test.columns,matplotlib=True)

	shap.force_plot(explainer.expected_value,
					shap_valsLin,
					x_test.values,
					x_test.columns)

	shap.dependence_plot("fare",shap_valsLin,x_test.values,x_test.columns,interaction_index="isMale",alpha=0.5)




if DO_NN_FIT:
	print("-----------------------------------------")
	print(" ######## NN MODEL ########")
	print("-----------------------------------------")
	n_inputs = x_train.shape[1]
	n_labels = y_train.shape[1] if len(y_train.shape)>1 else 1
	### Keras model
	NNModel = NN.kerasModel(x_train, y_train,  [n_inputs, 30, n_labels])
	NNPred : np.array = NNModel.predict(x_test, batch_size=None, verbose=0)

	## NN shap
	NNexplainer = shap.DeepExplainer(NNModel,x_test.values)

	""" Why is this with index 0? -> for multiple outputs?"""
	shap_valsNN = NNexplainer.shap_values(x_test.values)[0]

	shap.summary_plot(shap_valsNN,x_test.values,x_train.columns,alpha=0.5)
	shap.force_plot(NNexplainer.expected_value.numpy(),
					shap_valsNN,
					x_test.values,
					x_test.columns,
					matplotlib = False)

	shap.dependence_plot("isMale", shap_valsNN, x_test.values, x_test.columns, alpha=0.5)
	plt.title("NN Model")



if DO_NNLIN_FIT:
	print("-----------------------------------------")
	print(" ######## NNModel Linear ########")
	print("-----------------------------------------")
	NNModelLin = NN.kerasModel(x_train, y_train,  [n_inputs, n_labels])
	NNPredLin : np.array = NNModelLin.predict(x_test, batch_size=None, verbose=0)
	NNexplainer = shap.DeepExplainer(NNModelLin,x_test.values)
	""" Why is this with index 0? -> for multiple outputs? """
	shap_valsNNLin = NNexplainer.shap_values(x_test.values)[0]
	shap.summary_plot(shap_valsNNLin,x_test.values,x_train.columns,alpha=0.5)





if DO_LIN_FIT and DO_NNLIN_FIT and DO_NN_FIT:
	print("-----------------------------------------")
	print(" ######## ROC ########")
	print("-----------------------------------------")
	from sklearn import metrics
	M = {}
	fpr, tpr, thresholds = metrics.roc_curve(y_test, linPred)
	M["lin"] = {"fpr" : fpr, "tpr":tpr, "thresholds":thresholds}

	fpr, tpr, thresholds = metrics.roc_curve(y_test, NNPred)
	M["NN"] = {"fpr" : fpr, "tpr":tpr, "thresholds":thresholds}

	fpr, tpr, thresholds = metrics.roc_curve(y_test, NNPredLin)
	M["NNLin"] = {"fpr" : fpr, "tpr":tpr, "thresholds":thresholds}


	### get best tpr and fpr
	M["lin"]["best"] = F.maximizeTPRFPR(M["lin"]["tpr"], M["lin"]["fpr"])
	M["NN"]["best"] = F.maximizeTPRFPR(M["NN"]["tpr"], M["NN"]["fpr"])
	M["NNLin"]["best"] = F.maximizeTPRFPR(M["NNLin"]["tpr"], M["NNLin"]["fpr"])
	for key in M:
		print("Best", M[key]["best"])



	### plot ROC
	plt.figure(figsize=(4,4))
	plt.title("ROC")
	plt.plot(M["lin"]["fpr"], M["lin"]["tpr"], label = "Linear")
	plt.plot(M["NN"]["fpr"], M["NN"]["tpr"], label = "NeuralNet")
	plt.plot(M["NNLin"]["fpr"], M["NNLin"]["tpr"], label = "LinNeuralNet")
	plt.plot([0,1],[0,1],"k--",)
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.xlabel("FPR")
	plt.ylabel("TPR")
	plt.legend()

