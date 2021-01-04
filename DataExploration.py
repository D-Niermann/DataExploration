import seaborn
import numpy as np
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
softmax last layer ausprobieren
lin nn model
"""

### data
data = F.loadPreparedData()
### norm each feature to 1 (makes the weights comparable - not necessary with shap, but nice to have)
data = data/data.max(axis=0)
## append sqrt(fare) to better the distribution
data["sqrt(fare)"] = np.sqrt(data["fare"])
## test train split
x_train,y_train,x_test,y_test = F.splitData(data)
### correlation
corr = data.corr()["survived"]
F.prettyPrint("Correlation", corr.sort_values(key=lambda x: abs(x), ascending=False))

### Linear Model
linModel, w = F.linRegression(x_train, y_train, x_test, y_test)
linPred = linModel.predict(x_test)
scoreLin = F.score(linPred, y_test)
print("Score:" , np.round(scoreLin,3))


print("-----------------------------------------")
print(" ######## LIN MODEL ########")
print("-----------------------------------------")
## linear shap
shap.initjs()
explainer = shap.LinearExplainer(linModel,x_train,feature_dependence="independent")
shap_valsLin = explainer.shap_values(x_test)
## summary plot
shap.summary_plot(shap_valsLin,x_test.values,x_train.columns,alpha=0.5)
## one sample person
i=1
shap.force_plot(explainer.expected_value,shap_valsLin[i,:],x_test.values[i,:],x_test.columns,matplotlib=True)

shap.force_plot(explainer.expected_value,
				shap_valsLin,
				x_test.values,
				x_test.columns)

shap.dependence_plot("fare",shap_valsLin,x_test.values,x_test.columns,interaction_index="isMale",alpha=0.5)
plt.title("Linear Model")




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

shap.dependence_plot("sqrt(fare)",shap_valsNN,x_test.values,x_test.columns,interaction_index="fare",alpha=0.5)
plt.title("NN Model")



print("-----------------------------------------")
print(" ######## NNModel Linear ########")
print("-----------------------------------------")
NNModelLin = NN.kerasModel(x_train, y_train,  [n_inputs, n_labels])
NNPredLin : np.array = NNModelLin.predict(x_test, batch_size=None, verbose=0)
NNexplainer = shap.DeepExplainer(NNModelLin,x_test.values)
""" Why is this with index 0? -> for multiple outputs? """
shap_valsNNLin = NNexplainer.shap_values(x_test.values)[0]
shap.summary_plot(shap_valsNNLin,x_test.values,x_train.columns,alpha=0.5)





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

