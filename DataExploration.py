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

"""

### data
data = F.loadPreparedData()
x_train,y_train,x_test,y_test = F.splitData(data)
### correlation
corr = data.corr()["survived"]
F.prettyPrint("Correlation", corr)

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
shap_vals = explainer.shap_values(x_test)
## summary plot
shap.summary_plot(shap_vals,x_test.values,x_train.columns)
## one sample person
i=1
shap.force_plot(explainer.expected_value,shap_vals[i,:],x_test.values[i,:],x_test.columns,matplotlib=True)

shap.force_plot(explainer.expected_value,
				shap_vals,
				x_test.values,
				x_test.columns)

shap.dependence_plot("fare",shap_vals,x_test.values,x_test.columns,interaction_index="isMale")


print("-----------------------------------------")
print(" ######## NN MODEL ########")
print("-----------------------------------------")
### Keras model
NNModel = NN.kerasModel(x_train, y_train)
NNPred = NNModel.predict(x_test, batch_size=None, verbose=0)

## NN shap
NNexplainer = shap.DeepExplainer(NNModel,x_test.values)

""" Why is this with index 0? """
shap_vals = NNexplainer.shap_values(x_test.values)[0]

##plots
# shap.force_plot(NNexplainer.expected_value.numpy(),
# 				shap_vals[i],
# 				x_test.values[i],
# 				x_test.columns,
# 				matplotlib=False)

shap.summary_plot(shap_vals,x_test.values,x_train.columns)
shap.force_plot(NNexplainer.expected_value.numpy(),
				shap_vals,
				x_test.values,
				x_test.columns,
				matplotlib = False)

shap.dependence_plot("fare",shap_vals,x_test.values,x_test.columns,interaction_index="isMale")

print("-----------------------------------------")
print(" ######## NN MODEL ########")
print("-----------------------------------------")
from sklearn import metrics
M = {}
fpr, tpr, thresholds = metrics.roc_curve(y_test, linPred)
M["lin"] = {"fpr" : fpr, "tpr":tpr, "thresholds":thresholds}
fpr, tpr, thresholds = metrics.roc_curve(y_test, NNPred)
M["NN"] = {"fpr" : fpr, "tpr":tpr, "thresholds":thresholds}
### get best tpr and fpr
M["lin"]["best"] = maximizeTPRFPR(M["lin"]["tpr"], M["lin"]["fpr"])
M["NN"]["best"] = maximizeTPRFPR(M["NN"]["tpr"], M["NN"]["fpr"])

plt.figure(figsize=(4,4))
plt.title("ROC")
plt.plot(M["lin"]["fpr"], M["lin"]["tpr"], label = "Linear")
plt.plot(M["NN"]["fpr"], M["NN"]["tpr"], label = "NeuralNet")
plt.plot([0,1],[0,1],"k--",)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()


print("Done.")