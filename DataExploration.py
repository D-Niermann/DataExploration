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

### Linear Model
linModel, w = F.linRegression(x_train, y_train, x_test, y_test)
scoreLin = F.score(linModel.predict(x_test), y_test)
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


print("Done.")