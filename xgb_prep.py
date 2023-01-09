#!/usr/bin/python3
import xgboost as xgb
from sklearn import datasets
import onnxmltools
from onnxconverter_common.data_types import FloatTensorType, Int64TensorType

iris = datasets.load_iris()
X_train = iris.data
# dtrain = xgb.DMatrix(iris.data, label = iris.target)
param = {
    'max_depth': 3,
    'eta': 0.3,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 3 # three class 0, 1, 2
}
num_round = 20
print('training')
# bst = xgb.train(param, dtrain, num_round)
bst = xgb.XGBClassifier(**param)
bst.fit(X_train, iris.target)
bst.save_model("iris.xg.model")

print('converting to onnx')
initial_type = [('float_input', FloatTensorType([1, X_train.shape[1]]))]
onnx_model = onnxmltools.convert_xgboost(bst, initial_types=initial_type)
print('saving onnx model')
onnxmltools.utils.save_model(onnx_model, 'mnist_xgb.onnx')