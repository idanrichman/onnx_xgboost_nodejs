# Quick Start - Nodejs Binding

This example is a demonstration of basic usage of ONNX Runtime Node.js binding for xgboost.

This example contians a `package.json` file, which already lists "onnxruntime-node" as dependency. To work on your own `package.json`, use command `npm install onnxruntime-node` to install ONNX Runtime Node.js binding.

In this example, we load onnxruntime, create an inference session with a simple model, feed input, get output as result and write it to standard output. All functions are called in their basic form.

## Usage

```sh
pip install -r requirements.txt
python xgb_prep.py
npm install
node .
```

tested with miniconda3 + python 3.9