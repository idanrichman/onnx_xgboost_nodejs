// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const ort = require('onnxruntime-node');

// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create a new session and load the specific model.
        //
        // the model in this example contains a single MatMul node
        // it has 2 inputs: 'float_input'(float32, 1x4)
        // it has 2 outputs: 'label'(float32, 1x1), 'probabilities'(float32, 1x3)
        const session = await ort.InferenceSession.create('./mnist_xgb.onnx');

        // prepare inputs. a tensor need its corresponding TypedArray as data
        // try this [5.1, 3.5, 1.4, 0.2] to get class 0
        // try this [5.6, 3. , 4.5, 1.5] to get class 1
        // try this [6.3, 3.3, 6.0 , 2.5] to get class 2
        const dataA = Float32Array.from([5.6, 3. , 4.5, 1.5]);
        const tensorA = new ort.Tensor('float32', dataA, [1, 4]);

        // prepare feeds. use model input names as keys.
        const feeds = { float_input: tensorA};

        // feed inputs and run
        const results = await session.run(feeds);

        // read from results
        const infer_label = results.label.data;
        const infer_proba = results.probabilities.data;
        console.log(`data of result tensor 'c': ${infer_label}, prob: ${infer_proba}`);

    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

main();
