// from https://codelabs.developers.google.com/codelabs/tensorflowjs-teachablemachine-codelab/index.html#6
const webcamElement = document.getElementById('webcam');

let net;

async function app() {
  console.log('Loading model..');

  // Load the model.
  net = await mobilenet.load();
  // net = await tf.loadGraphModel('http://localhost:8000/customvision/new_model_TFjs/model.json');
  console.log('Successfully loaded model');
  // const img2 = 'img-test.png';
  // const output2 = net.predict(img2);
  // console.log(output2);

  // Create an object from Tensorflow.js data API which could capture image
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);
  while (true) {
    const img = await webcam.capture();
    const result = await net.classify(img);
    // const output = net.predict(img);
    // console.log(output);

    document.getElementById('console').innerText = `prediction: ${result[0].className}\n confidence: ${result[0].probability}`;
    // document.getElementById('console2').innerText = `prediction: ${output}`;
    // Dispose the tensor to release the memory.
    img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

app();
