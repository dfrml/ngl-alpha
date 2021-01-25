// from https://codelabs.developers.google.com/codelabs/tensorflowjs-teachablemachine-codelab/index.html#6
const webcamElement = document.getElementById('webcam');

let net;

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  // net = await tf.loadGraphModel('http://localhost:8000/customvision/tf-webmodel/model.json');
  console.log('Successfully loaded model');

  // Create an object from Tensorflow.js data API which could capture image
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);
  while (true) {
    const img = await webcam.capture();
    const result = await net.classify(img);

    document.getElementById('console').innerText = `
      prediction: ${result[0].className}\n
      probability: ${result[0].probability}
    `;
    // Dispose the tensor to release the memory.
    img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

app();

// // custom model https://www.tensorflow.org/js/tutorials/conversion/import_keras
// const model = await tf.loadLayersModel('model.json');
// console.log('Init');
// const example = tf.fromPixels(webcamElement);  // for example
// const prediction = model.predict(example);
