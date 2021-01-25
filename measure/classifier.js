//classifier.js based on https://towardsdatascience.com/deploying-an-image-classifier-using-javascript-84da1480b3a4
var model;
var predResult = document.getElementById("result");
async function initialize() {
    model = await tf.loadLayersModel('customvision/tf-webmodel/model.json');
}
async function predict() {
  // action for the submit button
let image = document.getElementById("img")
let tensorImg = tf.browser.fromPixels(image).resizeNearestNeighbor([150, 150]).toFloat().expandDims();
  prediction = await model.predict(tensorImg).data();
if (prediction[0] === 0) {
      predResult.innerHTML = "I think it's a cat";
} else if (prediction[0] === 1) {
      predResult.innerHTML = "I think it's a dog";
} else {
      predResult.innerHTML = "This is Something else";
  }
}
initialize();
predict(); // me added
