// ⭐️⭐️⭐️ SNAP PHOTO classification
// thanks https://davidwalsh.name/browser-camera
let video = document.getElementById('video');
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({
        video: true
    }).then(function (stream) {
        video.srcObject = stream;
        video.play();
    });
}
// Elements for taking the snapshot
let canvas = document.getElementById('imageCaptured');
let context = canvas.getContext('2d');
let buttonSnap = document.getElementById('snap');
// Trigger photo take
buttonSnap.addEventListener("click", function () {
    deletePreviousResult();
    startCalcAnimation(buttonSnap);
    context.drawImage(video, 0, 0); //context.drawImage(video, 0, 0, 640, 480);
    preprocessStaticImage(canvas);
});


// ⭐️⭐️⭐️ FROM FILE classification
// https://educity.app/web-development/how-to-upload-and-draw-an-image-on-html-canvas
let myCanvas = document.getElementById('imageUploaded'); // Creates a canvas object
let myContext = myCanvas.getContext("2d"); // Creates a contect object
let buttonUpload = document.getElementById('upload');
document.getElementById('imageInput').addEventListener('change', function (e) {
    deletePreviousResult();
    // startCalcAnimation(buttonUpload);
    // document.getElementById('upload').className = "none";
    // document.getElementById('fakeInputLoading').className = "loading"; // start calc anitmation
    if (e.target.files) {
        let imageFile = e.target.files[0]; //here we get the image file
        let reader = new FileReader();
        reader.readAsDataURL(imageFile);
        reader.onloadend = function (e) {
            let myImage = new Image(); // Creates image object
            myImage.src = e.target.result; // Assigns converted image to image object
            myImage.onload = function (ev) {
                myCanvas.width = myImage.width; // Assigns image's width to canvas
                myCanvas.height = myImage.height; // Assigns image's height to canvas
                myContext.drawImage(myImage, 0, 0); // Draws the image on canvas
                preprocessStaticImage(myCanvas);
            }
        }
    }
});

// ********** CROP INPUT IMAGE using OPENCV
function preprocessStaticImage(imgToPreprocess) {
    const utils = new Utils();
    let faceCascadeFile = 'haarcascade_frontalface_default.xml';
    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
        let src = cv.imread(imgToPreprocess);
        let gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
        let faces = new cv.RectVector();
        let faceCascade = new cv.CascadeClassifier();
        // load pre-trained classifiers
        faceCascade.load(faceCascadeFile);
        // detect faces
        let msize = new cv.Size(0, 0);
        faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

        // check is face detected. If not return error.
        if (faces.size() == 1) {
            console.log('Face detected');
            let roiGray = gray.roi(faces.get(0));
            let roiSrc = src.roi(faces.get(0));
            let point1 = new cv.Point(faces.get(0).x, faces.get(0).y);
            let point2 = new cv.Point(faces.get(0).x + faces.get(0).width, faces.get(0).y + faces.get(0).height);
            cv.rectangle(src, point1, point2, [255, 0, 0, 255],
                2); // cv2.rectangle(image, start_point, end_point, color, thickness)
            // let dsize = new cv.Size(src.rows, src.cols);
            // console.log(dsize);
            roiGray.delete();
            roiSrc.delete();
            let rectX = point1.x + 2;
            let rectY = point1.y + 2;
            let rectSize = (point2.x - point1.x) - 4;
            // check if the image is at least 224 in size. If not return error.
            if (rectSize >= 224) {
                let dst = new cv.Mat();
                let rect = new cv.Rect(rectX, rectY, rectSize, rectSize);
                dst = src.roi(rect);
                cv.imshow('imageFaceDetected', src); // image with rectangle
                cv.imshow('imageCropped', dst); // cropped image
                src.delete();
                gray.delete();
                faceCascade.delete();
                cv.FS_unlink(faceCascadeFile);
                myCanvas.className = "inputCanvasVisible";
                classifyStaticImage();
            } else {
                console.log('Face too small. Get closer.');
                document.getElementById('result').innerText = `Face too small. Get closer.`;
                cv.FS_unlink(faceCascadeFile);
                myCanvas.className = 'none';
                stopCalcAnimation();
            }
        } else {
            console.log('No face detected');
            document.getElementById('result').innerText = `No face detected.`;
            cv.FS_unlink(faceCascadeFile);
            myCanvas.className = 'none';
            stopCalcAnimation();
        }
    });
}

// ********** CLASSIFY IMAGE AND PRINT RESULTS
// @ ANT example https://www.alibabacloud.com/blog/tensorflow-js-helps-recognize-large-quantities-of-icons-in-milliseconds_597000
async function classifyStaticImage() {
    // ********** LOAD AND INIT
    const imageToClassify = imageCropped;
    const model = await tf.loadGraphModel('210126a_TFjs/model.json');
    const IMAGE_SIZE = 224;
    const LABELS = ['high', 'low', 'medium'];

    // this is used to match the prediction results and labels
    function findIndicesOfMax(inp, count) {
        const outp = [];
        for (let i = 0; i < inp.length; i += 1) {
            outp.push(i); // add index to output array
            if (outp.length > count) {
                outp.sort((a, b) => inp[b] - inp[a]); // descending sort the output array
                outp.pop(); // remove the last index (index of smallest element in output array)
            }
        }
        return outp;
    }

    // ********** MAKE TENSOR FROM IMAGE AND PREPROCESS
    // Convert images into tensors
    const img = tf.browser.fromPixels(imageToClassify).toFloat();
    const offset = tf.scalar(127.5);
    // Normalize an image from [0, 255] to [-1, 1]
    const normalized = img.sub(offset).div(offset);
    // Change the image size
    let resized = normalized;
    if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
        const alignCorners = true;
        resized = tf.image.resizeBilinear(
            normalized, [IMAGE_SIZE, IMAGE_SIZE], alignCorners,
        );
    }
    // Change the shape of a tensor to meet the model requirements
    const batched = resized.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 3]);


    // ********** RUN INFERENCE
    let pred = model.predict(batched).squeeze().arraySync();
    // console.log(pred);
    // Find the categories with the highest matching degree
    const predictions = findIndicesOfMax(pred, 3).map(i => ({
        className: LABELS[i],
        confidence: Math.round(pred[i] * 1000) / 1000,
    }));


    // ********** PRINT THE RESULTS
    console.log('Inference results: ', predictions);
    // find and print object with highest confidence score
    var res = Math.max.apply(Math, predictions.map(function (o) {
        return o.confidence;
    }))
    var obj = predictions.find(function (o) {
        return o.confidence == res;
    })
    console.log('Top prediction: ', obj);
    document.getElementById('result').innerText = `Detected state is ${obj.className}.`;
    // document.getElementById('result').innerText = `prediction: ${obj.className}\n confidence: ${obj.confidence}`;
    // print detailed results
    document.getElementById('resultJSON').innerText = 'Result JSON: ' + JSON.stringify(predictions); 
    //JSON.stringify(predictions, null, 4)
    // document.getElementById('result').innerText = `prediction: ${predictions[0].className}\n confidence: ${predictions[0].confidence}`;
    // save image to imgbb. thanks https://stackoverflow.com/a/59551120
    var base64img = imageToClassify.toDataURL().split(",")[1];
    data = new FormData()
    data.set('key', 'e5d552739119c31425f04a7885ffe04d')
    data.set('name', obj.className + obj.confidence + "QQQQ" + predictions[0].className + predictions[0]
        .confidence + predictions[1].className + predictions[1].confidence + predictions[2].className + predictions[
            2].confidence)
    data.set('image', base64img)
    let request = new XMLHttpRequest();
    request.open("POST", 'https://api.imgbb.com/1/upload', true);
    request.send(data);
    stopCalcAnimation();
}

function deletePreviousResult() {
    document.getElementById('result').innerText = ``;
    document.getElementById('resultJSON').innerText = ``;
    myCanvas.className = 'none';
    function clearCanvas(canvasToClear) {
        let fDet = document.getElementById(canvasToClear);
        let fDetCon = fDet.getContext("2d");
        fDetCon.clearRect(0, 0, fDet.width, fDet.height);
    }
    clearCanvas('imageUploaded');
    clearCanvas('imageFaceDetected');
    clearCanvas('imageCropped');
}

function startCalcAnimation(buttonToAnimate) {
    buttonToAnimate.className = "button loading";
    buttonToAnimate.innerHTML = "<i class='fas fa-atom fa-spin'></i> Computing";
}

function stopCalcAnimation() {
    buttonSnap.className = "";
    buttonSnap.innerHTML = "Snap Photo";
}
