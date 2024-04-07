import {FilesetResolver, HandLandmarker} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
import jsondata from './poseData.json'
with {type: "json"};

const demosSection = document.getElementById("demos");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;
const nn = ml5.neuralNetwork({task: 'classification', debug: true});

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
let posesArray = [];
let live_Data = [];
let pindakaas = []
let lastVideoTime = -1;
let results = undefined;

//controleer op foute data
for (let i = 0; i < jsondata.data.length; i++) {

    if (jsondata.data[i].pose.length === 63) {

        pindakaas.push(jsondata.data[i]);
    } else {
        console.log("data is niet de goede lengte" + jsondata.data[i].name + "nummer" + i);
    }
}


function flattenData(data) {
    return data.reduce((acc, obj) => {
        return acc.concat(Object.values(obj));
    }, []);
}


const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 2
    });
    demosSection.classList.remove("invisible");
};
createHandLandmarker();



if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

// camera aan
function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }

    const cameraOptions = document.getElementById('cameraOptions');
    const constraints = {
        video: {
            deviceId: cameraOptions.value ? {exact: cameraOptions.value} : undefined
        }
    };
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

async function predictWebcam() {
    canvasElement.style.width = video.videoWidth;
    canvasElement.style.height = video.videoHeight;
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({runningMode: "VIDEO"});
    }
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
        if (results.landmarks) {
            for (const landmarks of results.landmarks) {

                for (const point of landmarks) {
                    posesArray.push(point.x);
                    posesArray.push(point.y);
                    posesArray.push(point.z || 0);
                }
                live_Data = flattenData(landmarks);
            }
        }
    }
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: "#00FF00", lineWidth: 5});
            drawLandmarks(canvasCtx, landmarks, {color: "#FF0000", lineWidth: 2});
        }
    }
    canvasCtx.restore();
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
    await live();

}

//// *  input kiezen * ////

navigator.mediaDevices.enumerateDevices()
    .then(devices => {
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        const select = document.getElementById('cameraOptions');
        videoDevices.forEach(videoDevice => {
            const option = document.createElement('option');
            option.value = videoDevice.deviceId;
            option.text = videoDevice.label;
            select.appendChild(option);
        });
    });


//// *  Randomize the data * ////


/********************************************************************
 // training data.
 ********************************************************************/

let data = pindakaas;
data = data.sort(() => Math.random() - 0.5);
const training = data.slice(0, Math.floor(data.length * 0.7));

const start = document.getElementById("startTraining");
start.addEventListener("click", Training);


function Training() {

    for (let i = 0; i < training.length; i++) {
        const {pose, name} = training[i];
        nn.addData(pose, {name});
    }
    nn.normalizeData();
    nn.train({
        epochs: 100,
        learningRate: 0.4,
        hiddenUnits: 16,
    }, () => finishedTraining());

}

//// *  save model * ////
function saveModel(outputName, callback) {
    nn.save("model", () => console.log("model was saved!"))
}


async function finishedTraining() {
    console.log("Finished training!");
    const results = await nn.classify([0.6236324310302734, 0.557339072227478, -8.803996109918444e-8, 0.5791879892349243, 0.5421792268753052, -0.002231871010735631, 0.5438221096992493, 0.5217909812927246, -0.010455181822180748, 0.5135918855667114, 0.517388105392456, -0.022137979045510292, 0.48837047815322876, 0.5175199508666992, -0.03480968624353409, 0.5565288066864014, 0.44220221042633057, 0.0028354476671665907, 0.5180234313011169, 0.4380141794681549, -0.021742412820458412, 0.5031105279922485, 0.4699598252773285, -0.04591277241706848, 0.4974973499774933, 0.4990153908729553, -0.05952564999461174, 0.5699726939201355, 0.42569512128829956, -0.007597567979246378, 0.5389858484268188, 0.3776291310787201, -0.023256467655301094, 0.5122165083885193, 0.35608258843421936, -0.03879261016845703, 0.48655539751052856, 0.34117788076400757, -0.049240972846746445, 0.5897690653800964, 0.42223408818244934, -0.020910363644361496, 0.5763337016105652, 0.36190110445022583, -0.03423679992556572, 0.5572678446769714, 0.329062283039093, -0.04400315135717392, 0.5370699167251587, 0.30241474509239197, -0.04992298036813736, 0.6100855469703674, 0.43401625752449036, -0.034755125641822815, 0.6088380813598633, 0.3883325159549713, -0.0416024848818779, 0.5995600819587708, 0.3579040765762329, -0.04149336367845535, 0.5868493318557739, 0.33129143714904785, -0.041084155440330505
        //okay-pose
    ]);

    console.log("Top result:", results[0].label);
    //save model
    // saveModel( () => {
    //     console.log('Model succesvol opgeslagen');
    // });

}

/********************************************************************
 // Live data.
 ********************************************************************/

//// * loading model * ///

const testload = document.getElementById("load");
testload.addEventListener("click", load);
// load()

async function load() {

    const modelDetails = {
        model: "/model/model.json",
        metadata: "/model/model_meta.json",
        weights: "/model/model.weights.bin"
    }
    try {
        await nn.load(modelDetails);
        console.log("het model is geladen");

    } catch (error) {
        if (error instanceof TypeError) {
            console.error("Er is een TypeError opgetreden bij het laden van het model:", error.message);
        } else {
            console.error("Er is een fout opgetreden bij het laden van het model:", error);
        }
    }

}

//// * start game * ////

const startGameButton = document.getElementById("StartGame");
startGameButton.addEventListener("click", startGame);

let gameStarted = false;
let correctPoses = 0;
let canPose = false;
const game = ['okay-pose', 'rock-pose', 'peace-pose'];

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

function startGame() {
    if (gameStarted) {
        console.log("Het spel is al begonnen");
        return;
    }

    shuffleArray(game);
    displayGame(game[correctPoses]);
    console.log(game[correctPoses]);
    gameStarted = true;
    correctPoses = 0;
    canPose = true;
}

async function live() {
    if (!gameStarted) {
        console.log("Het spel is nog niet begonnen");
        return;
    }
    if (!canPose) {
        // displayErrorMessage("Je moet nog even wachten voordat je een nieuwe pose kan doen");
        return;
    }

    if (canPose) {
        const results = await nn.classify(live_Data);

        if (results[0].label === game[correctPoses]) {
            const gameDisplay = document.getElementById("gamedisplay");
            gameDisplay.textContent = `Goed gedaan, je hebt deze pose nagemaakt: ${results[0].label}`;
            correctPoses++;
            canPose = false;
            setTimeout(() => {
                if (correctPoses === 3) {
                    displaySuccessMessage("Gefeliciteerd je hebt gewonnen");
                    const gameDisplay = document.getElementById("gamedisplay");
                    gameDisplay.textContent = 'Je hebt gewonnen!';
                    gameStarted = false;
                    correctPoses = 0;
                } else {
                    setTimeout(() => {
                        displayGame(game[correctPoses]);
                        canPose = true;
                    }, 2000);
                }
            }, 2000);
        } else {
            console.log("Fout, probeer opnieuw!");
        }
    }
}

function displayGame(game) {
    const gameDisplay = document.getElementById("gamedisplay");
    gameDisplay.textContent = `Maak deze pose na: ${game} je hebt al ${correctPoses} poses goed`;
}

function displayErrorMessage(message) {
    const errorMessage = document.getElementById("foutje");
    errorMessage.textContent = message;
}

function displaySuccessMessage(message) {
    const successMessage = document.getElementById("resultDisplay");
    successMessage.textContent = message;
}

function resetCorrectPoses() {
    correctPoses = 0;
}


/********************************************************************
 // test Data.
 ********************************************************************/


const testing = data.slice(Math.floor(data.length * 0.7));
const goed = testing.sort(() => Math.random() - 0.5)


const test1 = document.getElementById("testrock");
test1.addEventListener("click", matrix);


async function matrix() {
    const classes = ['rock-pose', 'okay-pose', 'peace-pose'];
    const confusionMatrix = Array(classes.length).fill().map(() => Array(classes.length).fill(0));

    for (const testpose of goed) {

        const prediction = await nn.classify(testpose.pose);
        const predictedIndex = classes.indexOf(prediction[0].label);
        const actualIndex = classes.indexOf(testpose.name);

        confusionMatrix[predictedIndex][actualIndex]++;
    }

    const matrixWithLabels = [
        ['Class/Index', ...classes],
        ...confusionMatrix.map((row, index) => [classes[index], ...row])
    ];
    console.table(matrixWithLabels);
    let tableHTML = '<table>';

    matrixWithLabels.forEach(rowData => {
        tableHTML += '<tr>';
        rowData.forEach(cellData => {
            tableHTML += `<td>${cellData}</td>`;
        });
        tableHTML += '</tr>';
    });

    tableHTML += '</table>';

    const resultmatrix = document.getElementById("confusion-matrix");

    //accuracy berekenen
    let correctPredictions = 0;
    for (let i = 0; i < classes.length; i++) {
        correctPredictions += confusionMatrix[i][i];
    }
    const totalPredictions = goed.length;
    const accuracy = (correctPredictions / totalPredictions) * 100;
    resultmatrix.innerHTML = tableHTML + `Nauwkeurigheid: ${accuracy}%`;

}


