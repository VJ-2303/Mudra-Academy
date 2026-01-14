const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const ort = require('onnxruntime-node');

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
    app.quit();
}

let mainWindow;
let onnxSession = null;

// ================================================================================
// ONNX MODEL CONFIGURATION
// ================================================================================
const MODEL_PATH = path.join(__dirname, '../ml/models/kkkvp6.onnx');
const CLASSES = [
    'Anjali', 'MATSYA', 'Naagabandha', 'SVASTIKA',
    'berunda', 'chakra', 'garuda', 'karkota',
    'katariswastika', 'katva', 'pasha', 'pushpantha',
    'sakata', 'shanka', 'shivalinga', 'utsanga'
];
const CONF_THRESHOLD = 0.35;

// ================================================================================
// ONNX MODEL LOADING
// ================================================================================
async function loadModel() {
    try {
        console.log('[ONNX] Loading model from:', MODEL_PATH);
        onnxSession = await ort.InferenceSession.create(MODEL_PATH);
        console.log('[ONNX] Model loaded successfully');
        console.log('[ONNX] Input names:', onnxSession.inputNames);
        console.log('[ONNX] Output names:', onnxSession.outputNames);
        return true;
    } catch (error) {
        console.error('[ONNX] Failed to load model:', error);
        return false;
    }
}

// ================================================================================
// INFERENCE FUNCTION
// ================================================================================
async function runInference(imageData) {
    if (!onnxSession) {
        return { error: 'Model not loaded' };
    }

    try {
        // imageData is a Float32Array of shape [1, 3, 640, 640]
        const inputTensor = new ort.Tensor('float32', imageData, [1, 3, 640, 640]);
        const feeds = {};
        feeds[onnxSession.inputNames[0]] = inputTensor;

        // Run inference
        const results = await onnxSession.run(feeds);
        const outputTensor = results[onnxSession.outputNames[0]];
        const output = outputTensor.data;
        const dims = outputTensor.dims;

        // Determine shape (standard vs transposed)
        let N, numClasses, transposed;
        if (dims[1] > dims[2]) {
            N = dims[1];
            numClasses = dims[2] - 4;
            transposed = true;
        } else {
            numClasses = dims[1] - 4;
            N = dims[2];
            transposed = false;
        }

        const stride = numClasses + 4;
        let bestScore = 0;
        let bestClass = 0;
        let bestIdx = 0;

        // Find best detection
        for (let i = 0; i < N; i++) {
            for (let c = 0; c < numClasses; c++) {
                let score;
                if (transposed) {
                    score = output[i * stride + (4 + c)];
                } else {
                    score = output[(4 + c) * N + i];
                }

                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c;
                    bestIdx = i;
                }
            }
        }

        if (bestScore < CONF_THRESHOLD) {
            return { detected: false };
        }

        // Extract bounding box (cx, cy, w, h) for the best detection
        let cx, cy, w, h;
        if (transposed) {
            cx = output[bestIdx * stride + 0];
            cy = output[bestIdx * stride + 1];
            w = output[bestIdx * stride + 2];
            h = output[bestIdx * stride + 3];
        } else {
            cx = output[0 * N + bestIdx];
            cy = output[1 * N + bestIdx];
            w = output[2 * N + bestIdx];
            h = output[3 * N + bestIdx];
        }

        return {
            detected: true,
            name: CLASSES[bestClass],
            confidence: bestScore,
            box: { cx, cy, w, h }  // Bounding box in 640x640 space
        };

    } catch (error) {
        console.error('[ONNX] Inference error:', error);
        return { error: error.message };
    }
}

// ================================================================================
// IPC HANDLERS
// ================================================================================
ipcMain.handle('mudra:init', async () => {
    if (onnxSession) {
        return { success: true };
    }
    const success = await loadModel();
    return { success };
});

ipcMain.handle('mudra:detect', async (event, imageData) => {
    return await runInference(imageData);
});

ipcMain.handle('mudra:getClasses', () => {
    return CLASSES;
});

// ================================================================================
// WINDOW CREATION
// ================================================================================
const createWindow = () => {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
            webSecurity: false,
        },
        icon: path.join(__dirname, '../assets/images/logo.png'),
        autoHideMenuBar: true,
    });

    mainWindow.loadFile(path.join(__dirname, '../index.html'));

    if (process.argv.includes('--dev')) {
        mainWindow.webContents.openDevTools();
    }
};

// ================================================================================
// APP LIFECYCLE
// ================================================================================
app.whenReady().then(async () => {
    // Pre-load model on app start for faster first detection
    await loadModel();

    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
