const { app, BrowserWindow, ipcMain, session } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
    app.quit();
}

let mainWindow;
let pythonProcess = null;
let isPythonReady = false;
let detectQueue = [];

const VENV_PYTHON = path.join(__dirname, '../ml/venv/bin/python');
const SCRIPT_PATH = path.join(__dirname, '../ml/mudra_inference.py');

function spawnPythonProcess() {
    return new Promise((resolve, reject) => {
        if (pythonProcess) {
            if (isPythonReady) return resolve(true);
            return setTimeout(() => resolve(spawnPythonProcess()), 100);
        }

        console.log('[PY] Spawning Python process...', VENV_PYTHON);

        try {
            pythonProcess = spawn(VENV_PYTHON, ['-u', SCRIPT_PATH]);

            // Handle Stdout (Data + Ready signal)
            let buffer = '';
            pythonProcess.stdout.on('data', (data) => {
                buffer += data.toString();
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;

                    if (trimmed === 'READY') {
                        console.log('[PY] Python Engine Ready');
                        isPythonReady = true;
                        resolve(true);
                    } else if (trimmed.startsWith('{')) {
                        // JSON response
                        try {
                            const result = JSON.parse(trimmed);
                            // Resolve the oldest request
                            const req = detectQueue.shift();
                            if (req) req.resolve(result);
                        } catch (e) {
                            console.error('[PY] JSON Parse Error:', e);
                        }
                    } else {
                        console.log('[PY] stdout:', trimmed);
                    }
                }
            });

            // Handle Stderr (Logs)
            pythonProcess.stderr.on('data', (data) => {
                const msg = data.toString().trim();
                // Filter out common pytorch log spam if needed
                console.log(`[PY-LOG] ${msg}`);
            });

            pythonProcess.on('error', (err) => {
                console.error('[PY] Failed to start:', err);
                isPythonReady = false;
                reject(err);
            });

            pythonProcess.on('close', (code) => {
                console.log(`[PY] Process exited with code ${code}`);
                pythonProcess = null;
                isPythonReady = false;
            });

        } catch (e) {
            console.error('[PY] Spawn error:', e);
            reject(e);
        }
    });
}

function killPythonProcess() {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
    }
}

ipcMain.handle('mudra:init', async () => {
    try {
        await spawnPythonProcess();
        return { success: true };
    } catch (e) {
        console.error('Init failed:', e);
        return { success: false, error: e.message };
    }
});

// imageData is now a Base64 string from the frontend
ipcMain.handle('mudra:detect', async (event, base64Image) => {
    if (!isPythonReady || !pythonProcess) {
        return { error: 'Python model not ready' };
    }

    return new Promise((resolve) => {
        // Queue the resolution
        detectQueue.push({ resolve });

        // Write to stdin
        // Ensure newline for readline() in python
        try {
            pythonProcess.stdin.write(base64Image + '\n');
        } catch (e) {
            // Handle write error (pipe closed etc)
            const req = detectQueue.pop(); // remove self
            if (req) req.resolve({ error: 'Pipe write error' });
        }
    });
});

ipcMain.handle('mudra:getClasses', () => {
    // Return placeholder - classes are now managed by Python model
    return [];
});

const createWindow = () => {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
            // webSecurity: false, // Not needed unless loading local resource issues occur
        },
        icon: path.join(__dirname, '../assets/images/logo.png'),
        autoHideMenuBar: true,
    });

    mainWindow.loadFile(path.join(__dirname, '../index.html'));

    if (process.argv.includes('--dev')) {
        mainWindow.webContents.openDevTools();
    }
};

app.whenReady().then(async () => {
    // Grant camera and microphone permissions (required for Windows)
    session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
        const allowedPermissions = ['media', 'mediaKeySystem', 'geolocation', 'notifications'];
        if (allowedPermissions.includes(permission)) {
            callback(true);
        } else {
            callback(false);
        }
    });

    spawnPythonProcess().catch(e => console.log('Early spawn failed, will retry on init:', e));

    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    killPythonProcess();
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('will-quit', () => {
    killPythonProcess();
});
