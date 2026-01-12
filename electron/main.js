const { app, BrowserWindow, ipcMain, protocol } = require('electron');
const path = require('path');

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
    app.quit();
}

let mainWindow;

const createWindow = () => {
    // Create the browser window.
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
            webSecurity: false, // Allow loading local files
        },
        icon: path.join(__dirname, '../assets/images/logo.png'),
        autoHideMenuBar: true,
    });

    // Load the index.html of the app.
    mainWindow.loadFile(path.join(__dirname, '../index.html'));

    // Open DevTools in development
    if (process.argv.includes('--dev')) {
        mainWindow.webContents.openDevTools();
    }
};

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
    createWindow();

    // On macOS it's common to re-create a window when the dock icon is clicked
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

// Quit when all windows are closed, except on macOS.
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
