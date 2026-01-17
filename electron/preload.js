const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
    // IPC communication
    send: (channel, data) => {
        ipcRenderer.send(channel, data);
    },
    receive: (channel, func) => {
        ipcRenderer.on(channel, (event, ...args) => func(...args));
    },

    // Platform info
    platform: process.platform,
    isElectron: true,

    // ================================================================================
    // MUDRA DETECTION API (uses onnxruntime-node in main process)
    // ================================================================================

    /**
     * Initialize the mudra detection model in main process
     * @returns {Promise<{success: boolean}>}
     */
    mudraInit: () => ipcRenderer.invoke('mudra:init'),

    /**
     * Run mudra detection on image data
     * @param {Float32Array} imageData - Preprocessed image tensor [1, 3, 640, 640]
     * @returns {Promise<{detected: boolean, name?: string, confidence?: number, error?: string}>}
     */
    mudraDetect: (imageData) => ipcRenderer.invoke('mudra:detect', imageData),

    /**
     * Get list of class names
     * @returns {Promise<string[]>}
     */
    mudraGetClasses: () => ipcRenderer.invoke('mudra:getClasses'),
});
