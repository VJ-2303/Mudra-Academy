# Windows Build Guide - Mudra Academy

## Prerequisites

1. **Node.js and npm** - Already installed
2. **electron-builder** - Will be installed with `npm install`

## Build Steps

### 1. Install Dependencies

```bash
npm install
```

This will install `electron-builder` and all other dependencies.

### 2. Prepare Icon (Optional but Recommended)

Create an icon for your app:
- Place a 256x256 PNG file at: `assets/images/icon.png`
- Or create a 512x512 PNG for better quality
- You can use online tools like https://icon-icons.com/ or https://www.canva.com/

**If you don't have an icon:** The build will still work, it will just use the default Electron icon.

### 3. Create LICENSE File (Optional)

```bash
echo "MIT License - Add your license text here" > LICENSE
```

### 4. Build for Windows

#### Option A: NSIS Installer (Recommended for Distribution)
```bash
npm run build
```

This creates:
- `dist/Mudra Academy-Setup-1.0.0.exe` - Traditional installer (~100-150 MB)
- Users can install to Program Files with desktop shortcuts

#### Option B: Portable Version (No Installation Required)
```bash
npm run build:portable
```

This creates:
- `dist/Mudra Academy-Portable-1.0.0.exe` - Single executable (~100-150 MB)
- Runs directly, no installation needed
- Perfect for USB drives or quick sharing

#### Option C: Build Both
```bash
npm run build
```

The configuration is set to build both installer and portable versions.

### 5. Find Your Built Application

After building, check the `dist/` folder:

```bash
ls -lh dist/
```

You'll see files like:
- `Mudra Academy-Setup-1.0.0.exe` - Installer
- `Mudra Academy-Portable-1.0.0.exe` - Portable version

## File Size Optimization

Your app will be approximately **130-160 MB** due to:
- Electron runtime: ~60 MB (compressed)
- ONNX models: ~73 MB (kkpv_web.onnx 12MB + mudra_rf_model.onnx 61MB)
- Dependencies: ~10-20 MB

### Current Optimizations Applied:

1. **Maximum Compression** - `compression: "maximum"`
2. **ASAR Archive** - All files packed into single archive
3. **Excluded Files** - Source maps and unnecessary files excluded
4. **Models Unpacked** - ONNX models kept separate for faster loading

### Further Size Reduction Options:

#### Option 1: Quantize ONNX Models (Recommended)
Reduce model precision from float32 to float16 or int8:

```python
# Python script to quantize (run this separately if needed)
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "ml/models/mudra_rf_model.onnx",
    "ml/models/mudra_rf_model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

This can reduce the 61MB Random Forest model to ~15-20 MB with minimal accuracy loss.

#### Option 2: Use 7-Zip LZMA Compression
The NSIS installer supports ultra compression:

Update `package.json` build section:
```json
"nsis": {
  "differentialPackage": false,
  "compression": "7z"
}
```

#### Option 3: External Models (Advanced)
Keep models separate and download on first run:
- Initial download: ~60 MB
- Models download: ~73 MB on demand
- Trade-off: Requires internet on first launch

## Build on Different Platforms

### Building Windows App from Linux (Your Current Setup)

✅ **Works!** electron-builder supports cross-platform builds.

You might need Wine for some advanced features:
```bash
# Optional, only if you see errors
sudo apt-get install wine64
```

### Building for 32-bit Windows

```bash
npm run build:all
```

This builds both 64-bit and 32-bit versions.

## Troubleshooting

### Error: "icon.png not found"
- Either create the icon at `assets/images/icon.png`
- Or remove the icon lines from package.json build config

### Build is too slow
- First build takes longer (downloads Electron binaries)
- Subsequent builds are much faster
- Use `--dir` flag for faster testing: `electron-builder --win --dir`

### Out of disk space
- Clear electron-builder cache: `rm -rf ~/.cache/electron-builder`
- Ensure at least 2 GB free space

## Testing the Build

### Before Distribution:
1. Copy the `.exe` to a clean Windows machine
2. Test installation (for NSIS installer)
3. Run the portable version
4. Check all features work:
   - Single-hand detection
   - Double-hand detection
   - Ghost mode
   - Chatbot
   - Library and games

### Performance Check:
- App startup time
- Model loading time
- Detection accuracy
- Memory usage

## Distribution

### Installer Version (NSIS):
- **Best for**: End users who want traditional installation
- **File size**: ~130-160 MB
- **Pros**: Professional, creates shortcuts, adds to Programs list
- **Cons**: Requires admin rights for installation

### Portable Version:
- **Best for**: Quick sharing, USB drives, no-install scenarios
- **File size**: ~130-160 MB
- **Pros**: No installation, no admin rights needed
- **Cons**: No shortcuts, no automatic updates

## Build Statistics

Expected build output:
```
• Building Windows x64 installer
• Packing app files...
• Compressing files (maximum)...
• Creating NSIS installer...
✓ Built: dist/Mudra Academy-Setup-1.0.0.exe (145 MB)
✓ Built: dist/Mudra Academy-Portable-1.0.0.exe (142 MB)
```

Build time:
- First build: 3-5 minutes
- Subsequent builds: 1-2 minutes

## Next Steps

1. Run `npm install` to get electron-builder
2. Create an icon (optional)
3. Run `npm run build`
4. Test the built application
5. Share your app!

## Auto-Update (Future Enhancement)

For production apps, consider adding auto-update:
```bash
npm install electron-updater
```

This allows users to automatically receive updates without reinstalling.
