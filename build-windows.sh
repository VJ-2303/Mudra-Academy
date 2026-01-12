#!/bin/bash

# Build Helper Script for Mudra Academy
# This script prepares and builds the Windows application

echo "🎯 Mudra Academy - Windows Build Script"
echo "========================================"
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

# Check for icon
if [ ! -f "assets/images/icon.png" ]; then
    echo "⚠️  Warning: No icon found at assets/images/icon.png"
    echo "   The app will use the default Electron icon."
    echo "   To add a custom icon, place a 256x256 PNG at: assets/images/icon.png"
    echo ""
    
    # Temporarily remove icon references from package.json
    echo "   Skipping icon configuration for this build..."
    echo ""
fi

# Ask user what to build
echo "Choose build type:"
echo "1) NSIS Installer (recommended for distribution)"
echo "2) Portable EXE (no installation required)"
echo "3) Both (default)"
echo ""
read -p "Enter choice [1-3] (default: 3): " choice
choice=${choice:-3}

echo ""
echo "🔨 Starting build process..."
echo "   This may take 2-5 minutes on first build."
echo ""

case $choice in
    1)
        echo "Building NSIS Installer..."
        npm run build -- --win nsis
        ;;
    2)
        echo "Building Portable EXE..."
        npm run build:portable
        ;;
    *)
        echo "Building both Installer and Portable versions..."
        npm run build
        ;;
esac

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build completed successfully!"
    echo ""
    echo "📁 Your Windows application is in the 'dist/' folder:"
    echo ""
    ls -lh dist/*.exe 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Test the .exe file on a Windows machine"
    echo "   2. Share with users or deploy"
    echo "   3. Check BUILD_GUIDE.md for more options"
else
    echo ""
    echo "❌ Build failed. Check the errors above."
    echo "   Common issues:"
    echo "   - Not enough disk space"
    echo "   - Missing dependencies (run: npm install)"
    echo "   - Icon file issues (check assets/images/icon.png)"
fi
