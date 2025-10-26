#!/bin/bash
# Repository Readiness Check

echo "🔍 Checking Repository Organization..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 (missing)"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        return 0
    else
        echo -e "${RED}✗${NC} $1/ (missing)"
        return 1
    fi
}

echo "📁 Directory Structure:"
check_dir "streamlit_app"
check_dir "docs"
check_dir "notebooks"
check_dir "scripts"
check_dir ".streamlit"
echo ""

echo "📄 Essential Files:"
check_file "README.md"
check_file "QUICKSTART.md"
check_file "DEPLOYMENT_GUIDE.md"
check_file "requirements.txt"
check_file "packages.txt"
check_file ".gitignore"
check_file "LICENSE"
check_file ".streamlit/config.toml"
echo ""

echo "🎯 App Files:"
check_file "streamlit_app/app.py"
check_file "streamlit_app/plate_detector.py"
check_file "streamlit_app/ocr_module.py"
check_file "streamlit_app/requirements.txt"
echo ""

echo "📦 Models (Should be uploaded to GitHub Releases):"
if [ -f "license_plate_best.pt" ]; then
    SIZE=$(du -h "license_plate_best.pt" | cut -f1)
    echo -e "${YELLOW}⚠${NC} license_plate_best.pt ($SIZE) - Upload to GitHub Releases!"
else
    echo -e "${GREEN}✓${NC} license_plate_best.pt (not in repo, good!)"
fi

if [ -f "best_ocr_model.pth" ]; then
    SIZE=$(du -h "best_ocr_model.pth" | cut -f1)
    echo -e "${YELLOW}⚠${NC} best_ocr_model.pth ($SIZE) - Upload to GitHub Releases!"
else
    echo -e "${GREEN}✓${NC} best_ocr_model.pth (not in repo, good!)"
fi
echo ""

echo "🚫 Checking .gitignore..."
if grep -q "*.pth" .gitignore && grep -q "*.pt" .gitignore; then
    echo -e "${GREEN}✓${NC} Model files in .gitignore"
else
    echo -e "${RED}✗${NC} Model files not in .gitignore!"
fi

if grep -q "uploads/" .gitignore && grep -q "plate_crops/" .gitignore; then
    echo -e "${GREEN}✓${NC} Generated files in .gitignore"
else
    echo -e "${RED}✗${NC} Generated files not in .gitignore!"
fi
echo ""

echo "📊 Repository Size:"
REPO_SIZE=$(du -sh . 2>/dev/null | cut -f1)
echo "Total: $REPO_SIZE"
echo ""

echo "📝 Files to be committed:"
if [ -d ".git" ]; then
    git status --short | head -20
    UNTRACKED=$(git status --short | wc -l)
    echo ""
    echo "Files: $UNTRACKED"
else
    echo -e "${YELLOW}⚠${NC} Git not initialized yet"
    echo "Run: git init"
fi
echo ""

echo "🎯 Next Steps:"
echo "1. Upload models to GitHub Releases"
echo "2. Update README.md (replace YOUR_USERNAME)"
echo "3. Test app locally: cd streamlit_app && streamlit run app.py"
echo "4. Push to GitHub: git add . && git commit -m 'Initial commit' && git push"
echo "5. Deploy to Streamlit Cloud"
echo ""

echo "✅ Check complete! See DEPLOYMENT_GUIDE.md for details."
