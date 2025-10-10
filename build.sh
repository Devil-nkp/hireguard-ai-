#!/bin/bash
# build.sh

echo "Starting build process..."

# Install system dependencies for OCR
echo "Installing system dependencies..."
apt-get update
apt-get install -y tesseract-ocr libtesseract-dev poppler-utils

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install setuptools wheel

# Install requirements
echo "Installing from requirements.txt..."
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

echo "Build completed successfully!"