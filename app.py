# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import PyPDF2
import pdf2image
import re
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import tempfile
import os
import io
import base64
import easyocr
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'}

# Initialize components
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("spaCy model not found. Using limited NLP features.")
    nlp = None

# Initialize EasyOCR reader
try:
    easyocr_reader = easyocr.Reader(['en'])
except Exception as e:
    logger.error(f"EasyOCR initialization failed: {e}")
    easyocr_reader = None

class UltraStrongOCRProcessor:
    def __init__(self):
        self.tesseract_configs = [
            '--oem 3 --psm 6',
            '--oem 3 --psm 1',
            '--oem 3 --psm 3',
            '--oem 3 --psm 4',
            '--oem 3 --psm 8',
            '--oem 3 --psm 11',
        ]

    def super_preprocess_image(self, image_path):
        """Apply multiple advanced preprocessing techniques"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            processed_images = []
            height, width = gray.shape
            
            # Resize if too small
            if max(height, width) < 1000:
                scale_factor = 2000 / max(height, width)
                gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

            # Noise reduction
            denoised1 = cv2.fastNlMeansDenoising(gray)
            denoised2 = cv2.medianBlur(gray, 3)
            processed_images.extend([denoised1, denoised2])

            # Thresholding
            _, thresh_otsu = cv2.threshold(denoised1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_adapt = cv2.adaptiveThreshold(denoised1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            processed_images.extend([thresh_otsu, thresh_adapt])

            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(denoised1)
            processed_images.append(clahe_img)

            return processed_images

        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return None

    def extract_text_with_all_engines(self, image_path):
        """Extract text using multiple OCR engines"""
        all_texts = []

        processed_images = self.super_preprocess_image(image_path)
        if processed_images is None:
            return None

        # Tesseract OCR on preprocessed images
        for processed_img in processed_images:
            for config in self.tesseract_configs:
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    if text and len(text.strip()) > 5:
                        all_texts.append(text)
                except Exception as e:
                    continue

        # EasyOCR
        if easyocr_reader:
            try:
                easyocr_results = easyocr_reader.readtext(image_path, detail=0)
                easyocr_text = " ".join(easyocr_results)
                if easyocr_text and len(easyocr_text.strip()) > 5:
                    all_texts.append(easyocr_text)
            except Exception as e:
                logger.error(f"EasyOCR error: {e}")

        # Remove duplicates and clean
        unique_texts = [t for t in set(all_texts) if len(t.strip()) > 5]

        if unique_texts:
            combined_text = "\n".join(unique_texts)
            return self.post_process_text(combined_text)

        return None

    def post_process_text(self, text):
        """Clean and post-process extracted text"""
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        replacements = {
            r'(\w)\.(\w)': r'\1. \2',
            r'\,(\w)': r', \1',
            r'\s+\.': '.',
            r'\s+\,': ',',
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        return text.strip()

    def extract_text_from_pdf(self, pdf_path):
        """Enhanced PDF text extraction"""
        all_text = []

        try:
            # Method 1: pdf2image + OCR
            images = pdf2image.convert_from_path(pdf_path, dpi=300)
            for img in images:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    img.save(tmp.name, 'JPEG', quality=95)
                    text = self.extract_text_with_all_engines(tmp.name)
                    if text:
                        all_text.append(text)
                    os.unlink(tmp.name)

            # Method 2: PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text and len(text.strip()) > 10:
                            all_text.append(text)
            except Exception as e:
                logger.error(f"PyPDF2 error: {e}")

            if all_text:
                combined_text = "\n".join(all_text)
                return self.post_process_text(combined_text)

        except Exception as e:
            logger.error(f"PDF extraction error: {e}")

        return None

class AdvancedJobPostAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.ocr_processor = UltraStrongOCRProcessor()

    def extract_text_from_bytes(self, file_bytes, file_extension):
        """Extract text from file bytes"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name

            if file_extension.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                text = self.ocr_processor.extract_text_with_all_engines(tmp_file_path)
            elif file_extension.lower() == '.pdf':
                text = self.ocr_processor.extract_text_from_pdf(tmp_file_path)
            else:
                text = None

            os.unlink(tmp_file_path)
            return text

        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return None

    def is_job_post(self, text):
        """Check if text is a job posting"""
        if not text or len(text.strip()) < 50:
            return False

        text_lower = text.lower()
        job_keywords = [
            'job', 'position', 'hire', 'hiring', 'apply', 'application', 'role', 
            'career', 'vacancy', 'responsibilities', 'requirements', 'qualifications',
            'experience', 'salary', 'compensation', 'company', 'employer'
        ]

        keyword_count = sum(1 for keyword in job_keywords if keyword in text_lower)
        return keyword_count >= 3

    def extract_features(self, text):
        """Extract features from text"""
        features = {}

        if not text:
            return self._get_default_features()

        # Basic features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text.replace(" ", ""))

        # Sentiment
        sentiment = self.sia.polarity_scores(text)
        features['sentiment_compound'] = sentiment['compound']
        features['sentiment_positive'] = sentiment['pos']
        features['sentiment_negative'] = sentiment['neg']
        features['sentiment_neutral'] = sentiment['neu']

        # Linguistic features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        features['exclamation_count'] = text.count('!')
        features['question_mark_count'] = text.count('?')

        # Job-specific features
        urgency_phrases = ['urgent', 'immediate', 'quick', 'fast', 'ASAP', 'right away']
        features['urgency_score'] = sum(text.lower().count(phrase) for phrase in urgency_phrases)

        money_phrases = ['$', 'salary', 'pay', 'compensation', 'bonus', 'earn']
        features['money_mentions'] = sum(text.lower().count(phrase) for phrase in money_phrases)

        requirement_phrases = ['require', 'must have', 'necessary', 'qualification', 'experience']
        features['requirement_mentions'] = sum(text.lower().count(phrase) for phrase in requirement_phrases)

        # Contact patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(\+\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b'

        features['email_count'] = len(re.findall(email_pattern, text))
        features['phone_count'] = len(re.findall(phone_pattern, text))

        return features

    def _get_default_features(self):
        """Return default feature values"""
        return {
            'text_length': 0, 'word_count': 0, 'char_count': 0,
            'sentiment_compound': 0, 'sentiment_positive': 0, 
            'sentiment_negative': 0, 'sentiment_neutral': 0,
            'uppercase_ratio': 0, 'exclamation_count': 0, 
            'question_mark_count': 0, 'urgency_score': 0,
            'money_mentions': 0, 'requirement_mentions': 0,
            'email_count': 0, 'phone_count': 0
        }

    def analyze_job_post(self, file_bytes, file_extension):
        """Main analysis function"""
        # Extract text
        extracted_text = self.extract_text_from_bytes(file_bytes, file_extension)

        if not extracted_text or len(extracted_text.strip()) < 20:
            return self._create_error_response("OCR failed to extract sufficient text")

        logger.info(f"Text extracted: {len(extracted_text)} characters")

        # Check if job post
        is_job = self.is_job_post(extracted_text)
        if not is_job:
            return self._create_not_job_response(extracted_text)

        # Extract features and analyze
        features = self.extract_features(extracted_text)
        risk_score = self.calculate_risk_score(features)
        risk_factors = self.identify_risk_factors(extracted_text, features)

        is_fake = risk_score > 0.5
        confidence = risk_score if is_fake else 1 - risk_score

        return {
            'success': True,
            'is_job_post': True,
            'is_fake': is_fake,
            'confidence': float(confidence),
            'risk_score': float(risk_score),
            'risk_factors': risk_factors,
            'extracted_text_length': len(extracted_text),
            'extracted_text_preview': extracted_text[:1500] + "..." if len(extracted_text) > 1500 else extracted_text,
            'features': features
        }

    def calculate_risk_score(self, features):
        """Calculate fraud risk score"""
        risk_score = 0

        # Scoring rules
        if features['urgency_score'] > 3:
            risk_score += 0.3
        if features['money_mentions'] > 5:
            risk_score += 0.2
        if features['requirement_mentions'] < 2:
            risk_score += 0.2
        if features['email_count'] > 2 or features['phone_count'] > 2:
            risk_score += 0.2
        if features['word_count'] < 100:
            risk_score += 0.2
        if features['exclamation_count'] > 5:
            risk_score += 0.1
        if features['uppercase_ratio'] > 0.3:
            risk_score += 0.1

        return min(risk_score, 0.95)

    def identify_risk_factors(self, text, features):
        """Identify specific risk factors"""
        risk_factors = []

        # Feature-based risks
        if features['urgency_score'] > 3:
            risk_factors.append("Urgency phrases detected")
        if features['money_mentions'] > 5:
            risk_factors.append("Excessive focus on money")
        if features['requirement_mentions'] < 2:
            risk_factors.append("Vague or missing requirements")
        if features['email_count'] > 2 or features['phone_count'] > 2:
            risk_factors.append("Suspicious contact information")
        if features['word_count'] < 100:
            risk_factors.append("Unusually short job description")
        if features['exclamation_count'] > 5:
            risk_factors.append("Excessive exclamation marks")
        if features['uppercase_ratio'] > 0.3:
            risk_factors.append("Excessive uppercase text")

        # Pattern-based risks
        patterns = [
            (r'work from home', "Vague work-from-home promises"),
            (r'no experience', "No experience required claims"),
            (r'easy money', "Get-rich-quick language"),
            (r'pay.*fee', "Request for payment"),
            (r'wire transfer', "Request for wire transfer"),
            (r'immediate joining', "Immediate joining required"),
        ]

        for pattern, description in patterns:
            if re.search(pattern, text.lower()):
                risk_factors.append(description)

        return risk_factors if risk_factors else ["No obvious risk factors detected"]

    def _create_error_response(self, message):
        return {
            'success': False,
            'error': message,
            'is_job_post': False,
            'is_fake': 'Unknown',
            'confidence': 0.0
        }

    def _create_not_job_response(self, extracted_text):
        return {
            'success': True,
            'is_job_post': False,
            'is_fake': 'Not a Job Post',
            'confidence': 0.95,
            'extracted_text_length': len(extracted_text),
            'extracted_text_preview': extracted_text[:1500] + "..." if len(extracted_text) > 1500 else extracted_text,
            'risk_factors': ['Content does not appear to be a job posting']
        }

# Initialize analyzer
analyzer = AdvancedJobPostAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_job_post():
    """API endpoint for job post analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'}), 400

        # Read file
        file_bytes = file.read()
        file_extension = os.path.splitext(file.filename)[1].lower()

        # Analyze
        result = analyzer.analyze_job_post(file_bytes, file_extension)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'HireGuard AI API is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)