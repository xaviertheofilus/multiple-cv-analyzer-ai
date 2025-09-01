# 🚀 Smart ATS Analyzer Pro - Complete Setup Guide

![Smart ATS Analyzer Pro](https://img.shields.io/badge/ATS-Analyzer-blue?style=for-the-badge&logo=python)

## 🌟 Features Overview

### ✨ Enhanced Features
- **Multi-Resume Upload**: Process up to 10 CV files simultaneously (1GB total limit)
- **Real-time Dashboard**: Interactive analytics with Plotly visualizations
- **AI-Powered Analysis**: Google Gemini API for intelligent resume evaluation
- **TensorFlow Integration**: Advanced text similarity matching using TF-IDF
- **Professional UI**: Modern, responsive Streamlit interface
- **Comprehensive Reports**: Export to Excel and PDF formats
- **Batch Processing**: Efficient handling of multiple documents
- **Performance Ranking**: Compare and rank all uploaded resumes

### 📊 Dashboard Features
- Real-time metrics and KPIs
- Interactive charts and visualizations
- Comparative analysis between resumes
- Performance ranking system
- Export capabilities for reports

## 🛠️ Prerequisites

Before starting, ensure you have:
- Python 3.8+ installed
- Google Cloud account (for Gemini API)
- Git (optional, for cloning)
- 4GB+ RAM (recommended for TensorFlow)

## 📥 Installation Guide

### Step 1: Download/Clone the Project

**Option A: Download ZIP**
1. Download the project files
2. Extract to your desired directory

**Option B: Clone Repository**
```bash
git clone <repository-url>
cd smart-ats-analyzer-pro
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv ats_env

# Activate virtual environment
# Windows:
ats_env\Scripts\activate

# macOS/Linux:
source ats_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter issues, install packages individually:
pip install streamlit==1.28.1
pip install google-generativeai==0.3.0
pip install PyPDF2==3.0.1
pip install python-dotenv==1.0.0
pip install fpdf2==2.7.6
pip install pandas==2.0.3
pip install plotly==5.17.0
pip install openpyxl==3.1.2
pip install xlsxwriter==3.1.9
pip install python-multipart==0.0.6
pip install tensorflow==2.13.0
pip install scikit-learn==1.3.0
pip install numpy==1.24.3
```

### Step 4: Configure Environment Variables

1. Create a `.env` file in the project root:

```bash
# Windows
echo. > .env

# macOS/Linux
touch .env
```

2. Add your Google API key to the `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## 🔑 Google Gemini API Setup

### Getting Your Google Gemini API Key

1. **Access Google AI Studio**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account

2. **Create API Key**
   - Click "Create API Key"
   - Select "Create API key in new project" or use existing project
   - Copy the generated API key

3. **Enable Required APIs** (if using Google Cloud Console)
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to "APIs & Services" > "Library"
   - Search for "Generative Language API" and enable it

4. **Add to Environment File**
   ```env
   GOOGLE_API_KEY=AIza...your_actual_key_here
   ```

### API Usage Limits
- Free tier: 15 requests per minute
- For production use, consider upgrading to paid tier
- Monitor usage in Google Cloud Console

## 🚀 Running the Application

### Start the Application

```bash
# Make sure virtual environment is activated
streamlit run app.py
```

### Access the Application

1. Open your browser
2. Navigate to `http://localhost:8501`
3. The application will load with the enhanced interface

## 📱 Using the Application

### 1. Upload & Analyze Tab

1. **Job Description**
   - Paste the complete job description in the text area
   - Include all required skills, qualifications, and responsibilities

2. **Resume Upload**
   - Click "Browse files" or drag & drop PDF files
   - Upload up to 10 resume files (max 1GB total)
   - Supported format: PDF only

3. **Analysis**
   - Click "🔍 Analyze All Resumes"
   - Progress bar shows analysis status
   - Results are stored for dashboard and reports

### 2. Dashboard Tab

- **Metrics Overview**: Average scores and totals
- **ATS Score Comparison**: Bar chart comparing all resumes
- **Performance Radar**: Top 3 resumes comparison
- **Detailed Ranking**: Sortable table with all results

### 3. Reports Tab

- **Excel Report**: Comprehensive spreadsheet with multiple sheets
- **PDF Report**: Formatted document with detailed analysis
- **Download Options**: Generated reports with timestamps

## 🔧 Advanced Configuration

### TensorFlow Optimization

For better performance with TensorFlow:

```bash
# Install GPU support (if you have NVIDIA GPU)
pip install tensorflow-gpu==2.13.0

# For Apple Silicon Macs
pip install tensorflow-metal
```

### Memory Optimization

For large file processing, adjust memory settings:

```python
# In app.py, you can modify the TF-IDF parameters
vectorizer = TfidfVectorizer(
    stop_words='english', 
    max_features=3000,  # Reduce for lower memory usage
    ngram_range=(1, 2)  # Include bigrams
)
```

### Custom Styling

Modify the CSS in the app.py file to customize appearance:

```python
# Find the st.markdown with custom CSS and modify colors, fonts, etc.
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #your-color1 0%, #your-color2 100%);
        /* Modify other properties as needed */
    }
</style>
""", unsafe_allow_html=True)
```

## 🔍 Troubleshooting

### Common Issues & Solutions

#### 1. TensorFlow Installation Issues

**Problem**: TensorFlow installation fails
```bash
# Solution: Use specific version
pip install tensorflow==2.13.0 --no-cache-dir

# For Apple Silicon:
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal==1.0.1
```

#### 2. Google API Key Issues

**Problem**: API key not working
- Verify the key is correct in `.env` file
- Check API is enabled in Google Cloud Console
- Ensure no extra spaces or characters in the key
- Test API key with a simple request

#### 3. Memory Issues with Large Files

**Problem**: Out of memory errors
```python
# Reduce TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)

# Process files in smaller batches
# Modify the batch processing logic in app.py
```

#### 4. PDF Reading Errors

**Problem**: Cannot read certain PDF files
- Ensure PDFs are not password-protected
- Check PDF is not corrupted
- Try converting PDF to a different format and back
- Use alternative PDF readers if needed

#### 5. Streamlit Port Issues

**Problem**: Port 8501 already in use
```bash
# Use different port
streamlit run app.py --server.port 8502

# Kill existing Streamlit processes (Windows)
taskkill /f /im streamlit.exe

# Kill existing Streamlit processes (macOS/Linux)
pkill -f streamlit
```

### Performance Optimization

#### 1. Speed Up Analysis

```python
# Modify batch size in app.py
BATCH_SIZE = 5  # Process 5 files at a time

# Use threading for parallel processing
import concurrent.futures

def analyze_batch(files):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Process files in parallel
        pass
```

#### 2. Memory Management

```python
# Clear cache between analyses
import gc

def clear_memory():
    gc.collect()
    
# Call after processing each batch
```

## 📊 Understanding the Analysis

### ATS Score Components

1. **ATS Score (0-100%)**: Overall compatibility with ATS systems
2. **Match Percentage (0-100%)**: Alignment with job description
3. **Readability Score (0-100%)**: Format and structure quality
4. **Similarity Score**: TF-IDF cosine similarity metric

### Interpretation Guide

- **90-100%**: Excellent match, high chance of passing ATS
- **80-89%**: Good match, likely to pass with minor improvements
- **70-79%**: Moderate match, needs optimization
- **60-69%**: Fair match, requires significant improvements
- **Below 60%**: Poor match, major revisions needed

### Key Metrics Explained

- **Missing Keywords**: Important terms from job description not found in resume
- **Strengths**: Areas where the resume excels
- **Weaknesses**: Areas needing improvement
- **Recommendations**: Specific actions to improve ATS score

## 🎯 Best Practices

### For Job Descriptions

1. **Complete Information**: Include all required skills and qualifications
2. **Specific Keywords**: Use exact terms from the industry
3. **Clear Requirements**: Separate must-have vs. nice-to-have skills
4. **Format Consistency**: Use bullet points for easy parsing

### For Resume Optimization

1. **Keyword Integration**: Naturally incorporate job-relevant keywords
2. **ATS-Friendly Format**: Use standard fonts and simple formatting
3. **Clear Structure**: Use standard section headers
4. **Quantified Achievements**: Include numbers and metrics
5. **Relevant Experience**: Prioritize job-relevant experience

### For Batch Analysis

1. **Consistent Naming**: Use clear, descriptive filenames
2. **Quality Control**: Ensure all PDFs are readable
3. **Size Management**: Keep individual files under 5MB
4. **Batch Sizes**: Process 5-10 files at a time for optimal performance

## 🔄 Updates and Maintenance

### Keeping Dependencies Updated

```bash
# Check for outdated packages
pip list --outdated

# Update specific packages
pip install --upgrade streamlit
pip install --upgrade google-generativeai

# Update all packages (use with caution)
pip freeze > current_requirements.txt
pip install --upgrade -r requirements.txt
```

### Regular Maintenance Tasks

1. **API Key Rotation**: Update Google API keys regularly
2. **Dependency Updates**: Keep packages updated for security
3. **Performance Monitoring**: Monitor memory usage and response times
4. **Backup Configuration**: Save working configurations

## 📈 Advanced Features

### Custom Keyword Lists

Modify the keyword extraction in `app.py`:

```python
# Add industry-specific keywords
tech_keywords = [
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++',
    # Frameworks
    'react', 'angular', 'vue', 'django', 'flask',
    # Cloud Technologies
    'aws', 'azure', 'gcp', 'docker', 'kubernetes',
    # Add your industry-specific terms
]
```

### Custom Scoring Weights

Adjust scoring weights for different criteria:

```python
def calculate_weighted_score(ats_score, match_percentage, readability):
    weighted_score = (
        ats_score * 0.4 +           # 40% weight
        match_percentage * 0.4 +    # 40% weight
        readability * 0.2           # 20% weight
    )
    return weighted_score
```

## 🌐 Deployment Options

### Local Development
- Use `streamlit run app.py` for local testing
- Access at `localhost:8501`

### Streamlit Cloud Deployment

1. **Prepare Repository**
   ```bash
   # Ensure all files are committed
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub repository
   - Set environment variables (GOOGLE_API_KEY)
   - Deploy application

3. **Environment Variables**
   - Add `GOOGLE_API_KEY` in Streamlit Cloud settings
   - Use secrets management for sensitive data

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.headless", "true"]
```

## 🆘 Support and Resources

### Getting Help

1. **Issues**: Check troubleshooting section first
2. **Documentation**: Refer to official Streamlit and TensorFlow docs
3. **Community**: Join Streamlit community forums
4. **Updates**: Follow project repository for updates

### Useful Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [TensorFlow Guides](https://www.tensorflow.org/guide)
- [Plotly Documentation](https://plotly.com/python/)

### Contact Information

**Developer**: Anubhav Raj
- **GitHub**: [github.com/Anubhx](https://github.com/Anubhx)
- **LinkedIn**: [linkedin.com/in/anubhax](https://linkedin.com/in/anubhax/)
- **Email**: [anubhav0427@gmail.com](mailto:anubhav0427@gmail.com)

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini AI for intelligent analysis
- Streamlit team for the amazing framework
- TensorFlow community for ML capabilities
- Plotly for interactive visualizations

---

## 🚀 Quick Start Summary

1. **Install Python 3.8+**
2. **Clone/Download project**
3. **Create virtual environment**: `python -m venv ats_env`
4. **Activate environment**: `ats_env\Scripts\activate` (Windows) or `source ats_env/bin/activate` (Mac/Linux)
5. **Install dependencies**: `pip install -r requirements.txt`
6. **Get Google API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
7. **Create .env file** with `GOOGLE_API_KEY=your_key_here`
8. **Run application**: `streamlit run app.py`
9. **Open browser**: `http://localhost:8501`
10. **Start analyzing resumes!** 🎉

**Happy Analyzing!** 🚀📊
