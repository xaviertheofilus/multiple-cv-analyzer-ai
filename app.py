import streamlit as st
import os
import PyPDF2 as pdf
try:
    import google.generativeai as genai
except ImportError:
    print("Installing google-generativeai package...")
    import subprocess
    subprocess.check_call(["pip", "install", "google-generativeai"])
from dotenv import load_dotenv
from fpdf import FPDF
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
import json
from datetime import datetime
import io
import zipfile
import xlsxwriter
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure Google API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in environment variables")
    st.info("Please set your Google API Key in the .env file or in your environment variables")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"❌ Error configuring Google API: {str(e)}")
    st.info("Please check your API key and ensure it's valid")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🚀 Smart ATS Analyzer Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ATSAnalyzer:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.error(f"Error initializing Gemini model: {str(e)}")
            # Fallback to default analysis
            self.model = None
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        
    def extract_pdf_text(self, uploaded_file):
        """Extract text from PDF file"""
        try:
            reader = pdf.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def calculate_similarity_score(self, resume_text, job_description):
        """Calculate similarity using TF-IDF and cosine similarity"""
        try:
            documents = [resume_text, job_description]
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix[0][1] * 100
        except:
            return 0
    
    def extract_keywords(self, text):
        """Extract important keywords from text"""
        # Common tech keywords
        tech_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'node.js', 
            'sql', 'mongodb', 'postgresql', 'aws', 'azure', 'docker', 
            'kubernetes', 'machine learning', 'data science', 'ai',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        for keyword in tech_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def analyze_resume_with_gemini(self, resume_text, job_description, filename):
        """Analyze resume using Gemini API"""
        prompt = f"""
        As an expert ATS system, analyze this resume against the job description and provide a structured evaluation:

        Resume File: {filename}
        Resume Content: {resume_text[:3000]}...
        Job Description: {job_description}

        Please provide analysis in this JSON format:
        {{
            "ats_score": 85,
            "match_percentage": 78,
            "readability": 92,
            "missing_keywords": ["keyword1", "keyword2"],
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "recommendations": ["recommendation1", "recommendation2"],
            "overall_summary": "Brief summary of candidate fit"
        }}
        
        Focus on technical skills, experience relevance, and keyword matching.
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Try to parse JSON from response
            response_text = response.text
            
            # Extract JSON if wrapped in markdown
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            try:
                analysis = json.loads(json_text)
                return analysis
            except:
                # Fallback parsing
                return self.parse_response_fallback(response_text, filename)
                
        except Exception as e:
            st.error(f"Error analyzing resume {filename}: {str(e)}")
            return self.get_default_analysis(filename)
    
    def parse_response_fallback(self, response_text, filename):
        """Fallback parsing when JSON parsing fails"""
        try:
            # Extract numbers using regex
            scores = re.findall(r'(\d+)%', response_text)
            ats_score = int(scores[0]) if scores else 70
            match_percentage = int(scores[1]) if len(scores) > 1 else ats_score
            readability = int(scores[2]) if len(scores) > 2 else 85
            
            return {
                "ats_score": ats_score,
                "match_percentage": match_percentage,
                "readability": readability,
                "missing_keywords": ["API development", "Cloud computing"],
                "strengths": ["Technical experience", "Relevant background"],
                "weaknesses": ["Missing specific keywords", "Limited detail"],
                "recommendations": ["Add more technical keywords", "Improve formatting"],
                "overall_summary": response_text[:200] + "..."
            }
        except:
            return self.get_default_analysis(filename)
    
    def get_default_analysis(self, filename, resume_text="", job_description=""):
        """Default analysis structure with dynamic scoring"""
        # Calculate base scores using similarity if texts are provided
        if resume_text and job_description:
            similarity = self.calculate_similarity_score(resume_text, job_description)
            keywords = self.extract_keywords(resume_text)
            
            # Adjust scores based on content
            ats_score = max(50, min(95, similarity + len(keywords) * 2))
            match_percentage = max(45, min(90, similarity))
            readability = max(60, min(95, 75 + len(keywords)))
        else:
            # Randomize scores within reasonable ranges if no text is provided
            import random
            ats_score = random.randint(55, 85)
            match_percentage = random.randint(50, 80)
            readability = random.randint(70, 90)
        
        return {
            "ats_score": round(ats_score),
            "match_percentage": round(match_percentage),
            "readability": round(readability),
            "missing_keywords": ["Technical skills", "Industry experience"],
            "strengths": ["Professional experience"],
            "weaknesses": ["Needs more specific keywords"],
            "recommendations": ["Enhance technical skills section"],
            "overall_summary": f"Analysis completed for {filename}"
        }

def create_dashboard(analysis_results):
    """Create interactive dashboard with analysis results"""
    if not analysis_results:
        st.warning("No analysis results to display")
        return
    
    # Convert results to DataFrame
    df_results = []
    for filename, analysis in analysis_results.items():
        df_results.append({
            'filename': filename,
            'ats_score': analysis['ats_score'],
            'match_percentage': analysis['match_percentage'],
            'readability': analysis['readability']
        })
    
    df = pd.DataFrame(df_results)
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_ats = df['ats_score'].mean()
        st.metric("Average ATS Score", f"{avg_ats:.1f}%", 
                 delta=f"{avg_ats - 70:.1f}%" if avg_ats > 70 else f"{avg_ats - 70:.1f}%")
    
    with col2:
        avg_match = df['match_percentage'].mean()
        st.metric("Average Match", f"{avg_match:.1f}%",
                 delta=f"{avg_match - 65:.1f}%" if avg_match > 65 else f"{avg_match - 65:.1f}%")
    
    with col3:
        avg_readability = df['readability'].mean()
        st.metric("Average Readability", f"{avg_readability:.1f}%")
    
    with col4:
        total_resumes = len(df)
        st.metric("Total Resumes", total_resumes)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # ATS Score comparison
        fig_bar = px.bar(df, x='filename', y='ats_score', 
                        title='ATS Scores Comparison',
                        color='ats_score',
                        color_continuous_scale='Viridis')
        fig_bar.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Radar chart for top 3 resumes
        top_3 = df.nlargest(3, 'ats_score')
        fig_radar = go.Figure()
        
        for idx, row in top_3.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['ats_score'], row['match_percentage'], row['readability']],
                theta=['ATS Score', 'Match %', 'Readability'],
                fill='toself',
                name=row['filename'][:15] + '...',
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Top 3 Resumes Performance",
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Detailed ranking table
    st.subheader("📊 Detailed Ranking")
    df_display = df.sort_values('ats_score', ascending=False).reset_index(drop=True)
    df_display.index += 1
    df_display['Rank'] = df_display.index
    
    # Reorder columns
    df_display = df_display[['Rank', 'filename', 'ats_score', 'match_percentage', 'readability']]
    df_display.columns = ['Rank', 'Resume', 'ATS Score (%)', 'Match (%)', 'Readability (%)']
    
    st.dataframe(df_display, use_container_width=True)

def generate_excel_report(analysis_results, job_description):
    """Generate comprehensive Excel report"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#4472C4',
            'font_color': 'white'
        })
        
        # Summary sheet
        summary_data = []
        for filename, analysis in analysis_results.items():
            summary_data.append({
                'Resume': filename,
                'ATS Score': analysis['ats_score'],
                'Match Percentage': analysis['match_percentage'],
                'Readability': analysis['readability'],
                'Overall Summary': analysis['overall_summary']
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format summary sheet
        worksheet = writer.sheets['Summary']
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:D', 15)
        worksheet.set_column('E:E', 50)
        
        # Detailed analysis sheet
        detailed_data = []
        for filename, analysis in analysis_results.items():
            detailed_data.append({
                'Resume': filename,
                'ATS Score': analysis['ats_score'],
                'Strengths': ', '.join(analysis['strengths']),
                'Weaknesses': ', '.join(analysis['weaknesses']),
                'Missing Keywords': ', '.join(analysis['missing_keywords']),
                'Recommendations': ', '.join(analysis['recommendations'])
            })
        
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_excel(writer, sheet_name='Detailed Analysis', index=False)
        
        # Job Description sheet
        jd_data = pd.DataFrame([{'Job Description': job_description}])
        jd_data.to_excel(writer, sheet_name='Job Description', index=False)
    
    return output.getvalue()

def generate_pdf_report(analysis_results, job_description):
    """Generate comprehensive PDF report"""
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'Smart ATS Analyzer - Comprehensive Report', 0, 1, 'C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()} - Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
    
    pdf = PDF()
    pdf.add_page()
    
    # Executive Summary
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', 0, 1)
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    total_resumes = len(analysis_results)
    avg_score = sum(a['ats_score'] for a in analysis_results.values()) / total_resumes
    
    summary_text = f"""
Total Resumes Analyzed: {total_resumes}
Average ATS Score: {avg_score:.1f}%
Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}

This report provides a comprehensive analysis of all submitted resumes against the provided job description.
    """
    
    pdf.multi_cell(0, 10, summary_text.strip())
    pdf.ln(10)
    
    # Individual Analysis
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Individual Resume Analysis', 0, 1)
    pdf.ln(5)
    
    for filename, analysis in analysis_results.items():
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f'Resume: {filename}', 0, 1)
        
        pdf.set_font('Arial', '', 10)
        analysis_text = f"""
ATS Score: {analysis['ats_score']}%
Match Percentage: {analysis['match_percentage']}%
Readability: {analysis['readability']}%

Strengths: {', '.join(analysis['strengths'])}
Weaknesses: {', '.join(analysis['weaknesses'])}
Missing Keywords: {', '.join(analysis['missing_keywords'])}

Summary: {analysis['overall_summary']}
        """
        
        pdf.multi_cell(0, 8, analysis_text.strip())
        pdf.ln(10)
    
    return bytes(pdf.output(dest='S'))

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Smart ATS Analyzer Pro</h1>
        <p>Advanced Multi-Resume Analysis with AI-Powered Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ATSAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key check
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.success("✅ API Key Configured")
        else:
            st.error("❌ API Key Not Found")
            st.info("Please add GOOGLE_API_KEY to your .env file")
        
        st.markdown("---")
        
        # Clear results
        if st.button("🗑️ Clear All Results"):
            st.session_state.analysis_results = {}
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Dashboard", "📋 Reports"])
    
    with tab1:
        # Job description
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Enter the job description to match against:",
            placeholder="Paste the complete job description here...",
            height=200
        )
        
        # File upload
        st.subheader("📁 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload multiple resume PDFs (Max 10 files, 1GB total)",
            type="pdf",
            accept_multiple_files=True,
            help="Select up to 10 PDF resume files for batch analysis"
        )
        
        # Validation
        if uploaded_files:
            total_size = sum(file.size for file in uploaded_files)
            file_count = len(uploaded_files)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files Selected", file_count, delta=f"{file_count - 10}" if file_count > 10 else None)
            with col2:
                st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
            with col3:
                st.metric("Status", "✅ Valid" if file_count <= 10 and total_size <= 1024*1024*1024 else "❌ Exceed Limits")
        
        # Analysis button
        if st.button("🔍 Analyze All Resumes", type="primary", disabled=not uploaded_files or not job_description):
            if len(uploaded_files) > 10:
                st.error("Maximum 10 files allowed")
            elif sum(file.size for file in uploaded_files) > 1024*1024*1024:
                st.error("Total file size exceeds 1GB limit")
            else:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = {}
                total_files = len(uploaded_files)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Analyzing {uploaded_file.name}...")
                    progress_bar.progress((idx + 1) / total_files)
                    
                    # Extract text
                    resume_text = st.session_state.analyzer.extract_pdf_text(uploaded_file)
                    
                    if resume_text:
                        # Analyze with Gemini
                        analysis = st.session_state.analyzer.analyze_resume_with_gemini(
                            resume_text, job_description, uploaded_file.name
                        )
                        
                        # Calculate additional metrics
                        similarity_score = st.session_state.analyzer.calculate_similarity_score(
                            resume_text, job_description
                        )
                        analysis['similarity_score'] = similarity_score
                        
                        results[uploaded_file.name] = analysis
                
                st.session_state.analysis_results = results
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis Complete!")
                
                st.success(f"Successfully analyzed {len(results)} resumes!")
    
    with tab2:
        st.subheader("📊 Real-Time Dashboard")
        
        if st.session_state.analysis_results:
            create_dashboard(st.session_state.analysis_results)
        else:
            st.info("Upload and analyze resumes to see the dashboard")
    
    with tab3:
        st.subheader("📋 Generate Reports")
        
        if st.session_state.analysis_results:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Excel Report", type="primary"):
                    excel_data = generate_excel_report(
                        st.session_state.analysis_results, 
                        job_description if 'job_description' in locals() else ""
                    )
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"ATS_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                if st.button("📄 Download PDF Report", type="primary"):
                    pdf_data = generate_pdf_report(
                        st.session_state.analysis_results,
                        job_description if 'job_description' in locals() else ""
                    )
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_data,
                        file_name=f"ATS_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf"
                    )
        else:
            st.info("No analysis results available for report generation")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Smart ATS Analyzer Pro</strong> - Powered by Google Gemini AI & TensorFlow</p>
        <p>Developed by xaviertheofilus | 
        <a href="https://github.com/xaviertheofilus" target="_blank">GitHub</a> | 
        <a href="https://linkedin.com/in/xaviertheofilus/" target="_blank">LinkedIn</a> | 
        <a href="mailto:xavier.theofilus.munthe@gmail.com ">Email</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
