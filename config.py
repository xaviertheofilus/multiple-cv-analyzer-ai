# Configuration file for Smart ATS Analyzer Pro
# Modify these settings to customize the application behavior

# File Upload Settings
MAX_FILE_COUNT = 10  # Maximum number of files that can be uploaded
MAX_TOTAL_SIZE_GB = 1  # Maximum total size in GB
ALLOWED_EXTENSIONS = ['pdf']  # Supported file formats

# Analysis Settings
MAX_TEXT_LENGTH = 10000  # Maximum text length to process (characters)
BATCH_SIZE = 5  # Number of files to process simultaneously
ENABLE_PARALLEL_PROCESSING = True  # Enable threading for faster processing

# TF-IDF Settings
TFIDF_MAX_FEATURES = 5000  # Maximum number of features for TF-IDF
TFIDF_NGRAM_RANGE = (1, 2)  # N-gram range for feature extraction
TFIDF_MIN_DF = 2  # Minimum document frequency
TFIDF_MAX_DF = 0.95  # Maximum document frequency

# Scoring Weights (should sum to 1.0)
WEIGHTS = {
    'ats_score': 0.4,
    'match_percentage': 0.4,
    'readability': 0.2
}

# Industry-Specific Keywords
TECH_KEYWORDS = [
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
    'go', 'rust', 'scala', 'kotlin', 'swift', 'objective-c', 'r', 'matlab',
    
    # Web Technologies
    'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
    'django', 'flask', 'spring', 'laravel', 'rails', 'asp.net',
    
    # Databases
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
    'oracle', 'sqlite', 'cassandra', 'dynamodb',
    
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab',
    'terraform', 'ansible', 'nginx', 'apache', 'linux', 'unix',
    
    # Data Science & ML
    'machine learning', 'deep learning', 'artificial intelligence',
    'data science', 'big data', 'analytics', 'statistics',
    'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
    'spark', 'hadoop', 'kafka', 'tableau', 'power bi',
    
    # Mobile Development
    'android', 'ios', 'flutter', 'react native', 'xamarin',
    
    # Other Technologies
    'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum',
    'ci/cd', 'git', 'github', 'bitbucket', 'jira', 'confluence'
]

# Business/Finance Keywords
BUSINESS_KEYWORDS = [
    'excel', 'powerpoint', 'word', 'outlook', 'sharepoint',
    'crm', 'erp', 'sap', 'salesforce', 'hubspot',
    'project management', 'business analysis', 'stakeholder management',
    'process improvement', 'strategic planning', 'budget management',
    'financial analysis', 'market research', 'competitive analysis'
]

# Healthcare Keywords
HEALTHCARE_KEYWORDS = [
    'hipaa', 'clinical', 'patient care', 'medical records', 'ehr',
    'healthcare', 'nursing', 'pharmacy', 'medical', 'diagnosis',
    'treatment', 'therapy', 'rehabilitation', 'surgery'
]

# Education Keywords
EDUCATION_KEYWORDS = [
    'curriculum', 'teaching', 'learning', 'assessment', 'evaluation',
    'classroom management', 'student engagement', 'educational technology',
    'lesson planning', 'professional development'
]

# UI Theme Settings
THEME = {
    'primary_color': '#667eea',
    'secondary_color': '#764ba2',
    'background_color': '#f8f9fa',
    'text_color': '#333333',
    'success_color': '#28a745',
    'warning_color': '#ffc107',
    'error_color': '#dc3545'
}

# Chart Settings
CHART_CONFIG = {
    'height': 400,
    'color_scale': 'Viridis',
    'template': 'plotly_white',
    'font_family': 'Arial, sans-serif'
}

# Report Settings
REPORT_CONFIG = {
    'include_charts': True,
    'include_summary': True,
    'include_recommendations': True,
    'pdf_font_size': 12,
    'excel_header_color': '#4472C4'
}

# API Settings
API_CONFIG = {
    'max_retries': 3,
    'timeout_seconds': 30,
    'rate_limit_per_minute': 15  # Free tier limit
}

# Cache Settings
CACHE_CONFIG = {
    'enable_caching': True,
    'cache_ttl': 3600,  # 1 hour in seconds
    'max_cache_size': 100  # Maximum cached items
}

# Logging Settings
LOGGING_CONFIG = {
    'enable_logging': True,
    'log_level': 'INFO',
    'log_file': 'ats_analyzer.log',
    'max_log_size': 10  # MB
}

# Feature Flags
FEATURES = {
    'enable_batch_processing': True,
    'enable_advanced_analytics': True,
    'enable_export_features': True,
    'enable_real_time_dashboard': True,
    'enable_keyword_suggestions': True,
    'enable_similarity_matching': True
}

# Email Settings (for notifications)
EMAIL_CONFIG = {
    'enable_notifications': False,
    'smtp_server': '',
    'smtp_port': 587,
    'username': '',
    'password': ''
}

# Custom Prompts for Different Analysis Types
ANALYSIS_PROMPTS = {
    'detailed': """
    As an expert ATS system, provide a comprehensive analysis of this resume against the job description.
    Focus on technical skills, experience relevance, keyword matching, and overall fit.
    Provide specific, actionable recommendations for improvement.
    """,
    
    'quick': """
    Provide a quick ATS evaluation focusing on keyword matching and overall compatibility.
    Give a brief summary of strengths and areas for improvement.
    """,
    
    'technical': """
    Focus specifically on technical skills, programming languages, frameworks, and tools.
    Evaluate the technical depth and breadth of the candidate's experience.
    """
}

# Industry-specific configurations
INDUSTRY_CONFIGS = {
    'technology': {
        'keywords': TECH_KEYWORDS,
        'weight_technical_skills': 0.5,
        'weight_experience': 0.3,
        'weight_education': 0.2
    },
    'business': {
        'keywords': BUSINESS_KEYWORDS,
        'weight_technical_skills': 0.2,
        'weight_experience': 0.5,
        'weight_education': 0.3
    },
    'healthcare': {
        'keywords': HEALTHCARE_KEYWORDS,
        'weight_technical_skills': 0.3,
        'weight_experience': 0.4,
        'weight_education': 0.3
    }
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    'enable_metrics': True,
    'track_processing_time': True,
    'track_memory_usage': True,
    'alert_threshold_seconds': 30
}

# Default values for missing configurations
DEFAULTS = {
    'ats_score': 50,
    'match_percentage': 45,
    'readability': 70,
    'analysis_timeout': 60
}