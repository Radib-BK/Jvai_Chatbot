# How to Run

A simple AI chatbot that automatically processes financial policy documents and answers questions about them.

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your PDF Document
- Place your PDF file at: `data/Policy-file.pdf`
- The chatbot will automatically process this file when it starts

### 4. Run the Application
```bash
# Use streamlit run (NOT python app.py)
streamlit run app.py
```

**Important:** Always use `streamlit run app.py`, not `python app.py`

### 5. Use the Chatbot
- Open your browser to `http://localhost:8501`
- Wait for the PDF to be processed automatically
- Start asking questions about your financial policy document

## Requirements
- Python 3.8+
- PDF document placed at `data/Policy-file.pdf`
- Internet connection (for downloading AI models on first run)

That's it! The chatbot will handle everything else automatically.

