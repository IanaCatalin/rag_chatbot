# rag_chatbot


## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/rag_chatbot.git
cd rag_chatbot
```
2. Set Up a Virtual Environment
Create and activate a virtual environment to manage dependencies:
``` 
Windows:

```bash
python -m venv venv
venv\Scripts\activate
```
macOS/Linux:

```bash
python3 -m venv venv
source .venv/bin/activate
```
3. Install Dependencies
Install the required Python packages with:

```bash
pip install -r requirements.txt
```

4. Configure Environment Variables
Create a .env file in the project root and add your OpenAI API key:

env
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Make sure to add .env to your .gitignore to keep it secure.

Usage
Running the Chatbot Locally
Activate Your Virtual Environment:
Make sure your virtual environment is active.

Run the Streamlit App:

```bash
streamlit run src/streamlit_app.py
```
