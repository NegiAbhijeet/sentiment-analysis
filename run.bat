@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Running Streamlit app...
streamlit run app/app.py

pause
