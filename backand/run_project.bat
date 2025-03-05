@echo off
start cmd /k "cd backend && venv\Scripts\activate && python flask_api.py"
start cmd /k "cd frontend && npm start"
