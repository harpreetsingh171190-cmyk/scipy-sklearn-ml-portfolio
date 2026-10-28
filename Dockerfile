FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt requirements-optional-apps.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt -r requirements-optional-apps.txt
COPY . .
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8501 8000
# Default: dashboard. API: docker run -p 8000:8000 IMAGE uvicorn api.main:app --host 0.0.0.0
CMD ["streamlit", "run", "apps/portfolio_dashboard.py", "--server.address=0.0.0.0"]
