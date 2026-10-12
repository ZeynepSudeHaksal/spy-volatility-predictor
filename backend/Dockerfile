# Backend API image for the Market Analytics dashboard.
#
# This serves pre-computed predictions (backend/data/*.json, written by the
# scripts/{volatility,direction} pipelines) over a small Flask API -- it does
# NOT train models or run the pipeline, so it only needs Flask + a
# production WSGi server, not the ML stack. See backend/requirements.txt.

FROM python:3.11-slim

# Keep Python from writing .pyc files / buffering stdout -- useful for
# reading container logs in real time (docker logs -f).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first, separately from the app code, so Docker's
# layer cache can skip this step on rebuilds that only change app.py.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the pre-computed prediction JSON it serves.
COPY app.py .
COPY data/ ./data/

# Run as a non-root user inside the container.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5001

# FRONTEND_ORIGIN controls the CORS allow-origin header (app.py); override
# it at `docker run` time to match wherever the frontend is actually served
# from. Defaults to the Vite dev server for a zero-config local run.
ENV FRONTEND_ORIGIN=http://localhost:5173 \
    PORT=5001

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request,os,sys; sys.exit(0) if urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/api/health', timeout=2).status == 200 else sys.exit(1)"

# gunicorn, not `python app.py` -- a real production WSGi server (multiple
# worker processes, no Flask dev-server warning banner) instead of the
# single-threaded debug server the README's local setup uses.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 app:app"]
