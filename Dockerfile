# Use a lightweight Python image for the runtime
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# ===== FREE ALTERNATIVE ADDITION =====
# tesseract-ocr: pymupdf4llm shells out to this for scanned/image-heavy PDF
# pages during live uploads. Not needed by Azure's old pipeline (it never ran
# ingestion inside this container), but our free-stack uploads do local OCR.
# Retry via `||` chains (not a for-loop) so a final failure actually fails
# the build - the previous for-loop version exited 0 even if every attempt
# failed, since the last command run was `sleep`, not the failed apt-get.
RUN (apt-get update && apt-get install -y --no-install-recommends tesseract-ocr) \
    || (sleep 5 && apt-get update && apt-get install -y --no-install-recommends tesseract-ocr) \
    || (sleep 5 && apt-get update && apt-get install -y --no-install-recommends tesseract-ocr) \
    && rm -rf /var/lib/apt/lists/*
# Sanity check: fail the build loudly here rather than discovering a missing
# binary later at runtime when a user uploads a scanned PDF.
RUN tesseract --version

RUN pip install uv

COPY requirements.txt ./

RUN uv pip install --system -r requirements.txt

# Copy application code into the container
COPY . /app

# Expose the port the app will run on
EXPOSE 7860

# ===== FREE ALTERNATIVE ADDITION =====
# Read $PORT from the environment so this same image works on Render (which
# injects its own $PORT) - defaulting to 7860 since Hugging Face Spaces
# hardcodes that port and doesn't set $PORT itself.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]