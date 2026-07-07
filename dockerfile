# Use a lightweight Python image for the runtime
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# ===== FREE ALTERNATIVE ADDITION =====
# tesseract-ocr: pymupdf4llm shells out to this for scanned/image-heavy PDF
# pages during live uploads. Not needed by Azure's old pipeline (it never ran
# ingestion inside this container), but our free-stack uploads do local OCR.
# Retry loop guards against transient mirror hiccups (hash mismatches, EOF)
# seen when building through a home network's NAT/proxy path.
RUN for i in 1 2 3; do \
        apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && break; \
        echo "apt-get attempt $i failed, retrying..." && sleep 5; \
    done \
    && rm -rf /var/lib/apt/lists/*

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