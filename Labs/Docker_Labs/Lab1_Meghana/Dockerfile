# 1. Base image: slim Python
FROM python:3.11-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Install dependencies first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy source code
COPY src/ ./src/

# 5. Train the model at build time, so the model is baked into the image
RUN python src/train.py

# 6. Document the port the app listens on
EXPOSE 5000

# 7. Command run when a container starts
CMD ["python", "src/app.py"]
