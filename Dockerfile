# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Start with bash shell
CMD ["bash"]
