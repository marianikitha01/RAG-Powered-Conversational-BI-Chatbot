# 1. Use AWS public ECR Python slim base image to avoid Docker Hub auth
FROM public.ecr.aws/docker/library/python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy dependencies spec
COPY requirements.txt ./
# COPY .env.example .env   # only the template; real .env stays local

# 4. Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Default command: run the chatbot
ENTRYPOINT ["python", "ask_bi.py"]
