FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow and other Python dependencies
RUN pip install --upgrade pip

COPY . /app/
RUN pip install -r requirements.txt

CMD ["sh", "-c", "python ./server/manage.py runserver 0.0.0.0:1616"]
