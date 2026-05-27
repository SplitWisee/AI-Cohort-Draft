# Gunakan image Python 3.10 yang lengkap
FROM python:3.10

# Set folder kerja
WORKDIR /code

# Copy requirements dan install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy semua file project
COPY . .

# Jalankan aplikasi
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]