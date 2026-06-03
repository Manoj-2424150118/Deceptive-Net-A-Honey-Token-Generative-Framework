FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["python", "main.py", "--mode", "full"]
