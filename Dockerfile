FROM python
WORKDIR /app
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD  ["python","app.py"]