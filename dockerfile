FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
COPY ./data /code/data
#CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]