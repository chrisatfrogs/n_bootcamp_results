FROM python:3.9

EXPOSE 8501

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "streamlit", "run" ]
CMD [ "app.py" ]