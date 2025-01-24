FROM python:3.10

COPY requirements.txt .
COPY nutrional_agent.py .
COPY enchilada-image.jpg .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENV openai_key = ""

CMD ["python3", "nutritional_agent.py"]