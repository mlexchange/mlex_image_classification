FROM tensorflow/tensorflow:2.13.0-gpu

COPY requirements.txt requirements.txt
USER root
RUN apt-get update && apt-get install -y --no-install-recommends tree \
    python3-pip &&\
    pip install --upgrade pip

RUN pip install -r requirements.txt
WORKDIR /app/work/
COPY src/ src/

CMD ["bash"]