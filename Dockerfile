FROM python:3.8

MAINTAINER lovemefan lovemefan@outlook.com

ENV PORT=8080

WORKDIR /wav2vec2-webserver
COPY . /wav2vec2-webserver
COPY /userdata/zlf/pretrain-models/wav2vec/finetuned/zh-char  /model
RUN python -m pip install --upgrade pip && pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -e fairseq_lib
EXPOSE ${PORT}/tcp

CMD ["python", "backend/routes/app.py"]