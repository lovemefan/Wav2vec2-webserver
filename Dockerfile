FROM python:3.8

MAINTAINER lovemefan lovemefan@outlook.com

ENV PORT=8080

WORKDIR /wav2vec2-webserver
# 拷贝后端代码
COPY ./backend /wav2vec2-webserver
# 拷贝模型 包含fairseq 格式的wav2vec2 模型
# 具体包括 模型文件checkpoint_best.pt  词表文件dict.ltr.txt
COPY ./w2v2-fairseq-model  /w2v2-fairseq-model
RUN python -m pip install --upgrade pip && pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -e fairseq_lib
RUN apt-get update -y && apt-get install libsndfile1 -y
EXPOSE ${PORT}/tcp

CMD ["python", "backend/routes/app.py"]