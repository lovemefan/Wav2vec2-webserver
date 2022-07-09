# wav2vec2 语音识别http接口服务器

## 当前功能
* 中文语音识别功能: 给定中文语音识别为中文文本
* 中文语音文本对齐：给定中文语音与文本，输出时间锚点信息

## 部署
从源码构建
模型基于fairseq的wav2vec2-large 在wenetspeech 10000h的预训练模型，在约两千小时的开源语料微调得到(后续开源)

### 1. 下载源码以及模型
```bash
# 下载代码
git clone https://github.com/lovemefan/Wav2vec2-webserver.git
cd Wav2vec2-webserver
```

### 2. 修改配置文件 `/backend/config/config.ini`
以下参数必填
```ini
[wav2vec2]
; 只有当cpu值为true 使用cpu解码
cpu = false
;dict存放路径
data = /w2v2-fairseq-model
;checkpoint 路径
path = /w2v2-fairseq-model/checkpoint_best.pt
;粒度， 词表由 path和labels 组成
labels = ltr
```
### 3. 构建docker镜像

```bash
sudo docker build -t w2v2-webserver .
```

### 4. 创建容器

```bash
# 自行更改端口
sudo docker run -d -e "PORT=8080"--name w2v2-webserver -p8080:8080 w2v2-web:latest
```

### 5. 接口请求
#### 5.1 识别接口
语音文件格式要求采样率为16kHz,采样位宽为16bit，单通道的wav格式的音频。

```bash
curl --location --request POST 'http://localhost:8090/v1/api/speech/recognition' \
--form 'audio=@F:\语料\aishell\data_aishell\wav\S0002\train\S0002\BAC009S0002W0122.wav'
```

返回结果

```json
{
	"message": "Success",
	"status_code": "RECOGNITION_FINISHED",
	"code": 200,
	"data": "而对面楼市成交抑制作用最大的限购"
}
```

#### 5.2对齐接口
文本以句号、问号、感叹号、双引号为分句标志，输出以句子粒度的对齐结果。
```bash
curl --location --request POST 'http://localhost:8080/v1/api/speech/segment' \
--form 'audio=@F:\语料\aishell\data_aishell\wav\S0002\train\S0002\BAC009S0002W0122.wav' \
--form 'text=而对面楼市成交抑制作用最大的限购。'
```

返回结果

```json
{
	"message": "Success",
	"status_code": "RECOGNITION_FINISHED",
	"code": 200,
	"data": [
		{
			"start": 0.2808896321070234,
			"end": 5.527506688963211,
			"score": -0.6467344031540316,
			"text": "而对面楼市成交抑制作用最大的限购。"
		}
	]
}
```