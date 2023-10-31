# sentence_interpolation_AiMalpyeong - MLP Lab
- issue contact : deedd123@gmail.com
## 1.라이브러리 설치
```
!pip install requrements.txt
```
## 2. 데이터 전처리
originaldata/convert_to_alpaca.ipynb 주피터노트북 파일 모두 실행
 
## 3. 모델 학습
terminal 환경에서 `run_sft.sh` 파일에 `—-nproc_per_node 4` 부분 숫자를 사용할 GPU 개수로 지정
```
bash run_sft.sh
```

## test set 추론
- test set 데이터 경로 : 'data/test/test.json'
- 결과 저장 경로 : 'outputs.jsonl'
```
python inference_test.py
```

## batch data 추론
- "--data_path" : batch data 파일 경로
- 결과 저장 경로 : "--save_path" 인자로 직접 입력 출력 형식은 .jsonl 이어야 함

```
python inference_batch.py --data_path [data_path] --save_path [save_path]
```

## single data 추론
- 실행 시 결과가 바로 출력
- "--sentence_1" 과 "--sentence_3"에 문장 입력
```
python inference_single.py --sentence_1 [sentence_1] --sentence_3 [sentence_3]
```
