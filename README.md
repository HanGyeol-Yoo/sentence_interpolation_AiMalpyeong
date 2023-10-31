# sentence_interpolation_AiMalpyeong
origindata/convert_to_alpaca.ipynb 에서 train,val,test 데이터셋 전처리 과정 진행 후 생성

training 폴더에서 bash run_sft.sh으로 학습 진행 -> 데이터 경로 같은거 적기

test_bogan.ipynb으로 추론 진행 -> test_bogan.py 파일로 만들어서 python test_bogan.py 로 실행했을 때 추론 결과가 생성되어야 함

## test set 추론
- test set 데이터 경로 : 'data/test/test.json'
- 결과 저장 경로 : 'result/test/outputs.jsonl'
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
