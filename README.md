# sentence_interpolation_AiMalpyeong
origindata/convert_to_alpaca.ipynb 에서 train,val,test 데이터셋 전처리 과정 진행 후 생성

training 폴더에서 bash run_sft.sh으로 학습 진행 -> 데이터 경로 같은거 적기

test_bogan.ipynb으로 추론 진행 -> test_bogan.py 파일로 만들어서 python test_bogan.py 로 실행했을 때 추론 결과가 생성되어야 함

## test set 추론
- 결과 저장 경로 : result/test/outputs.jsonl
```
python inference_test.py
```
