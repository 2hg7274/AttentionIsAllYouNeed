# AttentionIsAllYouNeed  

Attention Is All You Need 논문에 나온 Transformer 모델 구현과 학습 코드  

영어를 한글로 번역하는 모델 학습 실행 코드  

데이터셋은 [korean-parallel-corpora](https://huggingface.co/datasets/Moo/korean-parallel-corpora) 를 다운 받아 진행.  
tokenizer는 [komt-mistral-7b-v1](https://huggingface.co/davidkim205/komt-mistral-7b-v1)에서 사용한 tokenizer로 사용.  

`python train.py`로 학습 진행.  
학습에 사용되는 configuration은 conf.py에서 수정 가능.