# fire_detector
repository for simple fire_detector

## How to use
### Set the environment
```bash
pip install -r requirements.txt
```
### Train model
* Gather data for trainig and validation
* Modify the train_data_dir, val_data_dir in train_model.py for running settings
```bash
python train_model.py
```

### Infer model
* Modify the model_path in infer_model.py for running settings
```bash
python infer_model.py
```


## References
[ImageFolder 사용법](https://computistics.tistory.com/7)
[Pytorch model transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
[Pytorch 모델 저장하기&불러오기](https://tutorials.pytorch.kr/beginner/saving_loading_models.html)