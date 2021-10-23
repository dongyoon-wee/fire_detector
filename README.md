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
python train_model.py --train_data_dir [TRAIN_DATA_DIR] --val_data_dir [VAL_DATA_DIR] --model_path [MODEL_PATH]
```
* TRAIN_DATA_DIR : training data folder path
* VAL_DATA_DIR : validation data folder path
* MODEL_PATH : path to save model trained
* Example
```bash
python train_model.py --train_data_dir train_data --val_data_dir val_data --model_path best.pth
```

### Infer model
* Modify the model_path in infer_model.py for running settings
```bash
python infer_model.py --infer_data_dir [INFER_DATA_DIR] --infer_image_path [INFER_IMAGE_PATH] --model_path [MODEL_PATH]
```
* INFER_DATA_DIR : training data folder path
* VAL_DATA_DIR : validation data folder path
* MODEL_PATH : path to save model trained
* Example to infer folder
```bash
python infer_model.py --infer_data_dir data --model_path best.pth
```
* Example to infer a single image
```bash
python infer_model.py --infer_image_path image.jpg --model_path best.pth
```

## References
[ImageFolder 사용법](https://computistics.tistory.com/7)
[Pytorch model transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
[Pytorch 모델 저장하기&불러오기](https://tutorials.pytorch.kr/beginner/saving_loading_models.html)