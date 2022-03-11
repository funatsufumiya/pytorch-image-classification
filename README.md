## Pytorch-Image-Classification

A simple demo of **image classification** using pytorch. Here, we use a **custom dataset** containing **43956 images** belonging to **11 classes** for training(and validation). Also, we compare three different approaches for training viz. **training from scratch, finetuning the convnet and convnet as a feature extractor**, with the help of **pretrained** pytorch models. The models used include: **VGG11, Resnet18 and MobilenetV2**.

### Dependencies

* Python3, Scikit-learn
* Pytorch, PIL
* Torchsummary, Tensorboard

```python
pip install torchsummary # keras-summary
pip install tensorboard  # tensoflow-logging
```

**NB**: Update the libraries to their latest versions before training.

### How to run

**Download** and extract training dataset: [imds_small](https://drive.google.com/file/d/1fPDnom5uGTpCb0abkzCvKbLadtNx8FlW/view?usp=sharing)

Run the following **scripts** for training and/or testing

```python
python train.py --help # For training the model [--mode=finetune/finetune_mnet/transfer/scratch]
python test.py --help # For testing the model on sample images
```

### Training results

|    | Accuracy | Size | Training Time | Training Mode |
|----|----|----|----|-----|
| **VGG11** | 96.73 | 515.3 MB  |  900 mins |  scratch |
| **Resnet18**  | 99.85  | 44.8 MB |  42 mins |  finetune |
| **MobilenetV2**  | 97.72  | 9.2 MB | 32 mins | transfer |

**Batch size**: 64, **GPU**: Tesla K80

Both **Resnet18 and MobilenetV2**(transfer leraning) were trained for **10 epochs**; whereas **VGG11**(training from scratch) was trained for **100 epochs**.


### Training graphs

**Resnet18:-** 

Finetuning the pretrained resnet18 model.
![Screenshot](results/resnet18.png)

**Mobilenetv2:-**

Mobilenetv2 as a fixed feature extractor.
![Screenshot](results/mobilenetv2.png)

### Sample outputs

Sample classification results

![Screenshot](results/output.png)