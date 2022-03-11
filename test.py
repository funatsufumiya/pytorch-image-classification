import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--class_names", required=True, type=str, default="apple,banana,grape", help="class names")
ap.add_argument("--model_path", type=str, default="models/resnet18.pth", help="Path to model")
ap.add_argument("--test_dir", type=str, default="imds_small/test", help="Directory for validation images")
ap.add_argument("--result_image_path", type=str, default="test_result.png", help="Path to result image")
args= vars(ap.parse_args())

print("args:",args)

# Paths for image directory and model
IMDIR=args["test_dir"]
MODEL=args["model_path"]

# Load the model for testing
model = torch.load(MODEL)
model.eval()

# Class labels for prediction
class_names=args["class_names"].split(",")

# Retreive 9 random images from directory
## print("imdir:", IMDIR)
files=Path(IMDIR).resolve().glob('**/*.*')
# print(list(files))
images=random.sample(list(files), 9)
## images=list(files)

# Configure plots
fig = plt.figure(figsize=(9,9))
rows,cols = 3,3

# Preprocessing transformations
preprocess=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Perform prediction and plot results
with torch.no_grad():
    for num,img in enumerate(images):
         img=Image.open(img).convert('RGB')
         inputs=preprocess(img).unsqueeze(0).to(device)
         outputs = model(inputs)
         _, preds = torch.max(outputs, 1)    
         label=class_names[preds]
         print("Image:",img,"Prediction:",label)
         plt.subplot(rows,cols,num+1)
         plt.title("Pred: "+label)
         plt.axis('off')
         plt.imshow(img)
    
    print("Saved test result to", args["result_image_path"])
    plt.savefig(args["result_image_path"])