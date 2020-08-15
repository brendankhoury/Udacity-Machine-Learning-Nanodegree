import argparse
import torch
import imageprocessing

parser = argparse.ArgumentParser()

parser.add_argument('path_to_image', action='store')
parser.add_argument('checkpoint', action='store')
parser.add_argument('--category_names', action='store', dest='category_names')
parser.add_argument('--top_k', action='store', dest='top_k')
parser.add_argument('--gpu', action="store_true", default=False)

arguments = parser.parse_args()

device = None
if arguments.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print ("Cuda (gpu) not availible, quitting.")
        quit()
else:
    device = torch.device("cpu")
top_k = 5
if arguments.top_k != None:
    top_k = (int)(arguments.top_k)

checkpoint = torch.load(arguments.checkpoint)
model = checkpoint["model"]
model.classifier = checkpoint["classifier"] 
model.state_dict = checkpoint["state_dict"]
model.class_to_idx = checkpoint["class_to_idx"]

import json
json_file_dir = "cat_to_name.json"
if arguments.category_names != None:
    json_file_dir = arguments.category_names
with open(json_file_dir, 'r') as f:
    cat_to_name = json.load(f)

prediction = imageprocessing.predict(arguments.path_to_image, model, device, top_k)
print(prediction)
for i in range(top_k):
    print("Flower: ", cat_to_name[str(prediction[1][0][i].item()+1)], "\t\tCategory: ",prediction[0][0][i].item(),'\n')