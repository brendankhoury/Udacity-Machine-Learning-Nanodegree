from PIL import Image
import numpy as np
import torch
def process_image(image_location):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_location)
    ratio = 0
    if image.size[0] < image.size[1]:
        ratio = image.size[1]/image.size[0]
        image = image.resize((256,(int)(256*ratio)))
    else:
        ratio = image.size[0]/image.size[1]
        image = image.resize(((int)(256*ratio),256))

        


    # Crop the center of the image
    width, height = image.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image)
    np_image = np_image.astype(np.float)

    np_image /= 255.0
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])

    np_image = np_image.transpose((2,0,1))


    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    return tensor_image


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = model.to(device)
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    with torch.no_grad():
        data = process_image(image_path).to(device)
        data.unsqueeze_(0)
        output = model.forward(data)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        return top_p, top_class
    
