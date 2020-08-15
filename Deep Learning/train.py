import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action="store")
parser.add_argument('--save_dir', action = 'store', dest='save_dir')
parser.add_argument('--arch', action='store', dest='architecture')
parser.add_argument('--learning_rate', action='store', dest='learning_rate')
parser.add_argument('--hidden_units', action='store', dest='hidden_units')
parser.add_argument('--epochs', action='store', dest='epochs')
parser.add_argument('--gpu', action="store_true", default=False)

arguments = parser.parse_args()

data_dir = arguments.data_dir
if not data_dir.endswith("/"):
    data_dir += "/"

save_dir = ""
if arguments.save_dir != None:
    save_dir = arguments.save_dir
    if not save_dir.endswith("/"):
        save_dir += "/"

arch = "vgg16"
if arguments.architecture != None:
    arch = arguments.architecture


learning_rate = 0.002
if arguments.learning_rate != None:
    learning_rate = (float)(arguments.learning_rate)

hidden_units = 1024
if arguments.hidden_units != None:
    hidden_units = (int)(arguments.hidden_units)

epoch_num = 4
if arguments.epochs != None:
    epoch_num = (int)(arguments.epochs)

device = None
if arguments.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print ("Cuda (gpu) not availible, quitting.")
        quit()
else:
    device = torch.device("cpu")

model = None
if arch == "vgg16":
    model = models.vgg16(pretrained= True)
elif arch == "densenet161":
    model = models.densenet161(pretrained=True)
else:
    print("Model not recognized")
    quit()


####################################
#                                  #
#           Loading Data           #
#                                  #
####################################
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
test_data = datasets.ImageFolder(test_dir, transform = valid_test_transforms)
 
# TODO: Using the image datasets and the trainforms, define the dataloaders
batchSize = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchSize, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batchSize,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batchSize,shuffle=True)




####################################
#                                  #
#           Training               #
#                                  #
####################################


for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.2), 
                           nn.Linear(hidden_units,102),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier

for param in model.classifier.parameters():
    param.requires_grad = True

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
step_counter = 0
print_every = 25
running_loss = 0
model.to(device)
model.train()
# print(model)
# quit()
print("Ready to begin training: ")
for e in range(epoch_num):
    for images, labels in train_loader:
        step_counter += 1
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        if step_counter % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for valid_images, valid_labels in valid_loader:
                    valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)
                    
                    logps = model.forward(valid_images)
                    batch_loss = criterion(logps, valid_labels)
                    
                    validation_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == valid_labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                else: 
                    print(f"Epoch: {e+1}/{epoch_num}\t"
                          f"Train loss: {running_loss/print_every:.3f}\t"
                          f"Validation loss: {validation_loss/len(valid_loader):.3f}\t"
                          f"Validation accuracy: {accuracy/len(valid_loader):.3f}\t")
            running_loss = 0
            model.train()

model.eval()
test_loss = 0
accuracy = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        accuracy += torch.mean((top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)).item()

print(f"Testing loss: {test_loss/len(test_loader):.3f}\t"
      f"Testing accuracy: {accuracy/len(test_loader):.3f}")



####################################
#                                  #
#       Saving Checkpoint          #
#                                  #
####################################

checkpoint = {'state_dict': model.state_dict(),
              'epochs' : epoch_num,
              'learning_rate': learning_rate,
              'batch_size': batchSize,
              'classifier': model.classifier,
              'optimizer':optimizer,
              'optimizer_state': optimizer.state_dict(),
              'class_to_idx': train_data.class_to_idx,
              'model': model
              }

torch.save(checkpoint,save_dir + 'checkpoint.pth')