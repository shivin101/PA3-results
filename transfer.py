#!/usr/bin/env python
# coding: utf-8

# In[6]:


from baseline_cnn import *
from baseline_cnn import BasicCNN
from classifying_metrics import *
import pickle as pkl
# Setup: initialize the hyperparameters/variables
num_epochs = 10           # Number of full passes through the dataset
batch_size = 16          # Number of samples in each minibatch
learning_rate = 0.00001  
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing
THRES=3

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
transform = transforms.Compose([transforms.Resize([512, 512]),
                                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomResizedCrop(size=512,scale=(0.8 ,1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0], std=[1])])



# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras,transfer=True)

# Instantiate a VGG19 to run on the GPU or CPU based on CUDA support
model = torchvision.models.resnet18(pretrained=True)
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)
for param in model.parameters():
    param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
#model.classifier[6] = nn.Linear(4096, 15) # Change the final layer 
model.fc = nn.Linear(51200,15)
model = model.to(computing_device) 
#model.cuda()


def wloss(outputs, labels):
    weights = torch.tensor([0.010413209133322493, 0.043375237611558454, 0.009038543543746693, 0.006050381239171344, 0.020817413416131914, 0.019012207293014484, 0.08411340626979365, 0.02270205288043657, 0.025790933013086503, 0.05226499538518225, 0.047859357603210625, 0.07139162774144409, 0.03555872507299105, 0.5496177368587886, 0.001994172938121485])
    weights = weights.to(computing_device)
    loss = -weights * (labels * torch.log(outputs + 10 ** -12) + (1 - labels) * torch.log(1 - outputs + 10 ** -12)) #TODO - loss criteria are defined in the torch.nn package
    loss = torch.mean(loss, 0)
    loss = torch.sum(loss)
    return loss


criterion = nn.BCEWithLogitsLoss() #TODO - loss criteria are defined in the torch.nn package
#TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #TODO - optimizers are defined in the torch.optim package


# In[7]:
# Track the loss across training
total_loss = []
avg_epoch_loss = []
avg_validation_loss = []
min_validation_loss = 10 ** 6
thres = 0

try:
    model.load_state_dict(torch.load("model5.dct"))
    print("Loaded pre-trained model!")
except:
    print("New model created!")


for epoch in range(num_epochs):
    avg_minibatch_loss = []
    N = 32
    N_minibatch_loss = 0.0    

    # Get the next minibatch of images, labels for training
    for minibatch_count, (images, labels) in enumerate(train_loader, 0):

        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)
        images.requires_grad = False
        labels.requires_grad = False

        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Automagically compute the gradients and backpropagate the loss through the network
        loss.backward()

        # Update the weights
        optimizer.step()

        # Add this iteration's loss to the total_loss
        total_loss.append(loss.item())
        N_minibatch_loss += loss.item()
        
        if minibatch_count % N == 0:     
            # Print the loss averaged over the last N mini-batches    
            N_minibatch_loss /= N
            avg_minibatch_loss.append(N_minibatch_loss)
            N_minibatch_loss = 0.0
            print('Epoch %d, average minibatch %d loss: %.7f' % (epoch + 1, minibatch_count, avg_minibatch_loss[-1]))
    
    avg_epoch_loss.append(np.mean(avg_minibatch_loss))
    with torch.no_grad():
        total_err, total_N = 0, 0
        for minibatch_count, (images, labels) in enumerate(val_loader, 0):
            images, labels = images.to(computing_device), labels.to(computing_device)
            images.requires_grad = False
            labels.requires_grad = False
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_err += loss.item()
            total_N += 1
        avg_validation_loss.append(total_err / total_N)
    if avg_validation_loss[-1] >= min_validation_loss:
        thres += 1
    else:
        thres = 0
        min_validation_loss = total_err / total_N
        torch.save(model.state_dict(), "model5.dct")
    if thres > THRES:
        print("Early exit!")
        break
    pkl.dump([avg_epoch_loss, avg_validation_loss], open("losses5.pkl", "wb"))
    print("Finished", epoch + 1, "epochs of training")
print("Training complete after", epoch, "epochs")
pkl.dump([avg_epoch_loss, avg_validation_loss], open("losses5.pkl", "wb"))


