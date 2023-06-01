import torch 
import os
from torchvision import transforms
import engine, data_setup, model_builder, utils 

# Set-up hyperparamater
# Note: in future set these up with argparse module to allow user to specify from the command line

epochs = 2
BATCH_SIZE = 32
learning_rate = 0.001

# Set-up test and train directories
train_dir = "data/train"
test_dir = "data/test"

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(degrees=10)])

# Create dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=transform,
    batch_size=BATCH_SIZE
)

# Instantiate the model
model = model_builder.DeepEmotion().to(device)

# Set-up the loss function and the optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start the train/ test loop
engine.train(model=model,
             train_dataloader=train_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=epochs,
             device=device)

# Save the model 
utils.save_model(model=model, 
                 model_name="emotion_detection_deep_emotion_model.pth")