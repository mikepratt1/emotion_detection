import torch


# Set manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

## 1. Training step function

def training_step(model, 
                  dataloader, 
                  loss_fn, 
                  optimizer,
                  device):

    # Put the model into training model
    model.train()

    # Loss and accuracy variables
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        # Move X and y to the device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_logits = model(X)

        # Calculate the loss and the accuracy
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()
        #scheduler.step()

        # predict the class
        y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        acc = (y_pred == y).sum().item()/len(y_pred)
        train_acc += acc

        if batch % 100 == 0:
            print(f"Batch {batch}: Train Accuracy: {acc} | Train Loss: {loss}")

    # Average the loss and accuracy over all batches
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(model, 
                  dataloader, 
                  loss_fn, 
                  optimizer,
                  device):
    # Put the model in evaluation mode
    model.eval()

    # Test loss and accuracy variables
    test_loss, test_acc = 0, 0

    # Test the model
    with torch.inference_mode():
        for batch , (X, y) in enumerate(dataloader):
        # Move X and y to the device (gpu)
            X, y = X.to(device), y.to(device)

            # Forward pass to determine the logits
            test_pred_logits = model(X)

            # Calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Convert logits to a classification prediction
            test_pred = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            
            # Calculate the accuracy
            acc = (test_pred == y).sum().item()/len(y)
            test_acc += acc
        
        # Average loss and accuracy over all batches
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        return test_loss, test_acc


def train(model, 
          train_dataloader, 
          test_dataloader, 
          loss_fn, 
          optimizer, 
          epochs,
          device):
    results = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    for epoch in range(epochs):
        print("-"*50, f"Epoch {epoch+1}", "-"*50)
        train_loss, train_acc = training_step(model,
                                            train_dataloader,
                                            loss_fn,
                                            optimizer,
                                            device)
        test_loss, test_acc = test_step(model,
                                        test_dataloader,
                                        loss_fn,
                                        optimizer,
                                        device)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        print(f"Train loss: {train_loss} | Train Accuracy: {train_acc} | Test Loss: "
            f"{test_loss} | Test Accuracy: {test_acc}")
    return results
