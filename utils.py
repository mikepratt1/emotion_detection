from pathlib import Path
import torch

def save_model(model, model_name):

    # Create models directory (if it doesn't already exist)
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                    exist_ok=True # if models directory already exists, don't error
    )

    # Create model save path
    MODEL_SAVE_PATH = MODEL_PATH / model_name

    # Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
            f=MODEL_SAVE_PATH)