# emotion_detection
Using PyTorch to develop an emotion detection app.

Currently WIP, emotion_notebook.ipynb is an experimental notebook where I have played around with transfer learning as well as implementing a custom convolutional neural net with an attention mechanism added through a spatial transformer network. Transfer learning was proving infeasible due to computational resources which lead me to implementing a model from the paper "Deep-Emotion: Facial Expression Recognition Using Attentional Convolutional Network".

To train a model, clone the repository and run "python train.py" from the command line. At present, hyperparameters cannot be parsed into the command line but will instead need to be manually adjusted within the train.py script.
