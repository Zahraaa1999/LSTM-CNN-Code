# CNN-LSTM-Code

This code implements a hybrid CNN-LSTM deep learning model aimed at classifying images into two categories. The model first uses convolutional neural networks (CNNs) to extract spatial features from input images, followed by a long short-term memory (LSTM) layer that helps model sequential dependencies within the extracted features. This architecture is particularly useful when spatial and temporal characteristics of the data both influence the output.

After training the initial model on a dataset of synthetic images, the code applies transfer learning. In this phase, the convolutional layers from the original model are reused and frozen, while new LSTM and dense layers are added and trained on a new dataset. This approach leverages previously learned feature representations to improve training efficiency and performance on similar tasks.

Finally, the model's performance is evaluated on a separate test dataset, and training metrics like accuracy and loss are visualized. Overall, the code demonstrates a practical use of CNN-LSTM architecture with transfer learning for effective image classification.
