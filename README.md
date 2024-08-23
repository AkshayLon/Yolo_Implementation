# Yolo_Implementation
This is an efficient implementation of the YOLO "You-only-look-once" model.
## Initialising data
During training, the model feeds in the data using the DataLoader class given in the archive_data.py. 
<p>Implement the __init__() method to pull the data from whichever datasource. The only conditions being:</p>
<ul>
  <li>self.x is the tensor containing image data with shape (dataset_size, 3, 448, 448)</li>
  <li>self.y is the tensor containing the labels with shape (dataset_size, max_objects, 5) - Each box having the format [x,y,w,h,class] and any unused boxes [-1,-1,-1,-1,-1].</li>
</ul>

## Setting hyperparameters
main.py includes all the hyperparameters of the model:
<ul>
  <li>bnd_box_number - Number of bounding boxes each cell predicts</li>
  <li>classes - Available classes in the dataset</li>
  <li>optimizer - Initialises how the training process minimises the loss function</li>
  <li>MAX_EPOCHS - Number of epochs to train the model for.</li>
</ul>
These are all available to set depending on the use case.

## Model information
The implementation of the model is given in the yolov1 class of the yolo_implementation.py file. The loss function is implemented in the CustomLoss class of the same file.

## Training and running the model
To install the dependencies of the process, run the following command:
```bash
$ pip install requirements.txt
```
To finally train the model, run:
```
$ python .\main.py
```
