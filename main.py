import torch
from yolo_implementation import yolov1, CustomLoss
from generate_data import TrainingDataset, TestingDataset
from archive_data import ArchiveDataset
from torch.utils.data import DataLoader
import torch.optim as optim

if __name__=="__main__":
    # Initialises the dataset, dataloader and hyperparameters
    archive_data = ArchiveDataset(function='train')
    bnd_box_number, classes = 2, 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = yolov1(B=bnd_box_number, C=classes).to(device)
    criterion = CustomLoss(B=bnd_box_number, C=classes)
    image_loader = DataLoader(dataset=archive_data, batch_size=1)
    optimizer = optim.SGD(model.parameters(), lr=0.0025)
    
    # Training
    MAX_EPOCHS = 1
    print("Starting training\n")
    for epoch in range(MAX_EPOCHS):
        running_loss = 0
        for inputs, labels in image_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss={running_loss}")
    print("Training Complete!")