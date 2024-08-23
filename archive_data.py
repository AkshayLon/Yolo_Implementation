import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os, torch
import xml.etree.ElementTree as ET

class ArchiveDataset(Dataset):

    def __init__(self, function):
        idir = os.path.join(os.path.dirname(__file__), 'archive', function)
        inputs, labels, max_objects = [], [], 0
        class_dict = {"banana":0, "orange":1, "apple":2}
        for f in os.listdir(idir):
            if f.endswith('jpg'):
                xml_f = f"{f[:len(f)-4]}.xml"
                root = ET.parse(os.path.join(idir, xml_f)).getroot()
                w,h = int(root.find("size").find("width").text), int(root.find("size").find("height").text)
                if w+h==0: continue
                objects = root.findall("object")
                max_objects = len(objects) if len(objects)>max_objects else max_objects
                count, bndboxes = 0, []
                for box in objects:
                    b = box.find("bndbox")
                    x1,y1,x2,y2 = int(b.find("xmin").text), int(b.find("ymin").text), int(b.find("xmax").text), int(b.find("ymax").text)
                    bndboxes.append([((x1+x2)/2)/w, ((y1+y2)/2)/h, abs(x2-x1)/w, abs(y2-y1)/h, class_dict[box.find("name").text]])
                    count += 1
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Resize((448,448))
                ])
                img = Image.open(os.path.join(idir, f))
                transformed_img = transform(img)
                inputs.append(transformed_img.unsqueeze(dim=0))
                labels.append(bndboxes)
        cleaned_labels = list(b + [[-1,-1,-1,-1,-1]]*(max_objects-len(b)) for b in labels)
        self.x, self.y = torch.cat(tensors=inputs, dim=0), torch.tensor(cleaned_labels)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
                
                


                