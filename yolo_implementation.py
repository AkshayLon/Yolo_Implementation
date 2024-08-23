import torch, itertools, math
from torch import nn
import torch.nn.init as initial

class yolov1(nn.Module):

    def __init__(self, B, C):
        super(yolov1, self).__init__()
        self.B, self.C = B,C

        def init_weights(layer):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                initial.xavier_normal_(layer.weight)

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=7, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.first_layer.apply(init_weights)
        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.second_layer.apply(init_weights)
        self.third_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.third_layer.apply(init_weights)
        layers_4 = list()
        layers_4.append(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1))
        layers_4.append(nn.LeakyReLU(negative_slope=0.1))
        layers_4.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        layers_4.append(nn.LeakyReLU(negative_slope=0.1))
        for i in range(3):
            layers_4.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1))
            layers_4.append(nn.LeakyReLU(negative_slope=0.1))
            layers_4.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
            layers_4.append(nn.LeakyReLU(negative_slope=0.1))
        layers_4.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1))
        layers_4.append(nn.LeakyReLU(negative_slope=0.1))
        layers_4.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0))
        layers_4.append(nn.LeakyReLU(negative_slope=0.1))
        self.fourth_layer = nn.Sequential(*layers_4)
        self.fourth_layer.apply(init_weights)
        self.fifth_layer = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.fifth_layer.apply(init_weights)
        self.sixth_layer = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.sixth_layer.apply(init_weights)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=50176, out_features=4096),
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=4096, out_features=49*(C+(5*B))),
            nn.ReLU()
        )
        self.fc.apply(init_weights)
        self.pool = nn.MaxPool2d(padding=1, stride=2, kernel_size=2)

    def forward(self, x):
        o1 = self.pool(self.first_layer(x))
        o2 = self.pool(self.second_layer(o1))
        o3 = self.pool(self.third_layer(o2))
        o4 = self.pool(self.fourth_layer(o3))
        o5 = self.fifth_layer(o4)
        o6 = self.sixth_layer(o5)
        x = self.fc(o6)
        return x.reshape(x.shape[0],49,(5*self.B)+self.C)

class CustomLoss(nn.Module):

    def __init__(self, B, C):
        super(CustomLoss, self).__init__()
        self.B, self.C = B, C
        self.lambda_c, self.lambda_n = 5, 0.5

    def iou_calc(self, gt_tensor, pred_tensor):
        """
        Outputs the IOU of the ground truth and predicted tensor
        """
        int_mins, _ = torch.max(input=torch.stack([gt_tensor[:,:,:,0,:], pred_tensor[:,:,:,0,:]]), dim=0)
        int_maxes, _ = torch.min(input=torch.stack([gt_tensor[:,:,:,1,:], pred_tensor[:,:,:,1,:]]), dim=0)
        int_dims = torch.clamp((int_maxes-int_mins), min=0)
        intersection = int_dims[:,:,:,0]*int_dims[:,:,:,1] # Shape = [batch, 49, bnd_box combos]
        gt_wh, pred_wh = torch.abs(gt_tensor[:,:,:,1,:]-gt_tensor[:,:,:,0,:]), torch.abs(pred_tensor[:,:,:,1,:]-pred_tensor[:,:,:,0,:])
        gt_area, pred_area = gt_wh[:,:,:,0]*gt_wh[:,:,:,1], pred_wh[:,:,:,0]*pred_wh[:,:,:,1] 
        union = gt_area+pred_area-intersection
        return torch.nan_to_num(intersection/union)

    def iou_matrix(self, tensor_1, tensor_2):
        """
        Outputs a tensor of shape [batch, 49, self.B^2] with each element being a IOU of the predicted bounding box with ground truth.
        """
        tensor_1, tensor_2 = tensor_1.reshape(tensor_1.shape[0], 49, self.B, 4), tensor_2.reshape(tensor_2.shape[0], 49, self.B, 4)
        gt_min, gt_max = tensor_1[:,:,:,[0,1]]-(tensor_1[:,:,:,[2,3]]**2), tensor_1[:,:,:,[0,1]]+(tensor_1[:,:,:,[2,3]]**2)
        pred_min, pred_max = tensor_2[:,:,:,[0,1]]-(tensor_2[:,:,:,[2,3]]**2), tensor_2[:,:,:,[0,1]]+(tensor_2[:,:,:,[2,3]]**2)
        gt, pred = torch.cat(tensors=[gt_min, gt_max], dim=3), torch.cat(tensors=[pred_min, pred_max], dim=3)
        pred_extension = list(itertools.chain(*list([i]*self.B for i in range(self.B))))
        extended_pred = pred[:,:,pred_extension,:].reshape(pred.shape[0], 49, len(pred_extension), 2, 2)
        gt_extension = list(itertools.chain(*list(range(self.B) for i in range(self.B))))
        extended_gt = gt[:,:,gt_extension,:].reshape(gt.shape[0], 49, len(gt_extension), 2, 2)
        final_matrix = self.iou_calc(gt_tensor=extended_gt, pred_tensor=extended_pred)
        return final_matrix

    def permute(self, t):
        """
        Orders the bounding boxes such that the B bounding boxes with non-zero area are at the top.
        This makes sure that no information about the objects is lost.
        """
        top_b, current_b = torch.zeros(self.B*5), 0
        for box in range(int(t.shape[0]/5)):
            if torch.sum(t[box*5:(box*5)+5])>0: 
                top_b[current_b*5:(current_b*5)+5] = t[box*5:(box*5)+5]
                current_b += 1
                if current_b==self.B: break
        return top_b

    def construct_model_ground_truth(self, bb_selection, ground_truth, prediction):
        """
        Assigns each predicted bounding box with the ground truth with highest IOU value.
        Returns reformatted ground truth tensor such that each predicted bounding box is aligned at the same index as the ground truth it predicts.
        """
        print(prediction.shape)
        final_gt = torch.zeros_like(ground_truth)
        for batch, cell, bnd_box in itertools.product(range(bb_selection.shape[0]), range(49), range(self.B)):
            current_box = int(bb_selection[batch, cell, bnd_box])
            final_gt[batch, cell, range((5*bnd_box),(5*bnd_box)+5)] = ground_truth[batch, cell, range((5*current_box),(5*current_box)+5)]
            final_gt[batch, cell, (5*bnd_box)+4] = self.ground_truth_confidence(prediction=prediction[batch, cell, range((5*bnd_box),(5*bnd_box)+5)], ground_truth=ground_truth[batch, cell, range((5*current_box),(5*current_box)+5)])
        return final_gt

    def ground_truth_confidence(self, prediction, ground_truth):
        """
        Inputs the dimensions of the prediction and ground truth in the format [x,y,w,h] and outputs intersection/union
        """
        rect1, rect2 = torch.cat([prediction[:2], torch.square(prediction[2:4])]), torch.cat([ground_truth[:2], torch.square(ground_truth[2:4])])
        min1, max1 = rect1[:2]-(0.5*rect1[2:4]), rect1[:2]+(0.5*rect1[2:4])
        min2, max2 = rect2[:2]-(0.5*rect2[2:4]), rect2[:2]+(0.5*rect2[2:4])
        int_mins, _ = torch.max(torch.stack((min1, min2)), dim=0)
        int_maxes, _ = torch.min(torch.stack((max1, max2)), dim=0)
        intersection = torch.clamp((int_maxes-int_mins), min=0)[0]*torch.clamp((int_maxes-int_mins), min=0)[1]
        union = ((max1-min1)[0]*(max1-min1)[1])+((max2-min2)[0]*(max2-min2)[1])-intersection
        if (int(union)==0 or int(intersection)==0): return 0
        return int(intersection/union)

    def calculate_obj_loss(self, pred_tensor, gt_tensor):
        """
        Takes in the model output and ground truth and outputs the loss of the object dimensions for that batch.
        For reference, this is the first 3 parts of the 5 part loss function.
        """
        difference = torch.square(gt_tensor-pred_tensor)
        dist_indices = list(i for i in range(difference.shape[2]) if i%5 != 4)
        difference[:,:,dist_indices] = self.lambda_c*difference[:,:,dist_indices]
        return torch.sum(difference)

    def calculate_noobj_loss(self, pred_tensor, gt_tensor, bb_selection):
        """
        Takes in the model prediction and ground truth and outputs the loss from having no object in the cell for that batch.
        For reference, this is the 4th part of the 5 part loss function.
        """
        pred_tensor, gt_tensor = pred_tensor[:,:,list(x for x in range(5*self.B) if x%5==4)], gt_tensor[:,:,list(x for x in range(5*self.B) if x%5==4)]
        running_loss = 0
        for batch, bbox in itertools.product(range(gt_tensor.shape[0]), range(self.B)):
            noobj_mask = torch.ones_like(gt_tensor)
            for cell in range(49):
                noobj_mask[batch, cell, int(bb_selection[batch, cell, bbox])] = 0
        pred_tensor, gt_tensor = pred_tensor*noobj_mask, gt_tensor*noobj_mask
        return self.lambda_n*torch.sum(torch.square(pred_tensor-gt_tensor))

    def calculate_class_loss(self, pred_tensor, gt_tensor):
        """
        Takes in the model output and ground truth and outputs the class loss for that batch.
        For reference, this is the last part of the 5 part loss function.
        """
        pred_tensor = pred_tensor[:,:,list((5*x)+4 for x in range(self.B))+list(range((5*self.B),(5*self.B)+self.C))]
        split_gt = gt_tensor.reshape(gt_tensor.shape[0], 49, int(gt_tensor.shape[2]/5), 5)
        mask = torch.sign(torch.sum(split_gt, dim=3))
        actual_classes = gt_tensor[:,:,list((5*x)+4 for x in range(self.B))]
        max_prediction, predicted_classes = torch.max(pred_tensor[:,:,self.B:], dim=2)
        max_prediction, predicted_classes = max_prediction.unsqueeze(2).repeat(1,1,2), predicted_classes.unsqueeze(2).repeat(1,1,2)
        predictions = max_prediction*pred_tensor[:,:,range(self.B)]
        truths = (actual_classes==predicted_classes).type(torch.uint8)
        return torch.sum(mask*torch.square(truths-predictions))
    
    def forward(self, outputs, target):
        """
        outputs = Tensor of size [batch_size,49,(5*B)+C]
        target = Tensor of size [batch_size,objects,5] with each entry [x,y,w,h,class]
        """
        loss, batch_size = 0, target.shape[0]

        # Clean target and outputs tensor such that both are of the shape [batch, 49, 5*self.B]
        target = target.reshape(batch_size, target.shape[1]*5).unsqueeze(1).repeat(1,49,1)
        wh_numbers = list(x for x in range(target.shape[2]) if x%5==2 or x%5==3)
        target[:,:,wh_numbers] = torch.sqrt(target[:,:,wh_numbers])
        target = torch.nan_to_num(input=target, nan=-1)

        max_objects = int(target.shape[2]/5) 
        target_selection = torch.zeros_like(target) 
        for batch, obj in itertools.product(range(batch_size), range(max_objects)):
            if target[batch][0][5*obj] == -1: continue
            x,y = int(7*target[batch][0][5*obj]), int(7*target[batch][0][1+(5*obj)])
            if x>=0: target_selection[batch, (7*y)+x,(obj*5):(obj*5)+5] = 1
        truth_info = target_selection*target
        ground_truth = torch.zeros(batch_size, 49, 5*self.B)
        for batch, cell in itertools.product(range(batch_size), range(49)):
            ground_truth[batch][cell] = self.permute(truth_info[batch][cell])
        ground_truth_dims = ground_truth[:,:,list(x for x in range(5*self.B) if x%5!=4)] 
        needed_outputs = outputs[:,:,list(x for x in range(5*self.B) if x%5!=4)]
        wh_numbers = list(x for x in range(4*self.B) if x%4==2 or x%4==3)
        needed_outputs[:,:,wh_numbers] = torch.sqrt(needed_outputs[:,:,wh_numbers]) 
        t = self.iou_matrix(tensor_1=ground_truth_dims, tensor_2=needed_outputs).reshape(batch_size, 49, self.B, self.B)      
        prediction = outputs[:,:,range(5*self.B)]
        t_index = t.argmax(dim=2)

        # Calculates loss based on cleaned model output and ground truth
        ground_truth = self.construct_model_ground_truth(bb_selection=t_index, ground_truth=ground_truth, prediction=prediction)
        loss += self.calculate_obj_loss(pred_tensor=prediction, gt_tensor=ground_truth)
        loss += self.calculate_noobj_loss(pred_tensor=prediction, gt_tensor=ground_truth, bb_selection=t_index)
        loss += self.calculate_class_loss(pred_tensor=outputs, gt_tensor=ground_truth)
        
        return loss/batch_size