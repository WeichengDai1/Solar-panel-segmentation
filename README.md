# Solar-panel-segmentation
Original dataset can be found in this link: 
https://doi.org/10.6084/m9.figshare.c.3255643.v1

This is a solar panel segmentation in computer vision. Here are some examples:

![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/input1.jpg)
![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/predict1.jpg)
![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/true1.jpg)

![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/input2.jpg)
![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/predict2.jpg)
![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/true2.jpg)

![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/input3.jpg)
![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/predict3.jpg)
![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/true3.jpg)

![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/input4.jpg)
![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/predict4.jpg)
![image](https://github.com/WeichengDai1/Solar-panel-segmentation/blob/main/IMG/true4.jpg)

# Update
Maybe we can try Dice loss:

def dice_loss(score, target):

    target = target.float()
    
    smooth = 1e-5
    
    intersect = torch.sum(score * target)
    
    y_sum = torch.sum(target * target)
    
    z_sum = torch.sum(score * score)
    
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    
    loss = 1 - loss
    
    return loss
