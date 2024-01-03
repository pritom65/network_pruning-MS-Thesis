from copy import deepcopy
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def train_one_epoch(model,optimizer,train_loder,dice_loss,device,epoch):
    print_loss,print_dice = 0,0
    for X,y in tqdm(train_loder,leave = False,desc = f'Epoch: {str(epoch)}'):
        X,y = X.to(device),y.to(device)
        model.train()
        pred = model(X)
        loss = dice_loss(pred,y)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        print_loss += loss.detach()
        with torch.no_grad():
            print_dice += dice_loss(pred,y,val_loss = True)[1]
        
    return print_loss.to('cpu')*train_loder.batch_size/len(train_loder.dataset),print_dice.to('cpu')*train_loder.batch_size/len(train_loder.dataset)

def val_one_epoch(model,val_loder,dice_loss,device,epoch):
    print_dice = 0
    for X,y in tqdm(val_loder,leave = False,desc = f'Epoch: {str(epoch)}'):
        X,y = X.to(device),y.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(X)
            print_dice += dice_loss(pred,y,val_loss = True)[1]

    return print_dice.to('cpu')*val_loder.batch_size/len(val_loder.dataset)



def train_model(model,optimizer,train_loder,val_loder,dice_loss,device,train_one_epoch,val_one_epoch,max_counter = 10,max_epoch = 100):
    counter,max_dice = 0,0
    for epoch in range(max_epoch):
        torch.mps.empty_cache()
        train_loss = train_one_epoch(model,optimizer,train_loder,dice_loss,device,epoch)
        val_loss = val_one_epoch(model,val_loder,dice_loss,device,epoch)
        counter += 1
        if epoch % 5 == 0:
            print(f'Epoch: {epoch} | Train Loss: {train_loss[0]:.5f} | Train Dice: {train_loss[1]:.5f} | Val Dice: {val_loss:.5f} | Max Val Dice: {max_dice:.5f} | Counter: {counter}')
        if max_dice < val_loss:
            counter = 0
            max_dice = val_loss
            model_parms = deepcopy(model.state_dict())
        elif counter > max_counter:
            break
    return model_parms
