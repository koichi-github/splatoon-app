import torch
import numpy as np

import os


def fit(train_loader, test_loader, model, loss_fn, optimizer, n_epochs, device, log_interval, save_epoch_interval, data_dirname, num_out):
    
    train_loss_value=[]      #trainingのlossを保持するlist
    train_acc_value=[]       #trainingのaccuracyを保持するlist
    test_loss_value=[]       #testのlossを保持するlist
    test_acc_value=[]        #testのaccuracyを保持するlist 

    for epoch in range(n_epochs):

        message = 'Epoch: {}/{}'.format(epoch + 1, n_epochs)

        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval)

        # Train test stage
        train_mean_loss, train_acc = test_epoch(train_loader, model, optimizer, loss_fn, device)

        train_loss_value.append(train_mean_loss)
        train_acc_value.append(train_acc)

        message += '\n\tTrain set: Average loss: {:.4f}'.format(train_mean_loss)
        message += '\n\t           Accuracy rate: {:.2%}'.format(train_acc)

        # Test stage
        test_mean_loss, test_acc = test_epoch(test_loader, model, optimizer, loss_fn, device)

        test_loss_value.append(test_mean_loss)
        test_acc_value.append(test_acc)

        message += '\n\tTest set: Average loss: {:.4f}'.format(test_mean_loss)
        message += '\n\t          Accuracy rate: {:.2%}'.format(test_acc)

        print(message)

        outdir = f"checkpoints/{data_dirname}/"
        os.makedirs(os.path.dirname(outdir), exist_ok=True)

        if (epoch+1) % save_epoch_interval == 0:
            torch.save(model.state_dict(), f"{outdir}model_out{num_out}_epoch%d.pth" % epoch)


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metric_fn=None):
    
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (_, data, labels) in enumerate(train_loader):
        
        data = data.to(device)
        labels = labels.to(device).long()
        for i in range(len(data)):
            if _[i].split("/")[-1] == "img9.png":
                print(data[i])
                exit()
    
        optimizer.zero_grad()
        outputs = model(data)

        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            
            print(message)
            losses = []



def test_epoch(test_loader, model, optimizer, loss_fn, device):
    with torch.no_grad():
        
        model.eval()


        sum_correct = 0         #正解率の合計
        sum_total = 0           #dataの数の合計        
        test_loss = 0
        for batch_idx, (img_path, data, labels) in enumerate(test_loader):
            
            data = data.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(data)

            for i in range(len(outputs)):
                print(outputs[i].data, labels[i], img_path[i].split("/")[-1])

            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1) 
            sum_total += labels.size(0)
            sum_correct += (predicted == labels).sum().item()

        test_mean_loss = test_loss*test_loader.batch_size/len(test_loader.dataset)
        accuracy = float(sum_correct/sum_total)
            
    return test_mean_loss, accuracy