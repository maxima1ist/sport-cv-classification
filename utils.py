import torch
from tqdm import tqdm


def train_model(model,
                data_loader,
                dataset_size,
                optimizer,
                scheduler,
                criterion,
                num_epochs,
                device):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        scheduler.step()
        model.train()

        running_loss = 0.0
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / dataset_size
        print('Loss: {:.4f}'.format(epoch_loss))
    return model
