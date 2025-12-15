"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-10:18
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch


def train_epoch_one(
        args,
        model,
        train_dataloader,
        optimizer,
        lr_scheduler,
        epoch,
        global_loss,
        count_loss,
        device
):
    for step, (imgs, dens) in enumerate(train_dataloader):
        imgs = imgs.to(device)
        dens = dens.to(device)

        pred_dens = model(imgs)
        optimizer.zero_grad()
        dens = dens.squeeze()
        pred_dens = pred_dens.squeeze()

        g_loss = global_loss(dens, pred_dens)
        c_loss = count_loss(torch.sum(dens), torch.sum(pred_dens))
        loss = g_loss + c_loss

        loss.backward()
        optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()

        if step % 5 == 0:
            print('[{}/{}] lr: {} g_loss: {}  c_loss: {} '.format(
                step,
                epoch,
                lr_scheduler.get_lr(),
                g_loss.item(),
                c_loss.item(),
            ))


