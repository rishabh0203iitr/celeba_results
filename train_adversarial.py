import os
import pickle
from tqdm import tqdm
from datetime import datetime

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
from torch.utils.data.dataset import Subset
from torchvision import transforms as T
import torch.nn.functional as F

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.util import get_model
from util import MultiDimAverageMeter
import attention

os.environ['TORCH_HOME'] = 'raid/ysharma_me/fair_lr/dnew' 

@ex.automain
def train(
    main_tag,
    dataset_tag,
    model_tag,
    data_dir,
    log_dir,
    device,
    target_attr_idx,
    bias_attr_idx,
    main_num_steps,
    main_valid_freq,
    main_batch_size,
    main_optimizer_tag,
    main_learning_rate,
    main_weight_decay,
):

    print(dataset_tag)

    device = 'cuda'

    # log_dir = "/raid/ysharma_me/fair_lr/LfF/log"
#    data_dir = "/raid/ysharma_me/fair_lr/LfF/datasets/debias"

    log_dir = "/raid/ysharma_me/fair_lr/LfF/workspace_adv_18/debias/log"


    log_writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))

    writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))

    train_dataset = get_dataset(
        dataset_tag,
        data_dir=data_dir,
        dataset_split="train",
        transform_split="train"
    )

    valid_dataset = get_dataset(
        dataset_tag,
        data_dir=data_dir,
        dataset_split="eval",
        transform_split="eval"
    )

    train_target_attr = train_dataset.attr[:, target_attr_idx]
    train_bias_attr = train_dataset.attr[:, bias_attr_idx]
    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    attr_dims.append(torch.max(train_bias_attr).item() + 1)
    num_classes = attr_dims[0]

    train_dataset = IdxDataset(train_dataset)
    valid_dataset = IdxDataset(valid_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    # define model and optimizer
    
    F_E = get_model(model_tag, attr_dims[0]).to(device)
    D = attention.Discriminator().to(device)
    C = attention.Classifier().to(device)
    Dummy_C = attention.Dummy_Classifier().to(device)


#    if main_optimizer_tag == "SGD":
#       optimizer1 = torch.optim.SGD(
#            params=list(D.parameters()),
#            lr=main_learning_rate,
#            weight_decay=main_weight_decay,
#            momentum=0.9,
#       )
#        optimizer2 = torch.optim.SGD(
#            params=list(F_E.parameters()) + list(C.parameters()),
#            lr=main_learning_rate,
#            weight_decay=main_weight_decay,
#            momentum=0.9,
#        )
#        optimizer3 = torch.optim.SGD(
#            params=list(F_E.parameters()),
#            lr=main_learning_rate,
#            weight_decay=main_weight_decay,
#            momentum=0.9,
#        )
    if main_optimizer_tag == "Adam":
        optimizer1 = torch.optim.Adam(
            params=list(D.parameters()),
            lr=1e-4,
            weight_decay=main_weight_decay,
        )
        optimizer2 = torch.optim.Adam(
            params=list(F_E.parameters()) + list(C.parameters()),
            lr=1e-3,
            weight_decay=main_weight_decay,
        )
        optimizer3 = torch.optim.Adam(
            params=list(F_E.parameters()),
            lr=1e-6,
            weight_decay=main_weight_decay,
        )
#    elif main_optimizer_tag == "AdamW":
#        optimizer1 = torch.optim.AdamW(
#            params=list(D.parameters()),
#            lr=main_learning_rate,
#            weight_decay=main_weight_decay,
#        )
#        optimizer2 = torch.optim.AdamW(
#            params=list(F_E.parameters()) + list(C.parameters()),
#            lr=main_learning_rate,
#            weight_decay=main_weight_decay,
#        )
#        optimizer3 = torch.optim.AdamW(
#            params=list(F_E.parameters()),
#            lr=main_learning_rate,
#            weight_decay=main_weight_decay,
#        )
    else:
        raise NotImplementedError

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    label_criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # define evaluation function
    def evaluate(F_E, C, Dummy_C, D, data_loader):
        F_E.eval()
        C.eval()
        D.eval()
        Dummy_C.load_state_dict(C.state_dict())
        Dummy_C.eval()

        acc = 0
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for index, data, attr in tqdm(data_loader, leave=False):
            label = attr[:, target_attr_idx]
            data = data.to(device)
            attr = attr.to(device)
            label = label.to(device)
            with torch.no_grad():
                features = F_E(data)
                logit = Dummy_C(features)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()

            attr = attr[:, [target_attr_idx, bias_attr_idx]]

            attrwise_acc_meter.add(correct.cpu(), attr.cpu())

        accs = attrwise_acc_meter.get_mean()

        F_E.train()
        C.train()
        D.train()

        return accs

    # define extracting indices function
    def get_align_skew_indices (lookup_list, indices):
        '''
        lookup_list:
            A list of non-negative integer. 0 should indicate bias-align sample and otherwise(>0) indicate bias-skewed sample.
            Length of (lookup_list) should be the number of unique samples
        indices:
            True indices of sample to look up.
        '''
        pseudo_bias_label = lookup_list[indices]
        skewed_indices = (pseudo_bias_label != 0).nonzero().squeeze(1)
        aligned_indices = (pseudo_bias_label == 0).nonzero().squeeze(1)

        return aligned_indices, skewed_indices

    def normal_evaluate(F_E, C, Dummy_C, D, data_loader):
        F_E.eval()
        C.eval()
        D.eval()
        Dummy_C.load_state_dict(C.state_dict())
        Dummy_C.eval()

        acc = 0
        correct = 0
        total = 0
        # attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for index, data, attr in tqdm(data_loader, leave=False):
            label = attr[:, target_attr_idx]
            data = data.to(device)
            attr = attr.to(device)
            label = label.to(device)
            with torch.no_grad():
                features = F_E(data)
                logit = Dummy_C(features)
                _, predicted = logit.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()

            # attr = attr[:, [target_attr_idx, bias_attr_idx]]

            # attrwise_acc_meter.add(correct.cpu(), attr.cpu())

        # accs = attrwise_acc_meter.get_mean()

        F_E.train()
        C.train()
        D.train()

        return correct/total


    valid_attrwise_accs_list = []

    total=0
    Dtotal=0
    correct=0
    Dcorrect=0

    best=0.0
    best2=0.0

    for step in tqdm(range(main_num_steps)):
        try:
            index, data, attr = next(train_iter)
        except:
            train_iter = iter(train_loader)
            index, data, attr = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)

        label = attr[:, target_attr_idx]
        domain = attr[:, bias_attr_idx]

        #update discriminator
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        feat = F_E(data)
        Dlogit = D(feat)
        loss_per_sample1 = label_criterion(Dlogit.squeeze(1), domain)

        loss1 = loss_per_sample1.mean()
        loss1.backward(retain_graph=True)
        optimizer1.step()
        optimizer3.step()


        # #calculate discriminator gradients
        # optimizer1.zero_grad()
        # optimizer2.zero_grad()
        # optimizer3.zero_grad()

        # fea=F_E(data)
        # log=D(fea)
        # l,_=torch.max(log, 1)
        # l.backward((torch.ones(l.shape)).to(device))

        # _, predicted_d = log.max(1)
        # dcor=predicted_d.eq(domain)
        # dcor=dcor.reshape(dcor.shape[0],1)

        # dcor=dcor*1
        # dincor=1-dcor

        # Dgradients = D.get_activations_gradient()
        # Dgradients1=Dgradients.detach().clone()
        # Dgradients1[Dgradients1>0]=0
        # Dgradients1[Dgradients1<0]=1
        # Dgradients1=Dgradients1*dincor

        
        # a = ((Dgradients**2).sum(axis=1))**(0.5)
        # a =a.reshape(a.shape[0],1)
        # a=a+1e-12

        # Dgradients=Dgradients/a
        # Dgradients=Dgradients*dcor

        # Dgradients[Dgradients>0]=1
        # Dgradients[Dgradients<0]=0
        # Dgradients=(1-(Dgradients+Dgradients1))

        #udate classifier
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        features = F_E(data)
        logit = C(features)
        loss_per_sample2 = label_criterion(logit.squeeze(1), label)

        loss2 = loss_per_sample2.mean()
        loss2.backward()
        optimizer2.step()

        _, predicted = logit.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        accuracy = correct*100. / total

        _, Dpredicted = Dlogit.max(1)
        Dtotal += domain.size(0)
        Dcorrect += Dpredicted.eq(domain).sum().item()
        Daccuracy = Dcorrect*100. / Dtotal
        
        main_log_freq = 636
        # if step % main_log_freq == 0:
        #     loss2 = loss2.detach().cpu()
        #     writer.add_scalar("loss/train", loss2, step/636)

        #     bias_attr = attr[:, bias_attr_idx]  # oracle
        #     loss_per_sample1 = loss_per_sample1.detach()
        #     if (label == bias_attr).any().item():
        #         aligned_loss = loss_per_sample1[label == bias_attr].mean()
        #         writer.add_scalar("loss/train_aligned", aligned_loss, step/636)

        #     if (label != bias_attr).any().item():
        #         skewed_loss = loss_per_sample[label != bias_attr].mean()
        #         writer.add_scalar("loss/train_skewed", skewed_loss, step/636)

        log_freq = 636
        if step % log_freq == 0:
            loss2 = loss2.detach().cpu()
            writer.add_scalar("loss/train", loss2, step/636)
            test_accuracy = normal_evaluate(F_E, C, Dummy_C, D, valid_loader)
            if test_accuracy>=best2:
                best2=test_accuracy
            log_writer.add_scalars('Accuracy_Classifier and Accuracy_Discriminator', {'Accuracy_Classifier': accuracy, 'Accuracy_Discriminator':Daccuracy}, step/636)
            log_writer.add_scalars('Test_accuracy', {'Test_accuracy': test_accuracy}, step/636)
            log_writer.close()
            bias_attr = attr[:, bias_attr_idx]  # oracle
            loss_per_sample2 = loss_per_sample2.detach()
            if (label == bias_attr).any().item():
                aligned_loss = loss_per_sample2[label == bias_attr].mean()
                writer.add_scalar("loss/train_aligned", aligned_loss, step/636)

            if (label != bias_attr).any().item():
                skewed_loss = loss_per_sample2[label != bias_attr].mean()
                writer.add_scalar("loss/train_skewed", skewed_loss, step/636)

        if step % log_freq == 0:
            valid_attrwise_accs = evaluate(F_E, C, Dummy_C, D, valid_loader)
            valid_attrwise_accs_list.append(valid_attrwise_accs)
            valid_accs = torch.mean(valid_attrwise_accs)
            if valid_accs>=best:
                best=valid_accs
            writer.add_scalar("acc/valid", valid_accs, step/636)
            eye_tsr = torch.eye(num_classes)
            writer.add_scalar(
                "acc/valid_aligned",
                valid_attrwise_accs[eye_tsr > 0.0].mean(),
                step/636
            )
            writer.add_scalar(
                "acc/valid_skewed",
                valid_attrwise_accs[eye_tsr == 0.0].mean(),
                step/636
            )

    print(best)
    print(best2)
    os.makedirs(os.path.join(log_dir, "result", main_tag), exist_ok=True)
    result_path = os.path.join(log_dir, "result", main_tag, "result.th")
    valid_attrwise_accs_list = torch.cat(valid_attrwise_accs_list)
    with open(result_path, "wb") as f:
        torch.save({"valid/attrwise_accs": valid_attrwise_accs_list}, f)

    

    model_path = os.path.join(log_dir, "result", main_tag, "model.th")
    state_dict = {
        'steps': step,
        'F_E': F_E.state_dict(),
        'D':D.state_dict(),
        'C':C.state_dict(),
        'optimizer2': optimizer2.state_dict(),
    }
    with open(model_path, "wb") as f:
        torch.save(state_dict, f)

    info = {"valid_accs": best,
            "accs ": best2}
    with open(os.path.join(log_dir, 'result.txt'), 'w') as f:
        f.write(str(info))




