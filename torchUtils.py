import torch
def one_hot(index_list,class_num):
    '''
    Convert indices to one-hot encoded tensor
    :param index_list: [2,1,3,0]
    :param class_num: number of classes
    :return:
    tensor([[0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.]])
    '''
    # if type(index_list) == torch.Tensor:
    #     index_list = index_list.detach().numpy()
    # create tensors on available device (CPU if CUDA unavailable)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not isinstance(index_list, torch.Tensor):
        indexes = torch.tensor(index_list, dtype=torch.long, device=device).view(-1, 1)
    else:
        indexes = index_list.view(-1, 1).to(device)
    out = torch.zeros(len(index_list), class_num, device=device)
    out = out.scatter_(dim=1, index=indexes, value=1)
    return out


