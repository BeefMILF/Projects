from torchaudio import transforms as TA
from torchvision import transforms as T
import torch


class ToTensor:
    """
    convert ndarrays in sample to Tensors.
    return:
        feat(torch.FloatTensor)
        label(torch.LongTensor of size batch_size x 1)
    """
    def __call__(self, data):
        return torch.from_numpy(data).float()


base = T.Compose([ToTensor()])


transform = {
    'val': {'base': base},
    'test': {'base': base},
    'train': {'base': base},
}


# Augmentation
logmel_aug = T.Compose([TA.TimeMasking(time_mask_param=30),
                        TA.FrequencyMasking(freq_mask_param=15)])

logmel_A = T.Lambda(lambd=lambda x: torch.cat((x, logmel_aug(x)), dim=2))


augment = {
    'val': {'logmel': logmel_A},
    'test': {'logmel': logmel_A},
    'train': {'logmel': logmel_A},
}