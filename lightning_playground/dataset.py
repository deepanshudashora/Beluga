import pytorch_lightning
class CIFARDataModule(pytorch_lightning.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    train_transforms = A.Compose(
        [
          A.Normalize(mean, std),
          A.PadIfNeeded(40, 40, p=1),
          A.RandomCrop(32, 32, p=1),
          # A.Sequential([A.CropAndPad(px=4, keep_size=False), #padding of 4, keep_size=True by default
          #                   A.RandomCrop(32,32)])
          A.IAAFliplr(always_apply=True),
          A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8,
                                min_width=8, fill_value=[0.49139968, 0.48215841, 0.44653091], always_apply=True),
          ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
          A.Normalize(mean, std),
          ToTensorV2(),
        ]
    )
    self.train = Cifar10SearchDataset(root='./data', train=True,
                                            download=True, transform=train_transforms)
    self.test = Cifar10SearchDataset(root='./data', train=False,
                                          download=True, transform=test_transforms)

  def train_dataloader(self):
    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    return train_loader



  def val_dataloader(self):
    # test dataloader
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return test_loader