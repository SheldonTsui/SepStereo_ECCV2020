import torch

def CreateDataLoader(opt):
    dataset = None
    if opt.dataset_mode == 'stereo':
        from data.stereo_dataset import StereoDataset
        dataset = StereoDataset(opt)
    elif opt.dataset_mode == 'sep':
        from data.sep_dataset import SepDataset
        dataset = SepDataset(opt)
    elif opt.dataset_mode == 'sepstereo':
        from data.sepstereo_dataset import SepStereoDataset
        dataset = SepStereoDataset(opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    print("#%s clips = %d" %(opt.mode, len(dataset)))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=opt.mode=='train',
        num_workers=int(opt.nThreads),
        drop_last=opt.mode=='train'
    )

    return dataloader
