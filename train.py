import pdb
import os
import os.path as osp
import time
import torch
import torchvision
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.networks import VisualNet, VisualNetDilated, AudioNet, AssoConv, APNet, weights_init, Rearrange
from tensorboardX import SummaryWriter

def compute_loss(output, audio_mix, loss_criterion, prefix="stereo", weight=1):
    loss_dict = {}
    
    key = '{}_loss'.format(prefix)
    loss_dict[key] = loss_criterion(output['binaural_spectrogram'], output['audio_gt'].detach()) * weight

    if 'pred_left' in output:
        fusion_loss1 = loss_criterion(2*output['pred_left']-audio_mix[:,:,:-1,:], output['audio_gt'].detach())
        fusion_loss2 = loss_criterion(audio_mix[:,:,:-1,:]-2*output['pred_right'], output['audio_gt'].detach())
        key = '{}_loss_fusion'.format(prefix)
        loss_dict[key] = (fusion_loss1 / 2 + fusion_loss2 / 2) * weight 

    return loss_dict

def save_model(net_audio, net_visual, net_fusion, opt, suffix=''):
    torch.save(net_visual.module.state_dict(), osp.join('.', opt.checkpoints_dir, opt.name, 'visual_{}.pth'.format(suffix)))
    torch.save(net_audio.module.state_dict(), osp.join('.', opt.checkpoints_dir, opt.name, 'audio_{}.pth'.format(suffix)))
    if net_fusion is not None:
        torch.save(net_fusion.module.state_dict(), osp.join('.', opt.checkpoints_dir, opt.name, 'fusion_{}.pth'.format(suffix)))

def create_optimizer(nets, opt):
    (net_visual, net_audio, net_fusion) = nets
    param_groups = [
                {'params': net_visual.parameters(), 'lr': opt.lr_visual},
                {'params': net_audio.parameters(), 'lr': opt.lr_audio}
            ]
    if net_fusion is not None:
        param_groups.append({'params': net_fusion.parameters(), 'lr': opt.lr_fusion})
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

#used to display validation loss
def display_val(nets, loss_criterion, writer, index, data_loader_val, opt, return_key):
    (net_visual, net_audio, net_fusion, net_rearrange) = nets
    val_loss_log = {} 
    with torch.no_grad():
        for i, val_data in enumerate(data_loader_val):
            if i < opt.validation_batches:
                val_total_loss = {}
                if 'stereo' in opt.dataset_mode:
                    audio_diff = val_data['audio_diff_spec'].to(opt.device)
                    audio_mix = val_data['audio_mix_spec'].to(opt.device)
                    visual_input = val_data['frame'].to(opt.device)
                    vfeat = net_visual(visual_input)
                    if net_fusion is not None:
                        upfeatures, output = net_audio(audio_diff, audio_mix, vfeat, return_upfeatures=True)
                        output.update(net_fusion(audio_mix, vfeat, upfeatures))
                    else:
                        output = net_audio(audio_diff, audio_mix, vfeat)

                    val_total_loss.update(compute_loss(output, audio_mix, loss_criterion, prefix='stereo', weight=opt.stereo_loss_weight))

                if 'sep' in opt.dataset_mode:
                    frame_sep = val_data['frame_sep']
                    sep_diff = val_data['sep_diff_spec'].to(opt.device)
                    sep_mix = val_data['sep_mix_spec'].to(opt.device)
                    assert isinstance(frame_sep, list)
                    img_feat1 = net_visual(frame_sep[0].to(opt.device))
                    img_feat2 = net_visual(frame_sep[1].to(opt.device))
                    img_feat = net_rearrange(img_feat1, img_feat2)
                    if net_fusion is not None: 
                        upfeatures, output_sep = net_audio(sep_diff, sep_mix, img_feat, return_upfeatures=True)
                        output_sep.update(net_fusion(sep_mix, img_feat, upfeatures))
                    else:
                        output_sep = net_audio(sep_diff, sep_mix, img_feat)

                    val_total_loss.update(compute_loss(output_sep, sep_mix, loss_criterion, prefix='sep', weight=opt.sep_loss_weight))

                for loss_name, loss_value in val_total_loss.items():
                    if loss_name not in val_loss_log:
                        val_loss_log[loss_name] = [loss_value.item()]
                    else:
                        val_loss_log[loss_name].append(loss_value.item())
            else:
                break

    avg_val_loss_log = {}
    print("--- Val loss info ---")
    for key, value in val_loss_log.items():
        avg_value = sum(value) / len(value)
        avg_val_loss_log[key] = avg_value 
        print("val_{}: {:.3f}".format(key, avg_value))
        if opt.tensorboard:
            writer.add_scalar('data/val_{}'.format(key), avg_value, index)
    print("\n")

    return avg_val_loss_log[return_key] 

#parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

#construct data loader
data_loader = CreateDataLoader(opt)

#create validation set data loader if validation_on option is set
if opt.validation_on:
    #temperally set to val to load val data
    opt.mode = 'val'
    data_loader_val = CreateDataLoader(opt)
    opt.mode = 'train' #set it back

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

## build network
# visual net
original_resnet = torchvision.models.resnet18(pretrained=True)
if opt.visual_model == 'VisualNet':
    net_visual = VisualNet(original_resnet)
elif opt.visual_model == 'VisualNetDilated':
    net_visual = VisualNetDilated(original_resnet)
else:
    raise TypeError("please input correct visual model type")

if len(opt.weights_visual) > 0:
    print('Loading weights for visual stream')
    net_visual.load_state_dict(torch.load(opt.weights_visual), strict=True)

# rearrange module
net_rearrange = Rearrange()

# audio net
net_audio = AudioNet(
    ngf=opt.unet_ngf,
    input_nc=opt.unet_input_nc,
    output_nc=opt.unet_output_nc,
    norm_mode=opt.norm_mode
)
net_audio.apply(weights_init)
if len(opt.weights_audio) > 0:
    print('Loading weights for audio stream')
    net_audio.load_state_dict(torch.load(opt.weights_audio), strict=True)

# fusion net
if opt.fusion_model == 'none':
    net_fusion = None
elif opt.fusion_model == 'AssoConv':
    net_fusion = AssoConv(norm_mode=opt.norm_mode)
elif opt.fusion_model == 'APNet':
    net_fusion = APNet(norm_mode=opt.norm_mode)
else:
    raise TypeError("Please input correct fusion model type") 

if net_fusion is not None and len(opt.weights_fusion) > 0:
    net_fusion.load_state_dict(torch.load(opt.weights_fusion), strict=True)

# data parallel
nets = (net_visual, net_audio, net_fusion)
net_visual.to(opt.device)
net_visual = torch.nn.DataParallel(net_visual, device_ids=opt.gpu_ids)
net_audio.to(opt.device)
net_audio = torch.nn.DataParallel(net_audio, device_ids=opt.gpu_ids)
net_rearrange.to(opt.device)
net_rearrange = torch.nn.DataParallel(net_rearrange, device_ids=opt.gpu_ids)
if net_fusion is not None:
    net_fusion.to(opt.device)
    net_fusion = torch.nn.DataParallel(net_fusion, device_ids=opt.gpu_ids)

# set up optimizer
optimizer = create_optimizer(nets, opt)

# set up loss function
if opt.loss_mode == 'mse':
    loss_criterion = torch.nn.MSELoss()
elif opt.loss_mode == 'l1':
    loss_criterion = torch.nn.L1Loss()
else:
    raise TypeError("Please use correct loss mode")
if len(opt.gpu_ids) > 0:
    loss_criterion.cuda(opt.gpu_ids[0])

# initialization
total_steps = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
loss_log = {}
best_err = float("inf")

for epoch in range(1, opt.niter+1):
    torch.cuda.synchronize()
    epoch_start_time = time.time()

    if opt.measure_time:
        iter_start_time = time.time()
    for i, data in enumerate(data_loader):
        if opt.measure_time:
            torch.cuda.synchronize()
            iter_data_loaded_time = time.time()

        total_steps += opt.batchSize

        total_loss = {}
        # forward
        if 'stereo' in opt.dataset_mode:
            audio_diff = data['audio_diff_spec'].to(opt.device)
            audio_mix = data['audio_mix_spec'].to(opt.device)
            visual_input = data['frame'].to(opt.device)
            vfeat = net_visual(visual_input)
            if net_fusion is not None:
                upfeatures, output = net_audio(audio_diff, audio_mix, vfeat, return_upfeatures=True) 
                output.update(net_fusion(audio_mix, vfeat, upfeatures))
            else:
                output = net_audio(audio_diff, audio_mix, vfeat)

            total_loss.update(compute_loss(output, audio_mix, loss_criterion, prefix='stereo', weight=opt.stereo_loss_weight))

        if 'sep' in opt.dataset_mode:
            frame_sep = data['frame_sep']
            sep_diff = data['sep_diff_spec'].to(opt.device)
            sep_mix = data['sep_mix_spec'].to(opt.device)
            assert isinstance(frame_sep, list)
            img_feat1 = net_visual(frame_sep[0].to(opt.device))
            img_feat2 = net_visual(frame_sep[1].to(opt.device))
            img_feat = net_rearrange(img_feat1, img_feat2)
            if net_fusion is not None: 
                upfeatures, output_sep = net_audio(sep_diff, sep_mix, img_feat, return_upfeatures=True)
                output_sep.update(net_fusion(sep_mix, img_feat, upfeatures))
            else:
                output_sep = net_audio(sep_diff, sep_mix, img_feat)

            total_loss.update(compute_loss(output_sep, sep_mix, loss_criterion, prefix='sep', weight=opt.sep_loss_weight))

        # parse loss
        loss = sum(_value for _key, _value in total_loss.items() if 'loss' in _key)
        for loss_name, loss_value in total_loss.items():
            if loss_name not in loss_log:
                loss_log[loss_name] = [loss_value.item()]
            else:
                loss_log[loss_name].append(loss_value.item())

        if opt.measure_time:
            torch.cuda.synchronize()
            iter_data_forwarded_time = time.time()

        # update optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if opt.measure_time:
            iter_model_backwarded_time = time.time()
            data_loading_time.append(iter_data_loaded_time - iter_start_time)
            model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
            model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

        if total_steps // opt.batchSize % opt.display_freq == 0:
            print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
            for key, value in loss_log.items():
                avg_value = sum(value) / len(value)
                print("{}: {:.3f}".format(key, avg_value))
                if opt.tensorboard:
                    writer.add_scalar('data/{}'.format(key), avg_value, total_steps) 
            print("\n")
            loss_log = {} 
            if opt.measure_time:
                print('average data loading time: ' + str(sum(data_loading_time)/len(data_loading_time)))
                print('average forward time: ' + str(sum(model_forward_time)/len(model_forward_time)))
                print('average backward time: ' + str(sum(model_backward_time)/len(model_backward_time)))
                data_loading_time = []
                model_forward_time = []
                model_backward_time = []

        if total_steps // opt.batchSize % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            save_model(net_audio, net_visual, net_fusion, opt, suffix='latest')

        if total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on:
            net_visual.eval()
            net_audio.eval()
            if net_fusion is not None:
                net_fusion.eval()
            opt.mode = 'val'
            print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
            nets = (net_visual, net_audio, net_fusion, net_rearrange)
            val_err = display_val(nets, loss_criterion, writer, total_steps, data_loader_val, opt, return_key=opt.val_return_key)
            net_visual.train()
            net_audio.train()
            if net_fusion is not None:
                net_fusion.train()
            opt.mode = 'train'
            #save the model that achieves the smallest validation error
            if val_err < best_err:
                best_err = val_err
                print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                save_model(net_audio, net_visual, net_fusion, opt, suffix='best')

        if opt.measure_time:
            iter_start_time = time.time()

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
        save_model(net_audio, net_visual, net_fusion, opt, suffix=str(epoch))

    #decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
    if opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0:
        decrease_learning_rate(optimizer, opt.decay_factor)
        print('decreased learning rate by ', opt.decay_factor)
