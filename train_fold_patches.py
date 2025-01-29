import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import argparse
import datetime
from torch.distributions import Normal
from torchvision.utils import save_image

from utils.file_utils import *
from utils.visualize import *
import torch.distributed as dist
from datasets.shapenet_data_pc import ShapeNet15kPointClouds

from copy import deepcopy
from collections import OrderedDict
from pgi_models import folding_model_configurers

from tensorboardX import SummaryWriter

from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
from metrics.PyTorchEMD.emd import earth_mover_distance
from geomloss import SamplesLoss
sinkhorn_distance = SamplesLoss("sinkhorn", p=2, blur=.005)

class Model(nn.Module):
    def __init__(self, args, loss_type: str):
        super(Model, self).__init__()
        self.k_local = args.patch_size ** 2
        self.n_patches = args.n_patches
        args.N = self.k_local
        self.loss_type = loss_type
        self.model = folding_model_configurers[args.model_type]()

    def get_loss_iter(self, data):
        B, D, N = data.shape                           # [16, 3, 2048]
        data = data.transpose(1,2)                      # [16, 2048, 3]
        centers = np.random.choice(N, self.n_patches, replace=False) # [256]
        data_centers = data[:, centers]                 # [16, 256, 3]
        local_groups = knn_points(data_centers, data, K = self.k_local, return_nn=True)[-1] # [16, 2048, args.k_local, 3]
        patches = local_groups.view(B*self.n_patches, self.k_local, 3) # [16*256, 16, 3]
        # normalize patches
        patches = patches - patches.mean(dim=1, keepdim=True)
        patches = patches / patches.norm(dim=2, keepdim=True).max()
        fold_rec, grids = self.model(patches)
        if self.loss_type == 'cd_1':
            loss, _ = chamfer_distance(patches, fold_rec, batch_reduction='mean', norm=1)
        elif self.loss_type == 'cd_2':
            loss, _ = chamfer_distance(patches, fold_rec, batch_reduction='mean', norm=2)
        elif self.loss_type == 'cd_1_2':
            loss = chamfer_distance(patches, fold_rec, batch_reduction='mean', norm=1)[0] + chamfer_distance(patches, fold_rec, batch_reduction='mean', norm=2)[0]
        elif self.loss_type == 'emd':
            loss = earth_mover_distance(patches, fold_rec, transpose=False).mean()
        elif self.loss_type == 'sinkhorn':
            loss = sinkhorn_distance(patches, fold_rec).mean()
        return loss
    
    def get_test_loss_iter(self, data):
        fold_rec, patches = self.reconstruct(data)
        cd_loss, _ = chamfer_distance(patches, fold_rec, batch_reduction='mean', norm=2)
        emd_loss = earth_mover_distance(patches, fold_rec, transpose=False).mean()
        return {'cd_loss': cd_loss, 'emd_loss': emd_loss}
    
    def reconstruct(self, data):
        B, D, N = data.shape                           # [16, 3, 2048]
        data = data.transpose(1,2)                      # [16, 2048, 3]
        centers = np.random.choice(N, self.n_patches, replace=False) # [256]
        data_centers = data[:, centers]                 # [16, 256, 3]
        local_groups = knn_points(data_centers, data, K = self.k_local, return_nn=True)[-1] # [16, 2048, args.k_local, 3]
        patches = local_groups.view(B*self.n_patches, self.k_local, 3) # [16*256, 16, 3]
        patches = patches - patches.mean(dim=1, keepdim=True)
        patches = patches / patches.norm(dim=2, keepdim=True).max()
        fold_rec, grids = self.model(patches)
        return fold_rec, patches

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)

def get_dataset(dataroot, npoints,category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=category.split(','), split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=category.split(','), split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return tr_dataset, te_dataset

def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler

def train(gpu, opt, output_dir):
    set_seed(opt)
    logger = setup_logging(output_dir)

    if not opt.debug:
        if opt.use_tb:
            # tb writers
            tb_writer = SummaryWriter(output_dir)
            

    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

        opt.saveIter =  int(opt.saveIter / opt.ngpus_per_node)
        opt.diagIter = int(opt.diagIter / opt.ngpus_per_node)
        opt.vizIter = int(opt.vizIter / opt.ngpus_per_node)


    ''' data '''
    train_dataset, test_dataset = get_dataset(opt.dataroot, opt.npoints, opt.category)
    print('train_dataset size:', len(train_dataset))
    print('test_dataset size:', len(test_dataset))
    if opt.fps_points > 0:
        train_dataset.cache_fps_points(opt.fps_points)
        test_dataset.cache_fps_points(opt.fps_points)
        # np.savetxt('.train_fps_points.xyz', train_dataset[0]['train_points'].cpu().numpy(), fmt='%.6f')
    train_dataloader, test_dataloader, train_sampler, test_sampler = get_dataloader(opt, train_dataset, test_dataset)
    print('train_dataloader size:', len(train_dataloader))
    print('test_dataloader size:', len(test_dataloader))
    
    '''
    create networks
    '''
    model = Model(opt, opt.loss_type)

    # Note that parameter initialization is done within the DiT constructor
    # TODO: check ema model
    # if opt.use_ema:
    #     ema = deepcopy(model).to(gpu)  # Create an EMA of the model for use after training
    #     requires_grad(ema, False)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)


    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

    print("Model = %s" % str(model))
    total_params = sum(param.numel() for param in model.parameters())/1e6
    print("Total_params = %s MB " % str(total_params))   
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=opt.lr/100)
    
    start_epoch = 0
    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        
    # # Prepare models for training:
    # if opt.use_ema:
    #     update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    #     model.train()  # important! This enables embedding dropout for classifier-free guidance
    #     ema.eval()  # EMA model should always be in eval mode

    for epoch in range(start_epoch, opt.niter):

        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)

        for i, data in enumerate(train_dataloader):
            model.train()
            x = data['train_points'].transpose(1,2) # B x N x 3

            '''
                    train fold centers
                    '''

            if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                x = x.cuda(gpu)
            elif opt.distribution_type == 'single':
                x = x.cuda()

            loss = model.get_loss_iter(x).mean()
            optimizer.zero_grad()
            loss.backward()
                    # netpNorm, netgradNorm = getGradNorm(model)
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()

                    # if opt.use_ema:
                    #     update_ema(ema, model)

            if not opt.debug:
                global_step = i + len(train_dataloader) * epoch
                if opt.use_tb:
                    tb_writer.add_scalar('train_loss', loss.item(), global_step)
                    tb_writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], global_step)

            if i % opt.print_freq == 0 and should_diag:
                logger.info('[{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                                    .format(
                                epoch, opt.niter, loss.item()
                                ))
        scheduler.step()
        # Quantitative evaluation
        model.eval()
        with torch.no_grad():
            if (epoch + 1) % opt.diagIter == 0 and should_diag:

                logger.info('Diagnosis:')
                test_loss_cd = 0
                test_loss_emd = 0
                for i, data in enumerate(test_dataloader):
                    x = data['train_points'].transpose(1,2) # B x N x 3

                    '''
                    test fold centers
                    '''

                    if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                        x = x.cuda(gpu)
                    elif opt.distribution_type == 'single':
                        x = x.cuda()

                    test_loss = model.get_test_loss_iter(x)
                    test_loss_cd += test_loss['cd_loss'].item()
                    test_loss_emd += test_loss['emd_loss'].item()

                test_loss_cd /= len(test_dataloader)
                test_loss_emd /= len(test_dataloader)
                
                if not opt.debug:
                    global_step = len(test_dataloader) * epoch
                    if opt.use_tb:
                        tb_writer.add_scalar('test_loss_cd', test_loss_cd, global_step)
                        tb_writer.add_scalar('test_loss_emd', test_loss_emd, global_step)

                if should_diag:
                    logger.info('[{:>3d}/{:>3d}]   test_loss_cd: {:.4e},    test_loss_emd: {:.4e},    '
                                        .format(
                                    epoch, opt.niter, test_loss_cd, test_loss_emd
                                    ))

            # Qualitative evaluation
            if (epoch + 1) % opt.vizIter == 0 and should_diag:
                logger.info('Reconstruction: eval')
                batch = next(iter(test_dataloader))
                x = batch['train_points'].transpose(1,2)
                if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                    x = x.cuda(gpu)
                elif opt.distribution_type == 'single':
                    x = x.cuda()
                model.eval()
                recon, gt = model.reconstruct(x)
                os.makedirs('%s/vis/epoch_%03d' % (outf_syn, epoch), exist_ok=True)
                for idx in range(5):
                    np.savetxt('%s/vis/epoch_%03d/gt_%d.xyz' % (outf_syn, epoch, idx), gt[idx].cpu().numpy(), fmt='%.6f')
                    np.savetxt('%s/vis/epoch_%03d/recon_%d.xyz' % (outf_syn, epoch, idx), recon[idx].cpu().numpy(), fmt='%.6f')
                save_image(recon.transpose(1, 2).reshape(-1, 3, opt.patch_size, opt.patch_size), '%s/vis/epoch_%03d/recon.png' % (outf_syn, epoch), nrow=5)
                # visualize_pointcloud_batch('%s/epoch_%03d_recon_eval.png' % (outf_syn, epoch),
                #                            fold2_rec.transpose(1, 2), None, None,
                #                            None)
                # visualize_pointcloud_batch('%s/epoch_%03d_x_eval.png' % (outf_syn, epoch), x.transpose(1, 2), None,
                #                            None,
                #                            None)

            if (epoch + 1) % opt.saveIter == 0:

                if should_diag:

                    
                    save_dict = {
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                    }

                    # if opt.use_ema:
                    #     save_dict.update({'ema': ema.state_dict()})

                    torch.save(save_dict, '%s/checkpoint.pth' % (output_dir))
                
    dist.destroy_process_group()

def test_step(gpu, opt, logger, tb_writer, should_diag, test_dataloader, model, epoch, i, data):
    x = data['train_points'].transpose(1,2) # B x N x 3

    '''
            test fold centers
            '''

    if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
        x = x.cuda(gpu)
    elif opt.distribution_type == 'single':
        x = x.cuda()

    loss = model.get_test_loss_iter(x)

    if not opt.debug:
        global_step = i + len(test_dataloader) * epoch
        if opt.use_tb:
            tb_writer.add_scalar('test_loss_cd', loss['cd_loss'].item(), global_step)
            tb_writer.add_scalar('test_loss_emd', loss['emd_loss'].item(), global_step)

    if i % opt.print_freq == 0 and should_diag:
        logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    cd_loss: {:.4e},    emd_loss: {:.4e},    '
                            .format(
                        epoch, opt.niter, i, len(test_dataloader),loss['cd_loss'].item().detach().cpu().numpy(), loss['emd_loss'].item().detach().cpu().numpy()
                        ))
                
    return x

def main():
    opt = parse_args()

    # get output directory
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    # output_dir = os.path.join(opt.model_dir, opt.experiment_name, current_time)
    output_dir = os.path.join(opt.model_dir, opt.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    copy_source(__file__, output_dir)

    # Use random port to avoid collision between parallel jobs
    if opt.world_size == 1:
        opt.port = np.random.randint(10000, 20000)
    opt.dist_url = f'tcp://{opt.node}:{opt.port}'
    print('Using url {}'.format(opt.dist_url))

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir))
    else:
        train(opt.gpu, opt, output_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='fold_p', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--dataroot', default='/home/warrenz/Documents/mmt/data/data/data_3D/PointFlow-Data/ShapeNetCore.v2.PC15k')
    parser.add_argument('--category', default='chair')
    parser.add_argument('--num_classes', type=int, default=55)

    parser.add_argument('--bs', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')

    parser.add_argument('--nc', type=int,  default=3)
    parser.add_argument('--npoints', type=int, default=2048)
    parser.add_argument('--fps_points', type=int, default=-1, help='number of points to sample from the point cloud')
    parser.add_argument('--n_patches', type=int, default=256, help='number of patch centers to sample')
    parser.add_argument('--patch_size', type=int, default=4, help='local patch size for 2D DiT')
    
    '''model'''
    parser.add_argument("--model_type", type=str, choices=list(folding_model_configurers.keys()), default="foldingnet_s")
    parser.add_argument('--ebd_dim', type=int, default=512, help='embedding dim')
    parser.add_argument('--loss_type', type=str, default='cd_2', help='loss type', choices=['cd_1', 'cd_2', 'cd_1_2', 'sinkhorn', 'emd', 'emd_cd_1', 'emd_cd_2', 'emd_cd_1_2'])

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")


    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveIter', default=100, type=int, help='unit: epoch')
    parser.add_argument('--diagIter', default=100, type=int, help='unit: epoch')
    parser.add_argument('--vizIter', default=100, type=int, help='unit: epoch')
    parser.add_argument('--print_freq', default=50, type=int, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    parser.add_argument('--debug', action='store_true', default=False, help = 'debug mode')
    parser.add_argument('--use_tb', action='store_true', default=False, help = 'use tensorboard')
    parser.add_argument('--use_pretrained', action='store_true', default=False, help = 'use pretrained 2d DiT weights')
    parser.add_argument('--use_ema', action='store_true', default=False, help = 'use ema')

    opt = parser.parse_args()

    return opt

    

if __name__ == '__main__':
    main()