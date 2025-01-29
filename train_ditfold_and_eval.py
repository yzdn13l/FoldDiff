import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics, EMD_CD
from tqdm import tqdm
import argparse
from torch.distributions import Normal
import pandas as pd

from utils.file_utils import *
from utils.visualize import *
import torch.distributed as dist
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from torchvision.utils import save_image
from copy import deepcopy
from collections import OrderedDict

from models.dit_fold import DiT_models
from pgi_models import folding_model_configurers
from utils.misc import Evaluator

from tensorboardX import SummaryWriter
from pytorch3d.ops import sample_farthest_points, knn_points


'''
some utils
'''
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        if name.startswith('model.module'):
            name = name.replace('model.module.', 'model.')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:,[1,2,0]].dot(M).dot(N).dot(K), faces[:,[1,2,0]]
    return v, f

def norm(v, f):
    v = (v - v.min())/(v.max() - v.min()) - 0.5

    return v, f

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)
        
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type # 'eps'
        self.model_var_type = model_var_type # 'fixedsmall' or 'fixedlarge'
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, y, clip_denoised: bool, return_pred_xstart: bool, denoise_fn_kwargs=None):
        model_output = denoise_fn(data, t, y) if denoise_fn_kwargs is None else denoise_fn(data, t, y, **denoise_fn_kwargs)

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, y, clip_denoised=False, return_pred_xstart=False, denoise_fn_kwargs=None):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, y=y, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True, denoise_fn_kwargs=denoise_fn_kwargs)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device, y,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False, denoise_fn_kwargs=None):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=clip_denoised, return_pred_xstart=False, denoise_fn_kwargs=denoise_fn_kwargs)

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, y, freq,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False, denoise_fn_kwargs=None):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn, y=y,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False, denoise_fn_kwargs=denoise_fn_kwargs)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, y, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=data_start, x_t=data_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, y=y, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None, y=None):
        """
        Training loss calculation
        """
        B = data_start.shape[0]
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start.shape, dtype=data_start.dtype, device=data_start.device)
        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(data_t, t, y)
            assert data_t.shape == data_start.shape
            assert eps_recon.shape == data_start.shape
            losses = ((noise - eps_recon)**2).mean(dim=list(range(1, len(data_start.shape))))
        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, y=y, clip_denoised=False,
                return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, y, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            # x_start: [B, D, M, N] clean data
            # Initialize tensors to store the VLB terms and MSE for progressive prediction loss
            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=self.q_sample(x_start=x_start, t=t_b), t=t_b,
                    y=y, clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart-x_start)**2).mean(dim=list(range(1, len(x_start.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()

class FoldingModel(nn.Module):
    def __init__(self, args):
        super(FoldingModel, self).__init__()
        if args.fold_p is None:
            args.fold_p = f'foldingnet_p_{args.n_p}'
        self.fold_p = folding_model_configurers[args.fold_p]()
        
        fold_p_state_dict = torch.load(args.fold_p_path)['model_state']
        for k in list(fold_p_state_dict.keys()):
            fold_p_state_dict[k.replace('model.module.', '')] = fold_p_state_dict.pop(k)
        self.fold_p.load_state_dict(fold_p_state_dict)
        self.fold_p.eval()
        self.N_C = args.n_c
        self.N_P = args.n_p
    
    def forward(self, data):
        with torch.no_grad():
            return self._forward(data)
            
    def _forward(self, data):
        B, D, N = data.shape
        data = data.transpose(1,2)
        N_C = self.N_C
        N_P = self.N_P
        data_centers = sample_farthest_points(data, K=N_C)[0] # [B, N_C, 3]
        # randomly permute the order of the centers
        data_centers = data_centers[:, torch.randperm(N_C), :]
        # _, folded_centers ,_ = self.fold_c(data_centers) # [B, N_C, 3]
        folded_centers = data_centers
        local_patches = knn_points(folded_centers, data, K=N_P, return_nn=True)[-1] # [B, N_C, N_P, 3]
        # normalize local patches
        center = local_patches.mean(dim=-2, keepdim=True) # [B, N_C, 1, 3]
        local_patches = local_patches - center
        scale = local_patches.norm(dim=-1, keepdim=True).max() # [B, N_C, N_P, 1]
        local_patches = local_patches / scale
        folded_patches, _ = self.fold_p(local_patches.view(-1, N_P, 3))
        folded_patches = folded_patches.view(B, N_C, N_P, 3) * scale + center
        return folded_patches
        

class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)
        self.fold_model = FoldingModel(args)
        self.model = DiT_models[args.model_type](input_size=args.n_c,
                                                 num_classes=args.num_classes)

    
    def parameters(self, recurse = True):
        return self.model.parameters(recurse)
    
    def load_state_dict(self, state_dict, strict = True):
        # only load the state dict of the model
        self.model.load_state_dict(state_dict, strict)
    
    def prior_kl(self, data):
        # data (B, D, M, N)
        return self.diffusion._prior_bpd(data)

    def all_kl(self, data, y, clip_denoised=True):
        # data (B, D, M, N)
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, data, y, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t, y):
        B, N_C, N_P, D = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64
        out = self.model(data, t, y)
        assert out.shape == torch.Size([B, N_C, N_P, D])
        return out
    
    def _denoise_with_cfg(self, data, t, y, cfg=1.5):
        B, N_C, N_P, D = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64
        out = self.model.forward_with_cfg(data, t, y, cfg)
        assert out.shape == torch.Size([B, N_C, N_P, D])
        return out

    def get_loss_iter(self, data, noises=None, y=None):
        # data [B, N_C, N_P, 3]
        
        B, N_C, N_P, D = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, y=y)
        assert losses.shape == t.shape == torch.Size([B])
        return losses, torch.mean(t.float())

    def gen_samples(self, shape, device, y, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False,
                    cfg=1.0):
        assert cfg >= 1.0
        if cfg != 1.0:
            denoise_fn_kwargs = {'cfg': cfg}
            return self.diffusion.p_sample_loop(self._denoise_with_cfg, shape=shape, device=device, y=y, noise_fn=noise_fn, 
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running, denoise_fn_kwargs=denoise_fn_kwargs)
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, y, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False, cfg=1.0):
        assert cfg >= 1.0
        if cfg != 1.0:
            denoise_fn_kwargs = {'cfg': cfg}
            return self.diffusion.p_sample_loop_trajectory(self._denoise_with_cfg, shape=shape, device=device, y=y, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running, denoise_fn_kwargs=denoise_fn_kwargs)
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas


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
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def generate_eval(model, opt, gpu, outf_syn, evaluator):

    _, test_dataset = get_dataset(opt.dataroot, opt.eval_npoints, opt.category)
    if opt.eval_fps_points > 0:
        test_dataset.cache_fps_points(opt.eval_fps_points)
    _, test_dataloader, _, test_sampler = get_dataloader(opt, test_dataset, test_dataset)


    def new_y_chain(device, num_chain, num_classes):
        return torch.randint(low=0,high=num_classes,size=(num_chain,),device=device)
    
    with torch.no_grad():

        samples = []
        reference = []

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):

            x = data['train_points'] # [B, N, 3]
            m, s = data['mean'].float(), data['std'].float() # using the global mean and std of the training set
            y = data['cate_idx']
            
            gen = model.gen_samples((x.shape[0], opt.n_c, opt.n_p, 3), 
                                    gpu, new_y_chain(gpu,y.shape[0],opt.num_classes), clip_denoised=False,
                                    cfg=opt.cfg_scale).detach().cpu()
            if opt.eval_fps_points > 0:
                # farthest point sampling
                gen = sample_farthest_points(gen.flatten(1,2).contiguous(), K=opt.eval_fps_points)[0]
            else:
                # random sampling
                print('random sampling')
                gen = gen.flatten(1,2)
                gen = gen[:, torch.randperm(gen.shape[1])[:opt.eval_npoints], :]
            # original normalization
            gen = gen * s + m
            x = x * s + m
            
            samples.append(gen.to(gpu).contiguous())
            reference.append(x.to(gpu).contiguous())

            ep_save_dir = os.path.join(outf_syn, 'batch_%03d' % i)
            os.makedirs(ep_save_dir, exist_ok=True)
            for idx in range(gen.shape[0]):
                pts = gen[idx].cpu().numpy() # [N, 3]
                np.savetxt(os.path.join(ep_save_dir, 'sample_%d.xyz' % idx), pts)
                ref_pts = x[idx].cpu().numpy()
                np.savetxt(os.path.join(ep_save_dir, 'reference_%d.xyz' % idx), ref_pts)
                concat_pts = np.concatenate([x[idx].cpu().numpy(), pts], axis=0)
                np.savetxt(os.path.join(ep_save_dir, 'concat_%d.xyz' % idx), concat_pts)

            
            # # Compute metrics
            # results = compute_all_metrics(gen, x, opt.bs)
            # results = {k: (v.cpu().detach().item()
            #             if not isinstance(v, float) else v) for k, v in results.items()}

            # jsd = JSD(gen.numpy(), x.numpy())

            # evaluator.update(results, jsd)

    samples = torch.cat(samples, dim=0)    
    reference = torch.cat(reference, dim=0)

    if opt.distribution_type:
        samples_gather = concat_all_gather(samples)
        reference_gather = concat_all_gather(reference)
        samples = samples_gather
        reference = reference_gather
    torch.save(samples, opt.eval_path)
    
    logger.info('Computing metrics...')
    logger.info('Samples shape: %s' % str(samples.shape))
    logger.info('Reference shape: %s' % str(reference.shape))
    
    results = compute_all_metrics(samples, reference, opt.bs)
    results = {k: (v.cpu().detach().item()
                if not isinstance(v, float) else v) for k, v in results.items()}

    jsd = JSD(samples.cpu().numpy(), reference.cpu().numpy())

    evaluator.update(results, jsd)
    stats = evaluator.finalize_stats() 
    '''
    stats.keys() = ['1-NNA-CD', '1-NNA-EMD', 'COV-CD', 'COV-EMD', 'MMD-CD', 'MMD-EMD', 'JSD']
    '''
    stats['model_name'] = opt.model_type
    stats['n_params'] = sum(param.numel() for param in model.parameters())/1e6
    return stats

def test(gpu, opt, output_dir):

    logger = setup_logging(output_dir)


    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True

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

    '''
    create networks
    '''

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

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

        logger.info("Model = %s" % str(model))
        total_params = sum(param.numel() for param in model.parameters())/1e6
        logger.info("Total_params = %s MB " % str(total_params))    # S4: 32.81 MB

    model.eval()

    evaluator = Evaluator(results_dir=output_dir)    

    with torch.no_grad():
        
        if should_diag:
            logger.info("Resume Path:%s" % opt.model)

        if opt.use_ema:
            
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            checkpoint = torch.load(opt.model, map_location=map_location)
            ema_weights = checkpoint['ema']
            checkpoint_dict = {k.replace('model.', 'module.'): ema_weights[k] for k in ema_weights if k.startswith('model.')}
            model.load_state_dict(checkpoint_dict)
        else:
            ckpt, state_dict = extract_model_state_dict(opt)
            model.load_state_dict(state_dict)
        
        for run_i in range(opt.nrepeats):

            outf_syn = os.path.join(output_dir, f'run_{run_i}', 'syn')
            os.makedirs(outf_syn, exist_ok=True)

            opt.eval_path = os.path.join(outf_syn, 'samples.pth')
            Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
            
            stats = generate_eval(model, opt, gpu, outf_syn, evaluator)

            if should_diag:
                logger.info(stats)
            
            # update results_512-9.csv
            # result_path = 'results_512-9.csv'
            result_path = os.path.join(output_dir, 'results.csv')
            if os.path.exists(result_path):
                df = pd.read_csv(result_path)
                df_stats = pd.DataFrame(stats, index=[0], columns=df.keys())
                df = pd.concat([df, df_stats], ignore_index=True)
                df.to_csv(result_path, index=False)
            else:
                # model_name,n_params,n_layers,n_hidden,patch_size,n_heads,Epoch,1-NNA-CD,1-NNA-EMD,COV-CD,COV-EMD,MMD-CD,MMD-EMD,JSD
                df_stats = pd.DataFrame(stats, index=[0], columns=['model_name','n_params','1-NNA-CD','1-NNA-EMD','COV-CD','COV-EMD','MMD-CD','MMD-EMD','JSD'])
                df_stats.to_csv(result_path, index=False)
            

def train(gpu, opt, output_dir, noises_init):

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

    ''' data '''
    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category)
    if opt.fps_points > 0:
        train_dataset.cache_fps_points(opt.fps_points)
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, None)

    '''
    create networks
    '''

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)
    
    # Note that parameter initialization is done within the DiT constructor
    if opt.use_ema:
        ema = deepcopy(model).to(gpu)  # Create an EMA of the model for use after training
        requires_grad(ema, False)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

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

    start_epoch = 0
    if opt.model != '':
        
        if opt.use_ema:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            checkpoint = torch.load(opt.model, map_location=map_location)
            ema_weights = checkpoint['ema']
            checkpoint_dict = {k.replace('model.', 'module.'): ema_weights[k] for k in ema_weights if k.startswith('model.')}
            model.load_state_dict(checkpoint_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            ckpt, state_dict = extract_model_state_dict(opt)
            
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt['epoch'] + 1

    else:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    def new_y_chain(y, num_chain, num_classes):
        return torch.randint(low=0,high=num_classes,size=(num_chain,),device=y.device)

    # Prepare models for training:
    if opt.use_ema:
        model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode

    for epoch in range(start_epoch, opt.niter):

        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)

        for i, data in enumerate(dataloader):

            x = data['train_points'].transpose(1,2) # [B, D, N]
            noises_batch = noises_init[data['idx']]
            y = data['cate_idx']

            '''
            train diffusion
            '''

            if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                x = x.cuda(gpu)
                noises_batch = noises_batch.cuda(gpu)
                y = y.cuda(gpu)
            elif opt.distribution_type == 'single':
                x = x.cuda()
                noises_batch = noises_batch.cuda()
                y = y.cuda()

            x = model.fold_model(x)
            
            loss, t_avg = model.get_loss_iter(x, noises_batch, y)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            # netpNorm, netgradNorm = getGradNorm(model)
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()

            if opt.use_ema:
                update_ema(ema, model)

            if not opt.debug:
                global_step = i + len(dataloader) * epoch
                if opt.use_tb:
                    tb_writer.add_scalar('train_loss', loss.item(), global_step)
                    tb_writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], global_step)

            if i % opt.print_freq == 0 and should_diag:

                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},   t_avg: {:.2f}'
                             .format(
                        epoch, opt.niter, i, len(dataloader), loss.item(), t_avg.item()
                        ))

        if (epoch + 1) % opt.vizIter == 0 and should_diag:
            logger.info('Generation: eval')

            model.eval()
            with torch.no_grad():

                x_gen_eval = model.gen_samples(new_x_chain(x, 25).shape, x.device, new_y_chain(y,25,opt.num_classes), clip_denoised=False).flatten(1, 2)
                x_gen_list = model.gen_sample_traj(new_x_chain(x, 1).shape, x.device, new_y_chain(y,1,opt.num_classes), freq=40, clip_denoised=False)
                x_gen_all = torch.cat(x_gen_list, dim=0).flatten(1, 2) # [num_timesteps*B, N_C*N_P, 3]
                N_S = x_gen_all.shape[0] // x_gen_eval.shape[0]
                gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]
                gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

                logger.info('      [{:>3d}/{:>3d}]  '
                             'eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                             'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
                    .format(
                    epoch, opt.niter,
                    *gen_eval_range, *gen_stats,
                ))
            ep_save_dir = os.path.join(outf_syn, 'epoch_%03d' % epoch)
            os.makedirs(ep_save_dir, exist_ok=True)            

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
                                       x_gen_eval, None, None,
                                       None)

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all.png' % (outf_syn, epoch),
                                       x_gen_all, None,
                                       None,
                                       None)

            visualize_pointcloud_batch('%s/epoch_%03d_x.png' % (outf_syn, epoch), x.transpose(1, 2), None,
                                       None,
                                       None)
            for idx in range(x_gen_eval.shape[0]):
                pts = x_gen_eval[idx].cpu().numpy() # [N, 3]
                np.savetxt(os.path.join(ep_save_dir, 'sample_%d.xyz' % idx), pts)

            logger.info('Generation: train')
            model.train()

        if (epoch + 1) % opt.saveIter == 0:

            if should_diag:

                
                save_dict = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }

                if opt.use_ema:
                    save_dict.update({'ema': ema.state_dict()})
                

                torch.save(save_dict, '%s/checkpoint.pth' % (output_dir))


            if opt.distribution_type == 'multi':
                dist.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
                if opt.use_ema:
                    checkpoint = torch.load('%s/checkpoint.pth' % (output_dir), map_location=map_location)['ema']
                    checkpoint_dict = {k.replace('model.', 'module.'): checkpoint[k] for k in checkpoint if k.startswith('model.')}
                    model.load_state_dict(checkpoint_dict)
                else:
                    model.load_state_dict(
                    torch.load('%s/checkpoint.pth' % (output_dir), map_location=map_location)['model_state'])
                

    dist.destroy_process_group()

def extract_model_state_dict(opt):
    ckpt = torch.load(opt.model)
    resumed_param = ckpt['model_state']
    state_dict = resumed_param.copy()
        # print the keys and weight shapes
    for k, v in resumed_param.items():
        if 'fold_model' in k:
            state_dict.pop(k)
        else:
            state_dict[k.replace('model.', '')] = state_dict.pop(k)
    return ckpt,state_dict

def main():
    opt = parse_args()

    # get output directory
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    output_dir = os.path.join(opt.model_dir, opt.experiment_name, current_time)
    os.makedirs(output_dir, exist_ok=True)
    copy_source(__file__, output_dir)

    ''' workaround '''
    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category)
    noises_init = torch.randn(len(train_dataset), opt.n_c, opt.n_p, 3)
    # Use random port to avoid collision between parallel jobs
    if opt.world_size == 1:
        opt.port = np.random.randint(10000, 20000)
    opt.dist_url = f'tcp://{opt.node}:{opt.port}'
    print('Using url {}'.format(opt.dist_url))

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        if not opt.evaluate:
            mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
            opt.model = os.path.join(output_dir, 'checkpoint.pth')
        mp.spawn(test, nprocs=opt.ngpus_per_node, args=(opt, output_dir))
    else:
        if not opt.evaluate:
            train(opt.gpu, opt, output_dir, noises_init)
            opt.model = os.path.join(output_dir, 'checkpoint.pth')
        test(opt.gpu, opt, output_dir)



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='dit2d', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--dataroot', default='../data/ShapeNetCore.v2.PC15k')
    parser.add_argument('--category', default='chair')
    parser.add_argument('--num_classes', type=int, default=1)

    parser.add_argument('--bs', type=int, default=16, help='input batch size')
    parser.add_argument('--eval_bs', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')

    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--npoints', type=int, default=8192)
    parser.add_argument('--fps_points', type=int, default=2048)
    parser.add_argument('--eval_npoints', type=int, default=8192)
    parser.add_argument('--eval_fps_points', type=int, default=2048)
    parser.add_argument('--n_c', type=int, default=256)
    parser.add_argument('--n_p', type=int, default=16)
    parser.add_argument("--fold_p", type=str, default="foldingnet_p_v2_16", help='if None, use default foldingnet_p_{n_p}')
    parser.add_argument("--fold_p_path", type=str, required=True, help='path to the pretrained foldingnet_p model')
    
    '''model'''
    parser.add_argument("--model_type", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', type=int, default=1000)

    #params
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate model')
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='scale the model cfg')


    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default=None, choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveIter', default=500, type=int, help='unit: epoch')
    parser.add_argument('--diagIter', default=500, type=int, help='unit: epoch')
    parser.add_argument('--vizIter', default=500, type=int, help='unit: epoch')
    parser.add_argument('--print_freq', default=50, type=int, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')
    parser.add_argument('--nrepeats', default=3, type=int, help='number of repeats for evaluation')

    parser.add_argument('--debug', action='store_true', default=False, help = 'debug mode')
    parser.add_argument('--use_tb', action='store_true', default=False, help = 'use tensorboard')
    parser.add_argument('--use_pretrained', action='store_true', default=False, help = 'use pretrained 2d DiT weights')
    parser.add_argument('--use_ema', action='store_true', default=False, help = 'use ema')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()
