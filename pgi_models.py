import warnings
warnings.filterwarnings('ignore')

import numpy as np
import itertools

import torch
import torch.nn as nn

# adopted from https://github.com/keeganhk/Flattening-Net/blob/master/cdbs/utils.py

def build_lattice(H, W):
    N = H * W # number of grids
    # generate grid points within the range of (0, 1)
    margin = 1e-4
    h_p = np.linspace(0+margin, 1-margin, H, dtype=np.float32)
    w_p = np.linspace(0+margin, 1-margin, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p))) # (N, 2)
    # generate grid indices
    h_i = np.linspace(0, H-1, H, dtype=np.int64)
    w_i = np.linspace(0, W-1, W, dtype=np.int64)
    grid_indices = np.array(list(itertools.product(h_i, w_i))) # (N, 2)
    
    return grid_points, grid_indices

class SMLP(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, slope=None):
        super(SMLP, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [B, N, ic]
        # y: [B, N, oc]
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1) # [B, ic, N, 1]
        y = self.conv(x) # [B, oc, N, 1]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)   
        y = y.squeeze(-1).permute(0, 2, 1).contiguous() # [B, N, oc]
        return y


class ResSMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResSMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.smlp_1 = SMLP(in_channels, in_channels, True, 'none')
        self.smlp_2 = SMLP(in_channels, out_channels, True, 'none')
        if in_channels != out_channels:
            self.shortcut = SMLP(in_channels, out_channels, True, 'none')
        self.nl = nn.ReLU(inplace=True)
    def forward(self, in_ftr):
        # in_ftr: [B, N, in_channels]
        out_ftr = self.smlp_2(self.nl(self.smlp_1(in_ftr)))
        if self.in_channels != self.out_channels:
            out_ftr = self.nl(self.shortcut(in_ftr) + out_ftr)
        else:
            out_ftr = self.nl(in_ftr + out_ftr)
        return out_ftr # [B, N, out_channels]
    
class CdwExtractorS(nn.Module):
    def __init__(self, ebd_dim=256):
        super(CdwExtractorS, self).__init__()
        self.res_smlp = nn.Sequential(ResSMLP(3, 64), ResSMLP(64, 128), ResSMLP(128, ebd_dim))
    def forward(self, pts):
        B, N, _ = pts.size()
        ftr_1 = self.res_smlp(pts)
        cdw = ftr_1.max(dim=1)[0]
        return cdw

class FoldingDecoder(nn.Module):
    def __init__(self, ebd_dim=1024):
        super(FoldingDecoder, self).__init__()
        fold_1_1 = SMLP(ebd_dim+2, 256, True, 'relu')
        fold_1_2 = SMLP(256, 128, True, 'relu')
        fold_1_3 = SMLP(128, 64, True, 'relu')
        fold_1_4 = SMLP(64, 3, False, 'none')
        self.fold_1 = nn.Sequential(fold_1_1, fold_1_2, fold_1_3, fold_1_4)
        fold_2_1 = SMLP(ebd_dim+3, 256, True, 'relu')
        fold_2_2 = SMLP(256, 128, True, 'relu')
        fold_2_3 = SMLP(128, 64, True, 'relu')
        fold_2_4 = SMLP(64, 3, False, 'none')
        self.fold_2 = nn.Sequential(fold_2_1, fold_2_2, fold_2_3, fold_2_4)
        
    def forward(self, cdw_dup, grids):
        concat_1 = torch.cat((cdw_dup, grids), dim=-1)
        rec_1 = self.fold_1(concat_1)
        concat_2 = torch.cat((cdw_dup, rec_1), dim=-1)
        rec_2 = self.fold_2(concat_2)
        return rec_2
    
class FoldingNetS(nn.Module):
    def __init__(self, num_grids, ebd_dim=512):
        super(FoldingNetS, self).__init__()
        self.num_grids = num_grids
        self.grid_size = int(np.sqrt(num_grids))
        assert self.grid_size**2 == self.num_grids
        self.lattice = torch.tensor(build_lattice(self.grid_size, self.grid_size)[0])
        
        self.backbone = CdwExtractorS(ebd_dim)
        self.fold1 = FoldingDecoder(ebd_dim)
        
    def forward(self, pts):
        B, N, device = pts.size(0), pts.size(1), pts.device
        grids_0 = (self.lattice).unsqueeze(0).repeat(B, 1, 1).to(device)
        cdw = self.backbone(pts)
        cdw_dup = cdw.unsqueeze(1).repeat(1, self.num_grids, 1)
        fold_rec = self.fold1(cdw_dup, grids_0)
        return fold_rec, grids_0

class CdwExtractorS_V2(nn.Module):
    def __init__(self, ebd_dim=256):
        super(CdwExtractorS_V2, self).__init__()
        self.res_smlp = nn.Sequential(ResSMLP(3, 64), ResSMLP(64, 128), ResSMLP(128, ebd_dim))
    def forward(self, pts):
        B, N, _ = pts.size()
        ftr_1 = self.res_smlp(pts)
        cdw = ftr_1.max(dim=1)[0]
        return cdw, ftr_1

class CrossAttention(nn.Module):
    def __init__(self, ebd_dim):
        super(CrossAttention, self).__init__()
        self.q_proj = SMLP(ebd_dim, ebd_dim, False, 'none')
        self.k_proj = SMLP(ebd_dim, ebd_dim, False, 'none')
        self.v_proj = SMLP(ebd_dim, ebd_dim, False, 'none')
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k):
        """
        q: B X N X ebd_dim
        k: B X M X ebd_dim
        """
        B, N, M, ebd_dim = q.size(0), q.size(1), k.size(1), q.size(2)
        q_proj = self.q_proj(q) # B X N X ebd_dim
        k_proj = self.k_proj(k) # B X M X ebd_dim
        v_proj = self.v_proj(k) # B X M X ebd_dim
        attn = self.softmax(torch.bmm(q_proj, k_proj.permute(0, 2, 1)) / np.sqrt(ebd_dim)) # B X N X M
        out = torch.bmm(attn, v_proj) # B X N X ebd_dim
        return out

class FoldingDecoderS_V2(nn.Module):
    def __init__(self, ebd_dim=256):
        super(FoldingDecoderS_V2, self).__init__()
        self.corr_input = nn.Sequential(SMLP(ebd_dim+2, ebd_dim*2, True, 'relu'), SMLP(ebd_dim*2, ebd_dim, True, 'relu'))
        self.cross_attn = CrossAttention(ebd_dim)
        
        self.fold_1 = nn.Sequential(SMLP(ebd_dim, 128, True, 'relu'), 
                                    SMLP(128, 64, True, 'none'))
        self.fold_2 = nn.Sequential(SMLP(ebd_dim+64, 128, True, 'relu'), 
                                    SMLP(128, 64, True, 'relu'),
                                    SMLP(64, 3, False, 'none'))
        
    def forward(self, cdw_dup, ftr_1, grids):
        """
        cdw_dup: B X grid_size^2 X ebd_dim
        ftr_1: B X grid_size^2 X ebd_dim
        grids: B X grid_size^2 X 2
        """
        concat_1 = torch.cat((cdw_dup, grids), dim=-1) # B X grid_size^2 X (ebd_dim+2)
        corr_feat = self.corr_input(concat_1) # B X grid_size^2 X ebd_dim
        
        # corr_feat (B, grid_size^2, ebd_dim) ftr_1 (B, grid_size^2, ebd_dim)
        cattn_feat = self.cross_attn(corr_feat, ftr_1) # (B, grid_size^2, ebd_dim)
        rec_1 = self.fold_1(cattn_feat) # (B, grid_size^2, 64)
        concat_2 = torch.cat((cdw_dup, rec_1), dim=-1)
        rec_2 = self.fold_2(concat_2)
        return rec_2

class FoldingNetS_V2(nn.Module):
    """
    Small version of FoldingNet
    Encoder: 3 -> 64 -> 128 -> ebd_dim
    Decoder: 
        fold_1: ebd_dim+2 -> 256 -> 128 -> 64
        fold_2: ebd_dim+64 -> 256 -> 128 -> 64 -> 3
    """
    def __init__(self, num_grids, ebd_dim=256):
        super(FoldingNetS_V2, self).__init__()
        self.num_grids = num_grids
        self.grid_size = int(np.sqrt(num_grids))
        assert self.grid_size**2 == self.num_grids
        self.lattice = torch.tensor(build_lattice(self.grid_size, self.grid_size)[0])
        
        self.enc = CdwExtractorS_V2(ebd_dim)
        self.dec = FoldingDecoderS_V2(ebd_dim)
        
    def forward(self, pts):
        B, N, device = pts.size(0), pts.size(1), pts.device
        grids_0 = (self.lattice).unsqueeze(0).repeat(B, 1, 1).to(device)
        cdw, ftr_1 = self.enc(pts)
        cdw_dup = cdw.unsqueeze(1).repeat(1, self.num_grids, 1)
        fold_rec = self.dec(cdw_dup, ftr_1, grids_0)
        return fold_rec, grids_0
    
def foldingnet_p_64():
    return FoldingNetS(64, 512)

def foldingnet_p_16():
    return FoldingNetS(16, 128)

def foldingnet_p_v2_64():
    return FoldingNetS_V2(64, 256)

def foldingnet_p_v2_16():
    return FoldingNetS_V2(16, 128)

def foldingnet_p_v2_9():
    return FoldingNetS_V2(9, 128)

def foldingnet_p_v2_4():
    return FoldingNetS_V2(4, 128)

folding_model_configurers = {
    'foldingnet_p_64': foldingnet_p_64,
    'foldingnet_p_16': foldingnet_p_16,
    'foldingnet_p_v2_64': foldingnet_p_v2_64,
    'foldingnet_p_v2_16': foldingnet_p_v2_16,
    'foldingnet_p_v2_9': foldingnet_p_v2_9,
    'foldingnet_p_v2_4': foldingnet_p_v2_4,
}

