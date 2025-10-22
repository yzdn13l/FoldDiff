import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.data import Data
from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool
import numpy as np
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

from geomloss import SamplesLoss
earth_mover_distance = SamplesLoss("sinkhorn", p=2)

def sample_2D(m, n):
    # m: number of points in each row
    # n: number of rows
    x = np.linspace(-1, 1, m)
    y = np.linspace(-1, 1, n)
    xv, yv = np.meshgrid(x, y)
    return torch.from_numpy(np.vstack((xv.flatten(), yv.flatten())).T).float()

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn)

    def forward(self, x, pos, batch, num_samples=32):
        # print('x, pos, batch', x, pos, batch)
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=num_samples)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


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

class AttPool(torch.nn.Module):
    def __init__(self, in_chs):
        super(AttPool, self).__init__()
        self.in_chs = in_chs
        self.linear_transform = SMLP(in_chs, in_chs, False, 'none')
    def forward(self, x):
        bs = x.size(0)
        num_pts = x.size(1)
        assert x.ndim==3 and x.size(2)==self.in_chs
        scores = F.softmax(self.linear_transform(x), dim=1)
        y = (x * scores).sum(dim=1)
        return y

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn
        # self.attn_pool = AttPool(512)
        # self.out_nn = MLP([1024, 512, 512])

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1)) # (B*N, 512)
        B = torch.max(batch) + 1
        max_pool_x = global_max_pool(x, batch) # (B, 512)
        # attn_pool_x = self.attn_pool(x.view(B, -1, x.size(-1))) # (B, 512)
        # x = torch.cat([max_pool_x, attn_pool_x], dim=-1) # (B, 1024)
        # x = self.out_nn(x) # (B, 512)
        # x = attn_pool_x
        x = max_pool_x
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        #               ,BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Attention(torch.nn.Module):
    def __init__(self, NN_h, NN_l, NN_g, NN_f=None):
        super(Attention, self).__init__()
        self.M_h = NN_h
        self.M_l = NN_l
        self.M_g = NN_g
        self.M_f = NN_f

    def forward(self, *inputs):
        pass


class SelfAttention(Attention):

    def __init__(self, NN_h, NN_l, NN_g, NN_f=None):
        super(SelfAttention, self).__init__(NN_h, NN_l, NN_g, NN_f)

    def forward(self, p):
        h = self.M_h(p).transpose(-1, -2) # q
        l = self.M_l(p) # k
        g = self.M_g(p) # v
        mm = torch.matmul(l, h)
        attn_weights = F.softmax(mm, dim=-1)
        atten_appllied = torch.bmm(attn_weights, g)
        if self.M_f is not None:
            return self.M_f(p + atten_appllied)
        else:
            return p + atten_appllied


class SkipAttention(Attention):

    def __init__(self, NN_h, NN_l, NN_g, NN_f=None):
        super(SkipAttention, self).__init__(NN_h, NN_l, NN_g, NN_f)

    def forward(self, p, r):
        h = self.M_h(p).expand(-1, -1, r.size(2), -1).unsqueeze(-2) # (B, )
        l = self.M_l(r).expand(-1, h.size(1), -1, -1).unsqueeze(-1)
        g = self.M_g(r).squeeze(1)
        mm = torch.matmul(h, l).squeeze(-2).squeeze(-1)
        attn_weights = F.softmax(mm, dim=-1)
        atten_appllied = torch.bmm(attn_weights, g)
        if self.M_f is not None:
            return self.M_f(p.squeeze() + atten_appllied)
        else:
            return p.squeeze() + atten_appllied


class FoldingBlock(torch.nn.Module):
    def __init__(self, input_shape, output_shape, attentions, NN_up, NN_down, NN_rec):
        super(FoldingBlock, self).__init__()
        self.in_shape = input_shape
        self.out_shape = output_shape
        self.self_attn = SelfAttention(*attentions)
        self.M_up = MLP(NN_up)
        self.M_down = MLP(NN_down)
        self.M_rec = Seq(Lin(NN_rec[0], NN_rec[1]), ReLU(), Lin(NN_rec[1], NN_rec[2]))

    def Up_module(self, p, m, n):
        p = p.repeat(1, int(self.out_shape / self.in_shape), 1).contiguous()
        points = sample_2D(m, n).to(p.device) # (m*n, 2)
        p_2d = torch.cat((p, points.unsqueeze(0).expand(p.size(0), -1, -1)), -1)
        p_2d = self.M_up(p_2d)
        p = torch.cat([p, p_2d], -1)
        return self.self_attn(p)

    def Down_module(self, p):
        p = p.view(-1, self.in_shape, int(self.out_shape / self.in_shape) * p.size(2))
        return self.M_down(p)

    def forward(self, p, m, n):
        p1 = self.Up_module(p, m, n)
        p2 = self.Down_module(p1)
        p_delta = p - p2
        p2 = self.Up_module(p_delta, m, n)
        rec = self.M_rec(p2)
        return p1 + p2, rec

class PyramidNet(torch.nn.Module):

    def __init__(self, 
                 pgi_resolutions=[[64, 32], [32, 16], [16, 16]], 
                 ebd_dims=[128, 256, 512]):
        super(PyramidNet, self).__init__()
        self.m0, self.n0 = pgi_resolutions[2]
        self.m1, self.n1 = pgi_resolutions[1]
        self.m2, self.n2 = pgi_resolutions[0]
        pts_res = [self.m0*self.n0, self.m1*self.n1, self.m2*self.n2]
        ratios = [pts_res[i] / pts_res[i+1] for i in range(2)]
        ratios.reverse()

        self.sa1_module = SAModule(ratios[0], 0.2, MLP([3 + 3, 64, 64, ebd_dims[0]]))
        self.sa2_module = SAModule(ratios[1], 0.4, MLP([ebd_dims[0] + 3, ebd_dims[0], ebd_dims[0], ebd_dims[1]]))
        self.sa3_module = GlobalSAModule(MLP([ebd_dims[1] + 3, ebd_dims[1], ebd_dims[2], ebd_dims[2]]))

        self.skip_attn1 = SkipAttention(MLP([ebd_dims[2] + 2, 128]), MLP([ebd_dims[1], 128]), MLP([ebd_dims[1], ebd_dims[2] + 2]), MLP([ebd_dims[2] + 2, ebd_dims[2]]))
        self.skip_attn2 = SkipAttention(MLP([256, 64]), MLP([128, 64]), MLP([128, 256]), MLP([256, 256]))

        m_, n_ = pgi_resolutions[2]
        # m_, n_ = 8, 8
        self.m_ , self.n_ = m_, n_
        input_1, output_1 = m_*n_, pts_res[0]
        self.folding1 = FoldingBlock(input_1, output_1, [MLP([512 + 512, 256]), MLP([512 + 512, 256]), MLP([512 + 512, 512 + 512]),
                                               MLP([512 + 512, 512, 256])], [512 + 2, 512], [int(output_1 / input_1) * 256, 512], [256, 128, 3])
        input_2, output_2 = pts_res[0], pts_res[1]
        self.folding2 = FoldingBlock(input_2, output_2, [MLP([256 + 256, 64]), MLP([256 + 256, 64]), MLP([256 + 256, 256 + 256]),
                                                MLP([256 + 256, 256, 128])], [256 + 2, 256], [int(output_2 / input_2) * 128, 256], [128, 64, 3])
        input_3, output_3 = pts_res[1], pts_res[2]
        self.folding3 = FoldingBlock(input_3, output_3, [MLP([128 + 128, 64]), MLP([128 + 128, 64]), MLP([128 + 128, 128 + 128]),
                                                 MLP([128 + 128, 128])], [128 + 2, 128], [int(output_3 / input_3) * 128, 256, 128], [128, 64, 3])

        self.lin = Seq(Lin(128, 64), ReLU(), Lin(64, 3))

    def Encode(self, data):
        # N0 = 2048, N1 = 512, N2 = 256
        # (x, pos, batch), in shapes: (B*N0, 3), (B*N0, 3), (B*N0), out shapes: (B*N1, 128), (B*N1, 3), (B*N1)
        sa1_out = self.sa1_module(data.pos, data.pos, data.batch)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        return sa1_out, sa2_out, sa3_out

    def Decode(self, encoded):
        # encoded[0] (B*N1, 128), (B*N1, 3), (B*N1)
        # encoded[1] (B*N2, 256), (B*N2, 3), (B*N2)
        # encoded[2] (B, 512), zeros (B, 3), (B)
        m0, n0, m1, n1, m2, n2 = self.m0, self.n0, self.m1, self.n1, self.m2, self.n2
        
        m_, n_ = self.m_, self.n_
        p = sample_2D(m_, n_).to(self.device) # (m_*n_, 2)
        out = encoded[2][0].contiguous() # (B, 512)
        out = out.view(out.size(0), 1, 1, out.size(-1)).repeat(1, m_*n_, 1, 1) # (B, m_*n_, 1, 512)
        out = torch.cat((out, p.view(1, p.size(0), 1, p.size(-1)).repeat(out.size(0), 1, 1, 1)), -1) # (B, m_*n_, 1, 514)
        # out (B, m_*n_, 1, 514), encoded[1][0] (B, 1, 256, 256) 
        out = self.skip_attn1(out, encoded[1][0].view(out.size(0), 1, m0*n0, encoded[1][0].size(-1)))
        # out (B, 256, 512)
        out, rec1 = self.folding1(out, m0, n0)
        # out (B, 256, 256), rec1 (B, 64*4, 3)
        out = out.unsqueeze(-2)
        # out (B, 64*4, 1, 256), encoded[0][0] (B, 1, 512, 128)
        out = self.skip_attn2(out, encoded[0][0].view(out.size(0), 1, m1*n1, encoded[0][0].size(-1)))
        # out (B, 64*4, 256)
        out, rec2 = self.folding2(out, m1, n1)
        rec1_up = F.interpolate(rec1.view(rec1.size(0), m0, n0, 3).permute(0, 3, 1, 2), size=(m1, n1), mode='nearest')
        rec2 = rec2 + rec1_up.permute(0, 2, 3, 1).contiguous().view(rec2.size(0), -1, 3)
        # out (B, 64*4*2, 256), rec2 (B, 64*4*2, 3)
        _, rec3 = self.folding3(out, m2, n2)
        # rec3 (B, 64*4*2*4, 3)
        rec2_up = F.interpolate(rec2.view(rec2.size(0), m1, n1, 3).permute(0, 3, 1, 2), size=(m2, n2), mode='nearest')
        rec3 = rec3 + rec2_up.permute(0, 2, 3, 1).contiguous().view(rec3.size(0), -1, 3)
        return rec1, rec2, rec3

    def pcl2data(self, pcl):
        assert len(pcl.size()) == 3
        assert pcl.size(2) == 3
        B, N = pcl.size(0), pcl.size(1)
        data = Data(pos=pcl.view(B*N, 3), batch=torch.arange(B).repeat_interleave(N).to(pcl.device))
        return data
    
    def forward(self, pcl):
        data = self.pcl2data(pcl)
        self.device = data.pos.device
        encoded = self.Encode(data)

        decoded = self.Decode(encoded)

        return encoded, decoded

    def get_loss(self, pcl, loss_type='cd'):
        encoded, decoded = self.forward(pcl)
        rec1, rec2, rec3 = decoded
        B = rec1.size(0)
        pts1, pts2, pts3 = encoded[1][1], encoded[0][1], pcl
        pts1, pts2, pts3 = pts1.view(B, -1, 3), pts2.view(B, -1, 3), pts3.view(B, -1, 3)
        if loss_type == 'cd':
            loss = chamfer_distance(rec1, pts1)[0].mean() + chamfer_distance(rec2, pts2)[0].mean() + chamfer_distance(rec3, pts3)[0].mean()
        elif loss_type == 'emd':
            loss = earth_mover_distance(rec1, pts1).mean() + earth_mover_distance(rec2, pts2).mean() + earth_mover_distance(rec3, pts3).mean()
        elif loss_type == 'cd+emd':
            loss = chamfer_distance(rec1, pts1)[0].mean() + chamfer_distance(rec2, pts2)[0].mean() + chamfer_distance(rec3, pts3)[0].mean()
            loss += 0.1*(earth_mover_distance(rec1, pts1).mean() + earth_mover_distance(rec2, pts2).mean() + earth_mover_distance(rec3, pts3).mean())
        elif loss_type == 'cd+emd+local':
            loss = chamfer_distance(rec1, pts1)[0].mean() + chamfer_distance(rec2, pts2)[0].mean() + chamfer_distance(rec3, pts3)[0].mean()
            loss += 0.1*(earth_mover_distance(rec1, pts1).mean() + 
                         earth_mover_distance(rec2, pts2).mean() + 
                         earth_mover_distance(rec3, pts3).mean())
            loss += 0.001*(local_neighbor_regularization(rec1.view(B, self.m0, self.n0, 3)) + 
                           local_neighbor_regularization(rec2.view(B, self.m1, self.n1, 3)) + 
                           local_neighbor_regularization(rec3.view(B, self.m2, self.n2, 3)))
        return loss
    
    def get_rec(self, pcl):
        with torch.no_grad():
            _, decoded = self.forward(pcl)
        rec1, rec2, rec3 = decoded
        return rec3
    
    def get_pgi(self, pcl):
        # pcl: [B, N, 3]
        B = pcl.size(0)
        
        _, decoded = self.forward(pcl)
        rec1, rec2, rec3 = decoded
        m0, n0, m1, n1, m2, n2 = self.m0, self.n0, self.m1, self.n1, self.m2, self.n2
        
        pgi1 = rec1.view(B, m0, n0, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, m0, n0]
        pgi2 = rec2.view(B, m1, n1, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, m1, n1]
        pgi3 = rec3.view(B, m2, n2, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, m2, n2]
        
        return pgi1, pgi2, pgi3
    
    def pcl2pgi(self, pcl, use_pgi_nn=False):
        # pcl [B, N, 3]
        with torch.no_grad():
            pgi = self.get_pgi(pcl)[-1].permute(0, 2, 3, 1) # [B, m2, n2, 3]
        if use_pgi_nn:
            m2, n2 = pgi.size(1), pgi.size(2)
            pgi_nn = knn_points(pgi.flatten(1, 2), pcl, K=1, return_nn=True)[-1].squeeze(1) # [B, m2*n2, 3]
            pgi_nn = pgi_nn.view(pcl.size(0), m2, n2, 3).contiguous() # [B, m2, n2, 3]
            return pgi_nn
        else:
            return pgi

def local_neighbor_regularization(pgi, K=9):
    # pgi: [B, m, n, 3]
    k = int(K**0.5)
    assert k == K ** 0.5, 'K should be a perfect square'
    assert k % 2 == 1, 'K should be an odd number'
    pgi_pad = F.pad(pgi.permute(0, 3, 1, 2), (k//2, k//2, k//2, k//2), mode='circular').permute(0, 2, 3, 1) # [B, m+k, n+k, 3]
    # gather pgi neighbors of size (B, 64, 32, K, 3)
    B, m, n, _ = pgi.size()
    pgi_neighbors = pgi_pad.unfold(1, k, 1).unfold(2, k, 1).permute(0,1,2,4,5,3).reshape(-1, K, 3).to(pgi.device) # [B*m*n, K, 3]
    
    # Compute distances without explicit loops
    center_points = pgi.reshape(-1, 1, 3)
    distances = torch.norm(pgi_neighbors - center_points, dim=-1)

    # Compute mean distance
    total_distance = distances.mean()

    return total_distance