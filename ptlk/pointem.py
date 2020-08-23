""" PointLK ver. 2018.07.06.
    using approximated Jacobian by backward-difference.
"""

import numpy as np
import torch

from . import pointnet
from . import se3, so3, invmat

eps = 1e-6
class PointEM(torch.nn.Module):
    def __init__(self, ptnet, delta=1.0e-2, learn_delta=False):
        super().__init__()
        self.ptnet = ptnet
        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp # [B, 6] -> [B, 4, 4]
        self.transform = se3.transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        w1 = delta
        w2 = delta
        w3 = delta
        v1 = delta
        v2 = delta
        v3 = delta
        twist = torch.Tensor([w1, w2, w3, v1, v2, v3])
        self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

        # results
        self.last_err = None
        self.g_series = None # for debug purpose
        self.prev_r = None
        self.g = None # estimation result
        self.itr = 0

    @staticmethod
    def rsq(r):
        # |r| should be 0
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, size_average=False)

    @staticmethod
    def comp(g, igt):
        """ |g*igt - I| (should be 0) """
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        return torch.nn.functional.mse_loss(A, I, size_average=True) * 16

    @staticmethod
    def do_forward(net, p0, p1, maxiter=10, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        a0 = torch.eye(4).view(1, 4, 4).expand(p0.size(0), 4, 4).to(p0) # [B, 4, 4]
        a1 = torch.eye(4).view(1, 4, 4).expand(p1.size(0), 4, 4).to(p1) # [B, 4, 4]
        if p0_zero_mean:
            p0_m = p0.mean(dim=1) # [B, N, 3] -> [B, 3]
            a0[:, 0:3, 3] = p0_m
            q0 = p0 - p0_m.unsqueeze(1)
        else:
            q0 = p0

        if p1_zero_mean:
            #print(numpy.any(numpy.isnan(p1.numpy())))
            p1_m = p1.mean(dim=1) # [B, N, 3] -> [B, 3]
            a1[:, 0:3, 3] = -p1_m
            q1 = p1 - p1_m.unsqueeze(1)
        else:
            q1 = p1

        r = net(q0, q1, maxiter=maxiter, xtol=xtol) #Bx1024

        if p0_zero_mean or p1_zero_mean:
            #output' = trans(p0_m) * output * trans(-p1_m)
            #        = [I, p0_m;] * [R, t;] * [I, -p1_m;]
            #          [0, 1    ]   [0, 1 ]   [0,  1    ]
            est_g = net.g
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            net.g = est_g

            est_gs = net.g_series # [M, B, 4, 4]
            if p0_zero_mean:
                est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
            if p1_zero_mean:
                est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
            net.g_series = est_gs

        return r

    def forward(self, p0, p1, maxiter=10, xtol=1.0e-7):
        g0 = torch.eye(4).to(p0).view(1, 4, 4).expand(p0.size(0), 4, 4).contiguous()

        r, g, itr = self.do_em(g0, p0, p1, maxiter, xtol)

        #r, g, itr = self.iclk(g0, p0, p1, maxiter, xtol)

        self.g = g
        self.itr = itr
        return r

    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g)

    def approx_Jic(self, p0, f0, dt):
        # p0: [B, N, 3], Variable
        # f0: [B, K], corresponding feature vector
        # dt: [B, 6], Variable
        # Jk = (ptnet(p(-delta[k], p0)) - f0) / delta[k]

        batch_size = p0.size(0)
        num_points = p0.size(1)

        # compute transforms
        transf = torch.zeros(batch_size, 6, 4, 4).to(p0)
        for b in range(p0.size(0)):
            d = torch.diag(dt[b, :]) # [6, 6]
            D = self.exp(-d) # [6, 4, 4]
            transf[b, :, :, :] = D[:, :, :]
        transf = transf.unsqueeze(2).contiguous()  #   [B, 6, 1, 4, 4]
        p = self.transform(transf, p0.unsqueeze(1)) # x [B, 1, N, 3] -> [B, 6, N, 3]

        #f0 = self.ptnet(p0).unsqueeze(-1) # [B, K, 1]
        f0 = f0.unsqueeze(-1) # [B, K, 1]
        f = self.ptnet(p.view(-1, num_points, 3)).view(batch_size, 6, -1).transpose(1, 2) # [B, K, 6]

        df = f0 - f # [B, K, 6]
        J = df / dt.unsqueeze(1)

        return J

    def iclk(self, g0, p0, p1, maxiter, xtol):
        training = self.ptnet.training
        batch_size = p0.size(0)

        g = g0 #initial pose
        self.g_series = torch.zeros(maxiter+1, *g0.size(), dtype=g0.dtype)
        self.g_series[0] = g0.clone()

        if training:
            # first, update BatchNorm modules
            f0 = self.ptnet(p0)
            f1 = self.ptnet(p1)
        self.ptnet.eval() # and fix them.

        # re-calc. with current modules
        f0 = self.ptnet(p0) # [B, N, 3] -> [B, K]

        # approx. J by finite difference
        dt = self.dt.to(p0).expand(batch_size, 6)
        J = self.approx_Jic(p0, f0, dt)

        self.last_err = None
        itr = -1
        # compute pinv(J) to solve J*x = -r
        try:
            Jt = J.transpose(1, 2) # [B, 6, K]
            H = Jt.bmm(J) # [B, 6, 6]
            B = self.inverse(H)
            pinv = B.bmm(Jt) # [B, 6, K]
        except RuntimeError as err:
            # singular...?
            self.last_err = err
            #print(err)
            # Perhaps we can use MP-inverse, but,...
            # probably, self.dt is way too small...
            f1 = self.ptnet(p1) # [B, N, 3] -> [B, K]
            r = f1 - f0
            self.ptnet.train(training)
            return r, g, itr

        itr = 0
        r = None
        for itr in range(maxiter):
            self.prev_r = r
            p = self.transform(g.unsqueeze(1), p1) # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f = self.ptnet(p) # [B, N, 3] -> [B, K]
            r = f - f0

            dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

            # DEBUG.
            #norm_r = r.norm(p=2, dim=1)
            #print('itr,{},|r|,{}'.format(itr+1, ','.join(map(str, norm_r.data.tolist()))))

            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0 # no update.
                break

            g = self.update(g, dx)
            self.g_series[itr+1] = g.clone()

        rep = len(range(itr, maxiter))
        self.g_series[(itr+1):] = g.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

        self.ptnet.train(training)
        return r, g, (itr+1)


    def compute_gaussian_parameters(self, f_batch, cov_type="fixed"):
        batch_size = f_batch.shape[0]
        pi_batch = 1 * torch.ones(batch_size).to(f_batch.device)
        mu_batch = f_batch.mean(axis=1) #Bx64

        
        if cov_type == "diag":
            var = f_batch.var(axis=1)
            sigma_batch = torch.diag_embed(var)
        
        elif cov_type == "full":
            diff_mu = f_batch-mu_batch.unsqueeze(1) #BxNx64
            sigma_batch = torch.matmul(diff_mu.transpose(1,2), diff_mu)/(f_batch.shape[1]-1) #Bx64
            sigma_batch += torch.eye(sigma_batch.shape[1]).unsqueeze(0).to(sigma_batch.device) * eps
        else: 
            sigma = 1e-2
            sigma_batch = torch.eye(f_batch.shape[-1]).unsqueeze(0).to(f_batch.device) * sigma

        return pi_batch, mu_batch, sigma_batch



    def e_step(self, p_target, f_target, f_source): #features: BxNx64

        #deal with batch size
        list_m0 = []
        list_m1 = []
        
        batch_size = f_target.shape[0]


        if False:
            
            import matplotlib.pyplot as plt
            import visdom
            vis = visdom.Visdom()
            
            plt.hist(f_target[0,:, 0].detach().cpu().numpy().transpose(),fc=(1, 0, 0, 0.5), label="dim0")
            plt.hist(f_target[0,:,10].detach().cpu().numpy().transpose(),fc=(0, 1, 0, 0.5), label="dim10")
            plt.hist(f_target[0,:,20].detach().cpu().numpy().transpose(),fc=(0, 1, 1, 0.5), label="dim20")
            plt.hist(f_target[0,:,30].detach().cpu().numpy().transpose(),fc=(1, 0, 1, 0.5), label="dim30")
            plt.hist(f_target[0,:,40].detach().cpu().numpy().transpose(),fc=(0, 0, 0, 0.5), label="dim40")
            plt.hist(f_target[0,:,50].detach().cpu().numpy().transpose(),fc=(1, 1, 0, 0.5), label="dim50")
            plt.hist(f_target[0,:,60].detach().cpu().numpy().transpose(),fc=(0, 0, 1, 0.5), label="dim60")
            vis.matplot(plt)

        k = f_target.shape[-1]
        pi_batch, mu_batch, sigma_batch = self.compute_gaussian_parameters(f_target)
        
        for b in range(batch_size):
            import pdb; pdb.set_trace()
            #gmm = list_gmm[b]
            pi = pi_batch[b] #gmm["weights"]
            mu = mu_batch[b] #gmm["means"]
            sigma = sigma_batch[b] #gmm["covars"]
            sigma_inv = torch.inverse(sigma)
            

            diff = f_source[b].unsqueeze(1) - mu #NxJx3
            diff = diff.unsqueeze(3)

            #if torch.det(sigma) == 1:
            #    normalize = torch.det(sigma)
            #else:
            #    normalize = (2*np.pi)**(-k/2)* 1/ torch.sqrt(torch.det(sigma)+eps)
            normalize = 1

            dist = -0.5* torch.matmul((torch.matmul(diff.transpose(2,3), sigma_inv.unsqueeze(0).repeat(diff.shape[0],1,1,1)) ), diff).squeeze(2).squeeze(2)
            d_range = 70#10
            probs = torch.exp(torch.clamp(dist, -d_range, d_range))
            m0 = probs * pi.unsqueeze(0) #Nx1
            
            m1 = m0 * p_target[0] #Nx3

            #rij = exp_sum#/ (eps+exp_sum.sum(axis=1).unsqueeze(1)) 
            #don't normalize! it's ill conditioned when there's only one Gaussian...

            list_m0.append(m0)
            list_m1.append(m1)

            #import pdb; pdb.set_trace()            
            #m = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)
            #torch.exp(m.log_prob(points[0,0]))
            if rij.sum() != rij.sum():
                import pdb; pdb.set_trace()

        return list_m0, list_m1

    def compute_kabsch(self, source, target, weights):

        weight_sum = weights.sum()
        c_s = (weights* source).sum(axis=1)/(weight_sum+eps)
        c_t = (weights* target).sum(axis=1)/(weight_sum+eps)
        c_s = source.mean(axis=1)#(weights* source).sum(axis=1)/(weight_sum+eps)
        c_t = target.mean(axis=1)#(weights* target).sum(axis=1)/(weight_sum+eps)

        P = source-c_s.unsqueeze(1)
        Q = target-c_t.unsqueeze(1)

        L = torch.matmul((P * weights).transpose(1,2), Q)
        L_safe = L + 1e-4*L.mean()*torch.eye(3).to(L.device)#torch.rand_like(L)
        #L_safe = L_safe.cpu()
        #print(L)
        if L.sum() != L.sum():
            import pdb; pdb.set_trace()
        U, S, V = L_safe.svd()#torch.svd(L_safe)
        #U, S, V = L_safe.cpu().svd()#torch.svd(L_safe.cpu())
        #except:
        #import pdb; pdb.set_trace()
        #U = U.to(source.device)
        #V = V.to(source.device)

        
        R = torch.matmul(V, U.transpose(1,2))
        t = c_t-torch.matmul(R, c_s.unsqueeze(2)).squeeze(2)

        return R, t

    def m_step(self,p0, p1,list_m0, list_m1):
        
        batch_size = p0.shape[0] 
             _,m, dim = t_source.shape
        n = target.shape[1]
        m0, m1, m2, nx = estep_res
        c = w / (1.0 - w) * n / m

        m1m0 = m1/(m0+eps)#np.divide(m1.T, m0).T
        m0m0 = m0 / (m0 + c +eps)
        drxdx = m0m0 * 1.0 / sigma2#torch.sqrt(m0m0 * 1.0 / sigma2)

        if(drxdx.sum() != drxdx.sum() ) or (m1m0.sum() != m1m0.sum()):
            import pdb; pdb.set_trace()

        dr, dt = self.compute_kabsch(t_source, m1m0, drxdx) 
        rx = drxdx* (t_source - m1m0) #np.multiply(drxdx, (t_source - m1m0).T).T
        rot, t = torch.matmul(dr, trans_p.rot), torch.matmul(trans_p.t.unsqueeze(1), dr.transpose(1,2)) + dt.unsqueeze(1)
        t = t[:,0,:]
        q = torch.norm(rx, dim=2).sum()

        return RigidTransformation(R, t)        

    def do_em(self, g0, p0, p1, maxiter, xtol):
        #p0: the target BxNx64
        #p1: source BxNx64
        #g0: initial pose Bx4x4

        training = self.ptnet.training
        batch_size = p0.size(0)

        g = g0 #initial pose
        self.g_series = torch.zeros(maxiter+1, *g0.size(), dtype=g0.dtype)
        self.g_series[0] = g0.clone()

        if training:
            # first, update BatchNorm modules
            f0 = self.ptnet(p0, return_local=True)
            f1 = self.ptnet(p1, return_local=True)
        self.ptnet.eval() # and fix them.

        # re-calc. with current modules
        f0 = self.ptnet(p0, return_local=True) # [B, N, 3] -> [B, K]
        f0 = f0.permute(0,2,1)
        f1 = f1.permute(0,2,1)
        # approx. J by finite difference
        #dt = self.dt.to(p0).expand(batch_size, 6)
        #J = self.approx_Jic(p0, f0, dt)

        self.last_err = None
        #itr = -1
        # compute pinv(J) to solve J*x = -r
        #try:
        #    Jt = J.transpose(1, 2) # [B, 6, K]
        #    H = Jt.bmm(J) # [B, 6, 6]
        #    B = self.inverse(H)
        #    pinv = B.bmm(Jt) # [B, 6, K]
        #except RuntimeError as err:
        #    # singular...?
        #    self.last_err = err
        #    #print(err)
        #    # Perhaps we can use MP-inverse, but,...
        #    # probably, self.dt is way too small...
        #    f1 = self.ptnet(p1) # [B, N, 3] -> [B, K]
        #    r = f1 - f0
        #    self.ptnet.train(training)
        #    return r, g, itr
#
        itr = 0
        r = None
        for itr in range(maxiter):
            self.prev_r = r
            p = self.transform(g.unsqueeze(1), p1) # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f = self.ptnet(p, return_local=True).permute(0,2,1) # [B, N, 3] -> [B, K]
            
            #compute the target gaussian

            #do em
            list_m0, list_m1 = self.e_step(p0, f0, f)
            dx = self.m_step(p0, p1,list_m0, list_m1)

            r = f - f0

            #dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

            # DEBUG.
            #norm_r = r.norm(p=2, dim=1)
            #print('itr,{},|r|,{}'.format(itr+1, ','.join(map(str, norm_r.data.tolist()))))

            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0 # no update.
                break

            g = self.update(g, dx)
            self.g_series[itr+1] = g.clone()

        rep = len(range(itr, maxiter))
        self.g_series[(itr+1):] = g.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

        self.ptnet.train(training)
        return r, g, (itr+1)

#EOF