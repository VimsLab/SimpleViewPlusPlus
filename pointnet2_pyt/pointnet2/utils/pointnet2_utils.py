from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import etw_pytorch_utils as pt_utils
import sys
from torch import cos, sin
import numpy as np

try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        return _ext.furthest_point_sampling(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return _ext.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply

class KQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return _ext.k_query(new_xyz, xyz, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


k_query = KQuery.apply

# class Augmentor(nn.Module):
#     def __init__(self, npoint, nsample):
#         super(Augmentor, self).__init__()
#         self.npoint = npoint
#         self.nsample = nsample
#         self.scale_factor = nn.Parameter(torch.FloatTensor(self.npoint, 1), requires_grad=False)
#         self.translation_factor = nn.Parameter(torch.FloatTensor(self.npoint, 1), requires_grad=False)
#         self.jitter_factor = nn.Parameter(torch.FloatTensor(self.npoint, nsample,  3), requires_grad=False)
#         self.theta = nn.Parameter(torch.FloatTensor(self.npoint, 1), requires_grad=False)
#
#     def forward(self, new_points):
#         #initialing the PatchAugment parameters
#         torch.nn.init.uniform_(self.theta, a=-0.1, b=0.1)
#         torch.nn.init.uniform_(self.scale_factor, a=0.95, b=1.05)
#         torch.nn.init.uniform_(self.translation_factor, a=-0.05, b=0.05)
#         torch.nn.init.uniform_(self.jitter_factor, a=-0.01, b=0.01)
#
#         new_points_x = new_points[:, :, :, 0]
#         new_points_y = new_points[:, :, :, 1]
#         new_points_z = new_points[:, :, :, 2]
#         # print(self.scale_factor[0][0])#, self.theta[0][0], self.translation_factor[0][0],self.jitter_factor[0][0])
#         # #Scale
#         # if self.training:
#         new_points_x *= self.scale_factor
#         new_points_y *= self.scale_factor
#         new_points_z *= self.scale_factor
#
#         #Rotation
#         a = (cos(torch.sqrt(self.theta**2))**2).cuda()
#         b = (cos(torch.sqrt(self.theta**2))*sin(torch.sqrt(self.theta**2))**2 - sin(torch.sqrt(self.theta**2))*cos(torch.sqrt(self.theta**2))).cuda()
#         c = (sin(torch.sqrt(self.theta**2))*cos(torch.sqrt(self.theta**2))**2 + sin(torch.sqrt(self.theta**2))**2).cuda()
#         d = (sin(torch.sqrt(self.theta**2)) * cos(torch.sqrt(self.theta**2))).cuda()
#         e = (sin(torch.sqrt(self.theta**2))**3 + cos(torch.sqrt(self.theta**2))**2).cuda()
#         f = (sin(torch.sqrt(self.theta**2))**2*cos(torch.sqrt(self.theta**2))-sin(torch.sqrt(self.theta**2)) * cos(torch.sqrt(self.theta**2))).cuda()
#         g = (-sin(torch.sqrt(self.theta**2))).cuda()
#         h = (sin(torch.sqrt(self.theta**2)) * cos(torch.sqrt(self.theta**2))).cuda()
#         i = (cos(torch.sqrt(self.theta**2))**2).cuda()
#         # print(a,b,c,d,e,f,g,h,i)
#         new_points_x_r = new_points_x * a + new_points_y * b + new_points_z * c
#         new_points_y_r = new_points_x * d + new_points_y * e + new_points_z * f
#         new_points_z_r = new_points_x * g + new_points_y * h + new_points_z * i
#         #Translate
#         new_points_x = new_points_x_r + self.translation_factor
#         new_points_y = new_points_y_r + self.translation_factor
#         new_points_z = new_points_z_r + self.translation_factor
#
#         new_xyz_norm = torch.stack([new_points_x, new_points_y, new_points_z], 3)
#
#         #jitter
#         # print(new_xyz_norm.shape)
#         # print(self.jitter_factor.shape)
#         new_xyz_norm[:, :, :, :] += self.jitter_factor#[:, 0:new_xyz_norm.shape[2],:]
#         new_points = torch.cat([new_xyz_norm, new_points[:, :, :, 3:]], 3)
#         return new_points

class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, npoint, radius, nsample, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.npoint = npoint
        # self.augmentor = Augmentor(self.npoint, self.nsample)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        # new_xyz_pc = new_xyz.detach().cpu().numpy()
        # np.savez('new_xyz_pc', tensorname=new_xyz_pc)
        # print(new_xyz_pc.shape)
        idx = k_query(self.nsample, xyz, new_xyz)
        # idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        # aug_grouped_xyz = self.augmentor(torch.reshape(grouped_xyz, (grouped_xyz.shape[0],grouped_xyz.shape[2],grouped_xyz.shape[3], grouped_xyz.shape[1])))
        # print(aug_grouped_xyz.shape)
        # new_grouped_xyz = torch.reshape(aug_grouped_xyz, (aug_grouped_xyz.shape[0],aug_grouped_xyz.shape[3],aug_grouped_xyz.shape[1], aug_grouped_xyz.shape[2]))
        # print(grouped_xyz.shape)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
