import torch
import torch.nn as nn
from all_utils import DATASET_NUM_CLASS
from models.model_utils import Squeeze, BatchNormPoint
from models.mv_utils import PCViews
# from pointnet2.models.pointnet2_msg_cls import Pointnet2SSG
from pointnet2.utils.pointnet2_modules import PointnetSAModule
import numpy as np

class MVModel(nn.Module):
    def __init__(self, task, dataset, backbone,
                 feat_size):

        super().__init__()
        assert task == 'cls'
        self.task = task
        self.num_class = DATASET_NUM_CLASS[dataset]
        self.dropout_p = 0.5
        self.feat_size = feat_size

        pc_views = PCViews()
        self.num_views = pc_views.num_views
        self._get_img = pc_views.get_img

        img_layers, in_features = self.get_img_layers(
            backbone, feat_size=feat_size)
        self.img_model = nn.Sequential(*img_layers)

        self.final_fc = MVFC(
            num_views=self.num_views*16,
            in_features=in_features*2,
            out_features=self.num_class,
            dropout_p=self.dropout_p)

        self.sampgroup = PointnetSAModule(
            npoint=16,
            radius=0.5,
            nsample=256,
            mlp=[0,256,512,128],
            use_xyz=True,
        )
        # 8, 1024, 3 -- 8, 256 -- > 8 8 256 3 --> 64, 256, 3 --> 8, 8, 6, 128, 128 --> 8, 48, 128, 128

    def forward(self, pc):
        """
        :param pc:
        :return:
        """
        # image, max of neighbors and neighbor point features
        res = 32
        pc = pc.cuda()
        #sample and group 3D points to get neighborhoods and
        new_xyz, new_pc, neigh_pc_feat = self.sampgroup(pc)
        new_pc = ((new_pc.permute(0,2,3,1)).reshape(-1, 256, 3))*2.5
        # max of 3D features learnt from the point cloud to represent the global features of the point cloud
        neigh_pc_feat = torch.max(neigh_pc_feat.permute(0,2,1),1)[0]#.repeat(6,1)

        img = self.get_img(pc,1)
        #capture orthogonal perspective views of neighborhoods with 2 as the point size
        new_img = self.get_img(new_pc,2)
        # repeat object views m times i.e. 16 times
        A = img.view((pc.shape[0],-1,res,res)).repeat(1,16,1,1)
        # restructure the tensor of neighobor views
        B = new_img.view((pc.shape[0], -1, res,res))
        # concatenate respective object and neighbor views
        final_img = torch.cat((A,B),3)
        # resructure the tensor to create intermediate samples
        final_img = final_img.view((-1, 1, res,res*2))
        # feed the features
        feat = self.img_model(final_img)
        # concatenate features from object and neighbor views with features from the 3D point cloud
        final_feat = torch.cat((neigh_pc_feat.repeat(96,1),feat),1)
        # pass learnt features into final classifier
        logit = self.final_fc(final_feat)
        out = {'logit': logit}
        return out

    def get_img(self, pc, size):
        img = self._get_img(pc, size)
        img = torch.tensor(img).float()
        img = img.to(dtype=torch.float32)#next(self.parameters()).device)
        assert len(img.shape) == 3
        img = img.unsqueeze(3)
        # [num_pc * num_views, 1, RESOLUTION, RESOLUTION]
        img = img.permute(0, 3, 1, 2)

        return img

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from models.resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2,2,2,2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1), dilation=[1,1],
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]
        print(img_layers)

        return img_layers, in_features


class MVFC(nn.Module):
    """
    Final FC layers for the MV model
    """

    def __init__(self, num_views, in_features, out_features, dropout_p):
        super().__init__()
        self.num_views = num_views
        self.in_features = in_features
        self.model = nn.Sequential(
                BatchNormPoint(in_features),
                # dropout before concatenation so that each view drops features independently
                nn.Dropout(dropout_p),
                nn.Flatten(),
                nn.Linear(in_features=in_features * self.num_views,
                          out_features=in_features),
                nn.BatchNorm1d(in_features),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(in_features=in_features, out_features=out_features,
                          bias=True))

    def forward(self, feat):
        feat = feat.view((-1, self.num_views, self.in_features))
        # print(feat.shape)
        out = self.model(feat)
        return out
