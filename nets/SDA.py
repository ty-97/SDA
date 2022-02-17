import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy





class LST(nn.Module):
    def __init__(self, inplanes, planes=0, cdiv=1, num_segments=8, actiFunc='sigmoid', useBn=True, sqzFunc='avg'):
        super(LST, self).__init__()
        self.num_segments = num_segments
        self.useBn = useBn
        if sqzFunc == 'avg':
            self.sqz = nn.AdaptiveAvgPool3d(1)
        elif sqzFunc == 'max':
            self.sqz = nn.AdaptiveMaxPool3d(1)
        elif sqzFunc == 'avg + max':
            self.sqz = nn.ModuleList([])
            self.sqz.append(nn.AdaptiveAvgPool3d(1))
            self.sqz.append(nn.AdaptiveMaxPool3d(1))
        self.conv = nn.Linear(inplanes, inplanes, bias=False)
        if actiFunc == 'sigmoid':
            self.actifunc = nn.Sigmoid()
        elif actiFunc == 'relu':
            self.actifunc = nn.ReLU()
        if useBn:
            self.bn = nn.BatchNorm1d(inplanes)
            nn.init.constant_(self.bn.weight,1)
            nn.init.constant_(self.bn.bias,0)
        #
        nn.init.normal_(self.conv.weight, 0, 0.001)

    def forward(self, x):
        batch_size, c, t ,h ,w = x.size()
        #batch_size = bn//self.num_segments
        # x = x.view(batch_size, self.num_segments, c, h, w)
        # x = x.permute(0, 2, 1, 3, 4).contiguous()
        if isinstance(self.sqz, nn.ModuleList):
            y = torch.cat((self.sqz[0](x),self.sqz[1](x)),dim=1).view(batch_size*2, c)
        else:
            y = self.sqz(x).view(batch_size, c)
        #y = self.avg_pool(x).view(batch_size, c)
        y = self.conv(y)
        if self.useBn:
            y = self.bn(y)
        if isinstance(self.sqz, nn.ModuleList):
            y = y.view(batch_size,2,c).sum(dim=1)
        y = self.actifunc(y).view(batch_size, c, 1, 1, 1)

        x = y.expand_as(x)

        return x

class S122(nn.Module):
    def __init__(self, inplanes, planes=0, cdiv=1, num_segments=8, actiFunc='sigmoid', useBn=True, sqzFunc='avg'):
        super(S122, self).__init__()
        self.num_segments = num_segments
        self.useBn = useBn
        #self.conv = nn.Conv3d(inplanes, inplanes, kernel_size=(3,1,1),stride=(1,1,1), padding=(1,0,0), bias=False)
        #self.relu = nn.ReLU(inplace=True)

        if sqzFunc == 'avg':
            self.sqz = F.adaptive_avg_pool2d
            self.conv = nn.Conv3d(inplanes, inplanes, kernel_size=(3,1,1),stride=(1,1,1), padding=(1,0,0), bias=False)
        elif sqzFunc == 'max':
            self.sqz = F.adaptive_max_pool2d
            self.conv = nn.Conv3d(inplanes, inplanes, kernel_size=(3,1,1),stride=(1,1,1), padding=(1,0,0), bias=False)
        elif sqzFunc == 'avg + max':
            self.sqz = []
            self.sqz.append(F.adaptive_avg_pool2d)
            self.sqz.append(F.adaptive_max_pool2d)
            self.conv = nn.Conv3d(inplanes*2, inplanes, kernel_size=(3,1,1),stride=(1,1,1), padding=(1,0,0), bias=False)

        if actiFunc == 'sigmoid':
            self.actifunc = nn.Sigmoid()
            nn.init.xavier_normal_(self.conv.weight)
        elif actiFunc == 'relu':
            self.actifunc = nn.ReLU()
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu') 
            
        if self.useBn:
            self.bn = nn.BatchNorm3d(inplanes)
            nn.init.constant_(self.bn.weight,1)
            nn.init.constant_(self.bn.bias,0)
    
    def forward(self, x):
        batch_size, c, t ,h ,w = x.size()
        #batch_size = bn//self.num_segments
        # x = x.view(batch_size, self.num_segments, c, h, w)
        #x = x.contiguous()
        h_new = max(h//2, 2)
        w_new = max(w//2, 2)
        x = x.contiguous()

        if isinstance(self.sqz, list):
            y = torch.cat((self.sqz[0](x.view(-1, t, h, w), (h_new, w_new)),
                self.sqz[1](x.view(-1, t, h, w), (h_new, w_new))),dim=1)
            y = y.view(batch_size, c*2, t, h_new, w_new)
        else:
            y = self.sqz(x.view(-1, t, h, w), (h_new, w_new))
            y = y.view(batch_size, c, t, h_new, w_new)

        # y = F.adaptive_avg_pool2d(x.view(-1, t, h, w), (h_new, w_new))
        y = self.conv(y)
        if self.useBn:
            y = self.bn(y)
        y = self.actifunc(y)
        y = F.adaptive_avg_pool2d(y.view(-1, t, h_new, w_new), (h, w))
        #x = x.expand(batch_size, c, num_segments, h, w)
        x = y.view(batch_size, c, t, h, w)
        return x

class LT(nn.Module):
    def __init__(self, inplanes, planes=0, cdiv=1, num_segments=8, actiFunc='sigmoid', sqzFunc='avg'):
        super(LT, self).__init__()
        self.num_segments = num_segments
        self.sqzFunc=sqzFunc
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        if self.sqzFunc == 'avg' or self.sqzFunc == 'max':
            self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)    
        elif self.sqzFunc == 'avg + max':
            self.conv = nn.Conv2d(inplanes*2, inplanes, kernel_size=3, stride=1, padding=1, bias=False)    
        self.bn = nn.BatchNorm2d(inplanes)
        #self.relu = nn.ReLU(inplace=True)
        if actiFunc == 'sigmoid':
            self.actifunc = nn.Sigmoid()
            nn.init.xavier_normal_(self.conv.weight)
        elif actiFunc == 'relu':
            self.actifunc = nn.ReLU()
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu') 
        
        nn.init.constant_(self.bn.weight,1)
        nn.init.constant_(self.bn.bias,0)
        
    
    def forward(self, x):
        batch_size, c, t ,h ,w = x.size()
        if self.sqzFunc == 'avg':
            y = x.mean(dim=2)
        elif self.sqzFunc == 'max':
            y = x.max(dim=2)[0]
        elif self.sqzFunc == 'avg + max':
            y = torch.cat((x.mean(dim=2),x.max(dim=2)[0]),dim=1)
        y = self.conv(y)
        y = self.bn(y)
        #x = self.relu(x)
        y = self.actifunc(y).view(batch_size, c, 1, h, w)
        x = y.expand_as(x)

        return x

class LS(nn.Module):
    def __init__(self, inplanes, planes=0, cdiv=1, num_segments=8, actiFunc='sigmoid', sqzFunc='avg'):
        super(LS, self).__init__()
        self.num_segments = num_segments
        if sqzFunc == 'avg':
            self.sqz = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)        
        elif sqzFunc == 'max':
            self.sqz = nn.AdaptiveMaxPool2d(1)
            self.conv = nn.Conv1d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)        
        elif sqzFunc == 'avg + max':
            self.sqz = nn.ModuleList([])
            self.sqz.append(nn.AdaptiveAvgPool2d(1))
            self.sqz.append(nn.AdaptiveMaxPool2d(1))
            self.conv = nn.Conv1d(inplanes*2, inplanes, kernel_size=3, stride=1, padding=1, bias=False)        
        
        self.bn = nn.BatchNorm1d(inplanes)
        #self.relu = nn.ReLU(inplace=True)
        if actiFunc == 'sigmoid':
            self.actifunc = nn.Sigmoid()
            nn.init.xavier_normal_(self.conv.weight)
        elif actiFunc == 'relu':
            self.actifunc = nn.ReLU()
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu') 
      
        nn.init.constant_(self.bn.weight,1)
        nn.init.constant_(self.bn.bias,0)
    
    def forward(self, x):
        #print(x.shape)
        batch_size, c, t ,h ,w = x.size()
        #batch_size = bn//self.num_segments
        # x = x.view(batch_size, self.num_segments, c, h, w)
        # x = x.contiguous() #b c t h w

        y = x.view(batch_size, c*t, h, w)
        if isinstance(self.sqz, nn.ModuleList):
            y = torch.cat((self.sqz[0](y),self.sqz[1](y)),dim=1).view(batch_size, c*2, t)
        else:
            y = self.sqz(y).view(batch_size, c, t)

        #print(x.shape)
        y = self.conv(y)
        y = self.bn(y)
        #x = self.relu(x)
        y = self.actifunc(y).view(batch_size, c, t, 1, 1)
        x = y.expand_as(x)

        return x

class sec_agg(nn.Module):
    def __init__(self, emb_d, ng_slope=0.02):
        super(sec_agg, self).__init__()
        self.emb_d = emb_d
        self.qVec = nn.Parameter(torch.randn(1, emb_d))
        self.qVec.data.uniform_(-0.1, 0.1)
        # self.activation = nn.LeakyReLU(negative_slope=ng_slope)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        b, c, t, h, w = x[0].size()
        n_depens = len(x)
        if str(x[0].device) != 'cpu':
            K = torch.zeros(b, n_depens, c).cuda()
        else: 
            K = torch.zeros(b, n_depens, c)
        for i in range(n_depens):
            K[:, i, :] = self.avg_pool(x[i]).squeeze()
        K = K.view(-1, c)
        projection = torch.mm(K, self.qVec.transpose(0, 1)) # [bsz*n_depens, 1]
        K = K.view(-1, n_depens, c)
        size = K.size() #[b, n_depens, c]
        projection_mat = torch.reshape(projection, (-1, n_depens, 1)) # [bsz, n_depens, 1]
        max_mat = torch.max(projection_mat, 1, keepdim=True)[0] # [b, 1, 1]
        proj_exp = torch.exp(projection_mat-max_mat) # [b, n_depens, 1]
        normalize = torch.sum(proj_exp, 1) # [b, 1]
        normalize = torch.unsqueeze(normalize, 1) # [b, 1, 1]
        proj_softmax = proj_exp / (normalize + 1e-08) # [b, n_depens, 1]
        # proj_softmax = proj_softmax.permute(0, 2, 1).contiguous() # [b, 1, n_depens]
        # outputs = torch.bmm(proj_softmax, x) # [b, 1, c]
        # outputs.view(size[0], -1) # [bsz, m*emb_d]
        proj_softmax = proj_softmax.squeeze(2)
        # print(proj_softmax.size())
        for j in range(n_depens):
            if j == 0:
                tt = proj_softmax[:, j].unsqueeze(1).view(b,1,1,1,1)
                y = x[j]*tt.expand_as(x[j])
            else:
                tt = proj_softmax[:, j].unsqueeze(1).view(b,1,1,1,1)
                y = y + x[j]*tt.expand_as(x[j])
        return y



class SDA(nn.Module):
    def __init__(self, inplanes, DMB = ['LST','LT','LS','S122'], aggregation='sec_agg', cdiv=4, num_segments=8):
        super(SDA, self).__init__()
        self.DMB_dir = {'LST': LST, 'LT': LT, 'LS': LS, 'S122': S122}
        self.cdiv = cdiv
        self.num_segments = num_segments
        self.reduc_planes = max(inplanes//cdiv, 16)
        self.channel_down = nn.Conv3d(inplanes, self.reduc_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(self.reduc_planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_up = nn.Conv3d(self.reduc_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(inplanes)
        self.sigmoid = nn.Sigmoid()
        # self.channel_up = nn.AdaptiveAvgPool1d(inplanes)
        # self.up_pool = nn.AdaptiveAvgPool1d(inplanes)
        self.mdm = nn.ModuleList([])

        for block in DMB:
            self.mdm.append(self.DMB_dir[block](inplanes=self.reduc_planes, actiFunc='relu'))

        if aggregation == 'sec_agg':
            self.agg = sec_agg(self.reduc_planes) 
        
        self.aggregation = aggregation

        nn.init.kaiming_normal_(self.channel_down.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.channel_up.weight)
        nn.init.constant_(self.bn1.weight,1)
        nn.init.constant_(self.bn1.bias,0)
        nn.init.constant_(self.bn2.weight,1)
        nn.init.constant_(self.bn2.bias,0)
    def forward(self, x):
        y = []
        bn, c, h, w = x.size()
        b = bn//self.num_segments
        x = x.view(-1, self.num_segments, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        temp = self.channel_down(x)
        temp = self.bn1(temp)
        temp = self.relu(temp)
        for block in self.mdm:
            y.append(block(temp))
       
        y = self.agg(y)
        # print("shape: {}".format(y.shape))
        y = self.channel_up(y)
        y = self.bn2(y)
        y = self.sigmoid(y)
        x = x*y
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, c, h, w)
        return x



