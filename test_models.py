# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix

from ops.dataset import TSNDataSet
# from ops.models import VideoNet
# from ops.models_test import VideoNet 
from ops.transforms import *
from opts_test import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
# from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'
def eval_video(video_data, net):
    net.eval()
    with torch.no_grad():
        i, data, label = video_data
        batch_size = label.numel()
        # print(data.size())
        # print(label.size())
        #+++++++++++++++++
        if args.dense_sample:
            num_crop = 10*args.test_crops
        elif args.twice_sample:
            num_crop = 2*args.test_crops
        else:
            num_crop = 1*args.test_crops
        #++++++++++++++++
        rst= net(data)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)
        #
        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        rst = rst.data.cpu().numpy().copy()

        return i, rst, label


def main():
    global args
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)

    assert args.modality == 'RGB'
    if args.test_list:
        test_list = args.test_list
    else:
        test_list = args.val_list

    # ==== get test args ====
    test_weights_list = args.test_weights.split(',')
    test_nets_list = args.test_nets.split(',')
    test_segments_list = [int(s) for s in args.test_segments.split(',')]
    assert len(test_nets_list) == len(test_segments_list)
    # test_cdivs_list = [int(s) for s in args.test_cdivs.split(',')]
    # =======================
    data_iter_list = []
    net_list = []

    scale_size = 256
    crop_size = 256 if args.full_res else 224 # 224 or 256 (scale_size)
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(crop_size),
            ])
    elif args.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(crop_size, scale_size, flip=False)
            ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(crop_size, scale_size, flip=False)
            ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(crop_size, scale_size)
            ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

    test_log = 'test_logs_256'
    if not os.path.exists(test_log):
        os.mkdir(test_log)

    log_path = './{}/log_{}_{}_{}_seg{}_{}.txt'.format(test_log, args.arch, args.dataset, "-".join(test_nets_list), \
        "-".join(str(a) for a in test_segments_list), \
        crop_size)

    if args.model == 'TSN':
        from ops.models_tsn import VideoNet
    elif args.model == 'TSM':
        from ops.models_tsm import VideoNet

    for this_net, this_segment, this_weight in zip(test_nets_list, test_segments_list, test_weights_list):
        
        model = VideoNet(num_class, this_segment, args.modality,
                backbone=args.arch, net=this_net,
                consensus_type=args.consensus_type,
                cdiv=args.cdiv)

        

        # weights_path = "./checkpoints/%s/%s_%s_c%d_s%d.pth"%(args.dataset, args.model, this_net, this_cdiv, this_segment)
        print(this_weight)
        if not os.path.exists(this_weight):
            raise ValueError('the checkpoint file doesnot exist: %s'%this_weight)

        checkpoint = torch.load(this_weight)
        print(checkpoint['best_prec1'])
        checkpoint_sd = checkpoint['state_dict']
        #print(checkpoint_sd.keys())
        
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint_sd.items())}
        for ks in list(base_dict.keys()):
            if ks.split('.')[-1] in ['total_params','total_ops']:
                base_dict.pop(ks)
                #print(ks)
        model.load_state_dict(base_dict)

        # crop_size = model.scale_size if args.full_res else model.input_size # 224 or 256 (scale_size)
        # scale_size = model.scale_size # 256
        input_mean = model.input_mean
        input_std = model.input_std

        # Data loading code
        if args.modality != 'RGBDiff':
            normalize = GroupNormalize(input_mean, input_std)
        else:
            normalize = IdentityTransform()

        if args.modality == 'RGB':
            data_length = 1
        elif args.modality in ['Flow', 'RGBDiff']:
            data_length = 5

        # print('----Validation----')
        print('batch size', args.batch_size)

        test_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, test_list, num_segments=this_segment,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   test_mode=True,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                   	    cropping,
                        GroupScale(224),   
                       # GroupScale(int(scale_size)),
                       
                       #GroupScale(256),
                       #GroupCenterCrop(224),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, twice_sample=args.twice_sample),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        

        #
        total_num = len(test_loader.dataset)
        print('total test number:', total_num)
        #
        model = torch.nn.DataParallel(model).cuda()
        model.eval()

        net_list.append(model)

        data_gen = enumerate(test_loader)
        data_iter_list.append(data_gen)
    #
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_times = AverageMeter()

    #
    proc_start_time = time.time()

    output = []
    fw = open(log_path, 'w')
    weights_data = np.zeros((num_class, 4, 4))
    for i, data_label_pairs in enumerate(zip(*data_iter_list)):
        with torch.no_grad():
            this_rst_list = []
            
            this_label = None
            # end = time.time()
            #weight_data = []
            for (_, (data, label)), net in zip(data_label_pairs, net_list):
                end = time.time()
                rst = eval_video((i, data, label), net)
                batch_times.update(time.time()-end, label.size(0))
                this_rst_list.append(rst[1])
                #weight_data = rst[3] #bsz, 4, num_blocks, 4
                this_label = label
            # assert len(this_rst_list) == len(coeff_list)
            # for i_coeff in range(len(this_rst_list)):
            #     this_rst_list[i_coeff] *= coeff_list[i_coeff]
            ensembled_predict = sum(this_rst_list) / len(this_rst_list)

            for p, g in zip(ensembled_predict, this_label.cpu().numpy()):
                output.append([p[None, ...], g])

            #for j in range(len(weight_data)):
            #    weight_data[j] = sum(weight_data[j]).cpu().numpy()
            
            #weight_data = np.array(weight_data) # 4 bsz 4 
            
            
            #weight_data = weight_data.transpose(1,0,2) # bsz 4 4
            #print(weight_data.shape)

            # for weight, l in zip(weight_data, this_label.cpu().numpy()): # 4, num_blocks, 4
            #     weights_data[l] = weights_data[l] + weight
                
            cnt_time = time.time() - proc_start_time
            prec1, prec5 = accuracy(torch.from_numpy(ensembled_predict), this_label, topk=(1, 5))
            top1.update(prec1.item(), this_label.numel())
            top5.update(prec5.item(), this_label.numel())
            if i % 20 == 0:
                txt = 'video {} done, total {}/{}, average {:.3f} sec/video, moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size, i * args.batch_size, total_num,
                                                            float(cnt_time) / (i+1) / args.batch_size, top1.avg, top5.avg)
                print(txt)
                fw.write(txt+'\n')
                fw.flush()

    # fw.close()

    print('avg computing time', batch_times.avg)
    video_pred = [np.argmax(x[0]) for x in output]
    video_pred_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output]

    video_labels = [x[1] for x in output]

    cf = confusion_matrix(video_labels, video_pred).astype(float)

    # np.save('cm.npy', cf)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt
    print(cls_acc*100)
    # upper = np.mean(np.max(cf, axis=1) / cls_cnt)
    # print('upper bound: {}'.format(upper))
    cls_acc_avg = np.sum(cls_acc*cls_cnt)/cls_cnt.sum()
    print(cls_acc_avg)
    #weights_data = weights_data/np.expand_dims(np.expand_dims(cls_cnt,-1).repeat(4,axis=-1),-1).repeat(4,axis=-1)

    # import csv
    # with open(args.test_nets+'_cls_acc.csv','w') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(cls_acc)
    # with open('cls_count.csv','w') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(cls_cnt.tolist())
    #     return 0
    # with open('cls_weight_layer1.csv','w') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerows((weights_data[:,0,:]/3).tolist())
    # with open('cls_weight_layer2.csv','w') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerows((weights_data[:,1,:]/4).tolist())
    # with open('cls_weight_layer3.csv','w') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerows((weights_data[:,2,:]/6).tolist())
    # with open('cls_weight_layer4.csv','w') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerows((weights_data[:,3,:]/3).tolist())
        


    print('-----Evaluation is finished------')
    print('Class Accuracy {:.02f}%'.format(cls_acc_avg*100))
    txt = 'Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg)
    fw.write(txt)
    fw.close()
    print(txt)


if __name__ == '__main__':
    main()
