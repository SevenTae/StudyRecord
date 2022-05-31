import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn


'''
需要：
anchor文件以及索引(anchors_mask)
numclass
input_shape


'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape     = [416, 416]
Cuda = True
mosaic = True
label_smoothing = 0
anchors_path    = '../model_data/yolo_anchorsdnf.txt'
classes_path = '../model_data/dnf.txt'
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


anchors, num_anchors =get_anchors(anchors_path)
class_names, num_classes= get_classes(classes_path)


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]], label_smoothing = 0, focal_loss = False, alpha = 0.25, gamma = 2):
        super(YOLOLoss, self).__init__()
        # -----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        # -----------------------------------------------------------#
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # xywhz+nuclass
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask #是个长度为3的列表 每一个列表代表一个尺度的那3个anchor

        self.label_smoothing = label_smoothing
        self.balance = [0.4, 1.0, 4] #每一个特整层的loss占比
        #下边这三部分是三部分损失的权重。。我也不知道为啥要这样算
        self.box_ratio = 0.05 #定位损失的权重
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2) #置信度损失的权重
        self.cls_ratio = 1 * (num_classes / 80) #类别损失的权重

        #选定貌似是选定正负样本的阈值
        self.ignore_threshold = 0.5
        self.cuda = cuda

    def forward(self, l, input, targets=None):
        # ----------------------------------------------------#
        #   l 代表使用的是第几个有效特征层
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #   targets 真实框的标签情况 [batch_size, num_gt, 5]
        # ----------------------------------------------------#
        # --------------------------------#
        #   获得图片数量，特征层的高和宽
        # --------------------------------#

        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        # -----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        # -----------------------------------------------------------------------#
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #   把anchor映射到对应的特征层上，也就是anchor/步长
        # -------------------------------------------------#
        #1.将anchor映射到对应的步长
        scaled_anchors= [( a_w/stride_w,a_h/stride_h)for a_w ,a_h in self.anchors]

        #---------------------------------
        #input 是网络原来的输出 形状是 batch 3*(5+numclass) ,13,13
        #把它转成 batch 3 13 13 5 +numclass的形状
        #---------------------------------

        #2.将网络的原输出转换个形式
        predicttion = input.view(bs,len(self.anchors_mask[l]),self.bbox_attrs,in_h,in_w).permute(0,1,3,4,2).contiguous()

        #3.获得网络的那些偏移量的输出
        #获得偏移量
        x =torch.sigmoid(predicttion[...,0])
        y =torch.sigmoid(predicttion[...,1])
        w = predicttion[...,2]
        h =predicttion[...,3]
        conf = torch.sigmoid(predicttion[...,4])
        pred_cls=torch.sigmoid(predicttion[...,5:])#!!他的种类置信度是sigmod！！

        #4.对target进行编码 编码成 batch 3 13 13 5+numclass
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        #5.对预测值进行解码
        # ---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        #l, 是对应特征层的索引号
        # x, y, h, w, 是上边网络输出的结果
        # targets 是原始没有处理的标签
        # scaled_anchors,是对应缩放后的anchor  in_h, in_w, noobj_mask是上边初步得出来的没有物体的mask蒙版
        # ----------------------------------------------------------------#
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        #6.基础工作（真实值编码，预测值解码）做完后 开始算loss了
        if self.cuda:
            y_true =y_true.type_as(x) #真实值编码的结果
            noobj_mask=noobj_mask.type_as(x) #没有物体的蒙版
            box_loss_scale = box_loss_scale.type_as(x)

            # --------------------------------------------------------------------------#
            #   box_loss_scale是真实框宽高的乘积，宽高均在0-1之间，因此乘积也在0-1之间。
            #   2-宽高的乘积代表真实框越大，比重越小，小框的比重更大。
            #   使用iou损失时，大中小目标的回归损失不存在比例失衡问题，故弃用..
            # --------------------------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale #没用了
        #7.重点来了
        loss = 0
        obj_mask = y_true[...,4]==1 #有物体的蒙版（batch 3 13 13）
        n = torch.sum(obj_mask) #看看多少物体是这一层上的
        # ---------------------------------------------------------------#
        #   yolo的原始思想 如果物体的中心点落在这个grid 里边 ...就有=由这个grid负责这个物体
        #这句话现在体现在 y_true 的那个confidenc 值如果==1就代表这个grid有物体，否则这个grid就没有物体
        #
        # ---------------------------------------------------------------#
        if n!=0:
            #计算定位（iou）和分类损失
            loss_cls = torch.mean(self.BCELoss(pred_cls[obj_mask],y_true[...,5:][obj_mask]))
            #ciou 计算是 pred 的（1 3 13 13 4） 和ytrue的（1 3 13 13 4）做ciou ..用现成的代码就行
            ciou = self.box_ciou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc  = torch.mean((1-ciou)[obj_mask])
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

        #计算置信度 #不太懂 但是下边的这个操作是将有物体的没有物体的conf一块给算了 所以只需要这一行就行 ，，要不就先死记把
        # |这玩意目前是取或 就是有True 则 True  全为flase 才是False
        #obj_mask有物体就是1没有物体就是0
        #BCELoss出来的还是3 13 13 的每个grid loss
        #[noobj_mask.bool() | obj_mask]
        loss_cnf = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_cnf * self.balance[l] * self.obj_ratio
        return loss


    def get_target(self,l, targets, anchors, in_h, in_w):
       #l对应的索引特征层，target原本送进来的
       bs = len(targets)

       #1.先初始化三个东西
       #----------------------
       #y_true,是个 初始化全0的batch 3 13 13 5+num class 的为了将target铺在上边
       # noobj_mask,是个初始化全1的掩模 batch 3 13 13 是个关于没有物体的掩模
       # box_loss_scale #是个初始化全0的  batch 3 13 13  为了记录哪些框是小目标，让网络更加关注小目标
       #---------------------

       y_true=torch.zeros(bs,len(self.anchors_mask[l]),in_h,in_w,self.bbox_attrs,requires_grad=False)
       noobj_mask=torch.ones(bs,len(self.anchors_mask[l]),in_h,in_w,requires_grad=False)
       box_loss_scale=torch.zeros(bs,len(self.anchors_mask[l]),in_h,in_w,requires_grad=False)

       #2.开始遍历bs 把真实值往里边铺
       for b in range(bs):
           if len(targets[b])==0: #如果这张图里边一个框也没有，直接跳出
               continue
           # 2.1初始化一个和原target一样的尺寸的全零矩阵 形状大概是 (numbox,5)
           batch_target = torch.zeros_like(targets[b])
           #2.2把原target的东西往里边填 ，原target是x, y，w,h
           #注意原来target的x, y都是归一在0-1之间的 往里边铺的的时候要乘以对应的w,h 比如13 比如5.3 4.8 就是中心点坐在5 4 这个格子
           batch_target[: ,[0,2]]=targets[b][:,[0,2]]*in_w
           batch_target[:,[1,3]]=targets[b][:,[1,3]]*in_h
           batch_target[:,4]=targets[b][:,4]
           batch_target=batch_target.cpu()

           #2.2把弄出个gt_box (num_true_box, 4) 。只取w,h x和y设置为0 为了和先验框左上角对其然后做iou寻找适合的anchor

           gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
           #2.3先验框也转换成 和gt_box一样的 （9，4） 然后前两个维度的值（x,y）都是0
           #anchor原来是个长度为9的list

           anchor_shapes =torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))

           #2.4第一次计算交并比 这个是用于选取哪个anchor与gt是最合适的
           #---------------------
           # self.calculate_iou(gt_box, anchor_shapes) 返回的是 （gt_数量 ，9） 代表每一个gt与每一个（9）anchor的交并比的值
           # torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1) 取最大的那个anchor的索引
           #关于计算交并比 这个直接用人家的代码吧
           #---------------------
           best_ns = torch.argmax(self.calculate_iou(gt_box,anchor_shapes),dim=-1)
           #返回一个索引比如一共5个gt 返回[0,0,0,4,0]
           #2.4 开始迭代这个几个anchor索引 ，找找是这个特征层上的anchor 的索引 比如13 是6 7 8
           for t, best_n in enumerate(best_ns): #这里的
               if best_n not in self.anchors_mask[l]:
                   continue

               #2.6如果一个是这个层上的也没有直接返回，这个函数也就到此为止，代表这一层的编码就那样了
               #2.6如果 还有gt的anchor是这一层上的比如 8 是13*13 这一层上的
               #2.6.1  判断这个先验框是当前特征点的哪一个先验框 这里是指6 7 8 中的哪一个
               k = self.anchors_mask[l].index(best_n) #比如8 这里的k就是2
               #2.6.2！！！重点来了
               #--------------------------
               #获得真实框是数据哪一个格子
               # torch.floor 即取不大于元素的最大整数
               #-------------------------
               i  =torch.floor(batch_target[t,0]).long() #t是指遍历到的第几个gt
               j =torch.floor(batch_target[t,1]).long()

               #取出真框的种类
               c = batch_target[t,4].long()
               #既然 ij这个格子有物体了 那么noj_mask 对应的位置设置为0
               noobj_mask[b,k,i,j]=0
               #对应的y_true 给填上
               y_true[b,k,i,j,0]=batch_target[t,0]
               y_true[b, k, i, j, 1] = batch_target[t, 1]
               y_true[b, k, i, j, 2] = batch_target[t, 2] #w
               y_true[b, k, i, j, 3] = batch_target[t, 3]#h
               y_true[b,k,i,j,c+5] = 1

               # ----------------------------------------#
               #   用于获得xywh的比例
               #   大目标loss权重小，小目标loss权重大
               #计算方式 w*h/in_w/in_h  ,先别管为啥人家就这么算的
               # ----------------------------------------#
               box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
       return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        bs = len(targets)
        #1.！！生成网格 13*13
        ##x坐标的网格
        #算了还是用别人的把
        #这一长串大概就是先生成一个0-12的一个一维数组shape =13 然后重复in_h次 变成 13*13的 再重复bs * len(self.anchors_mask[l])次，最后把它拉成 1 3 13 13 的大小
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        #2.生成先验框的宽高 （就是把对应层的对应的先验框的宽高给拿出来）
        scaled_anchors_l= np.array(scaled_anchors)[self.anchors_mask[l]]#找到对应层的anchor
        anchor_w =torch.Tensor(scaled_anchors_l).index_select(1,torch.LongTensor([0])).type_as(x) #w拿出来
        anchor_h =  torch.Tensor(scaled_anchors_l).index_select(1,torch.LongTensor([1])).type_as(x)# h拿出来

        #将anchor的宽和高整成和上边gride一样的batch 3 13 13 的样子
        anchor_w =anchor_w.repeat(bs,1).repeat(1,1,in_h*in_w).view(w.shape)
        anchor_h =anchor_h.repeat(bs,1).repeat(1,1,in_h*in_w).view(h.shape)


        #3.以上准备工作做完了 开始根据公式将网络输出的的偏移量和grid 和anchor连接起来
        pred_boxes_x = torch.unsqueeze(x + grid_x,-1)
        pred_boxes_y = torch.unsqueeze(y + grid_y ,-1)
        pred_boxes_w= torch.unsqueeze(torch.exp(w)*anchor_w,-1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h)*anchor_h , -1)
        pred_boxes = torch.cat([pred_boxes_x,pred_boxes_y,pred_boxes_w,pred_boxes_h],dim=-1)

        #这个pred_box 的形状是（batch ，3 13 13 4）
        #4.有开始计算iou 这次是为了选取正负样本
        for b in  range(bs):
            #4.1将预测结果转换一个形式
            #-------------------
            #原本是（3 13 13 4） ---》（3 * 13*13 ，4）
            #---------------------
            pred_boxes_for_ignore = pred_boxes[b].view(-1,4)
            #4.2(注意这部分和给真实值编码的时候差不多 但是那个时候是选取适合的anchor 现在这里
            # anchor选好了 ，在计算一边这个batch_target 是将gt与它选好的anchor做iou来选取正负样本
            # )
            # -------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            # -------------------------------------------------------#
            batch_target = torch.zeros_like(targets[b]) #这个玩意的形状是（num gt ，5）
            #4.2. 真实框的x,y w,h 取出来
            batch_target[:,[0,2]] =targets[b][:,[0,2]]*in_w
            batch_target[:,[1,3]]=targets[b][:,[1,3]]*in_h
            batch_target = batch_target[:,:4].type_as(x)

            #4.3计算交并比
            # -------------------------------------------------------#
            #   计算交并比
            #   anch_ious       num_true_box, num_anchors
            #   每一个truebox对应的这个层上（比如13*13）每一个ancho（3*13*13）的iou
            #
            #
            # -------------------------------------------------------#
            anchor_ious = self.calculate_iou(batch_target,pred_boxes_for_ignore)
            #4.4
            # -------------------------------------------------------#
            #   每个先验框对应真实框的最大重合度
            #在这里不是gt找anchor 是给每个anhor 分配gt
            #   anch_ious_max   num_anchors
            # -------------------------------------------------------#
            anchor_ious_max,_= torch.max(anchor_ious,dim=0) #torch.max(a, 0): 返回每一列的最大值,且返回索引(返回最大元素在各列的行..)
            #上边这个就是3*13*13=507 个最大iou的数
            #把它再reshape 成3 13 13
            anchor_ious_max =anchor_ious_max.view(pred_boxes[b].size()[:3])  #这个玩意是用来选取正负样本的

            #4.5选取“要忽略的样本” 也就是正负样本
            #-------------
            #它这里大于阈值的都是被忽略的（也就是正样本）  小于阈值的被noobj——mask记录下来 作为负样本

            #------------
            noobj_mask[b][anchor_ious_max>self.ignore_threshold]=0
        return noobj_mask, pred_boxes

    def calculate_iou(self, _box_a, _box_b):
        # -----------------------------------------------------------#
        #   计算真实框的左上角和右下角
        # -----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        # -----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        # -----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        # -----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        # -----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        # -----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        # -----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        # -----------------------------------------------------------#
        #   计算交的面积
        # -----------------------------------------------------------#
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        # -----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        # -----------------------------------------------------------#
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        # -----------------------------------------------------------#
        #   求IOU
        # -----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def clip_by_tensor(self, t, t_min, t_max):
        #这个玩意的功能就是差不多是平滑的意思 就是
        #clip_by_tensor作用是使数据在min到max之间，小于min的变为min，大于max的变为max
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2) #实现张量和标量之间逐元素求指数操作,

    def BCELoss(self, pred, target):#计算分类损失
        #因为那个类别用的俄式sigmod 我觉得可能会出现一个框它猫和狗的概率是一样大的情况
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        #二值交叉熵的公式： l = -ylog(pre)-(1-y)log(1-pred)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_ciou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # ----------------------------------------------------#
        #   求出预测框左上角右下角
        # ----------------------------------------------------#
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # ----------------------------------------------------#
        #   求出真实框左上角右下角
        # ----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # ----------------------------------------------------#
        #   求真实框和预测框所有的iou
        # ----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)

        # ----------------------------------------------------#
        #   计算中心的差距
        # ----------------------------------------------------#
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

        # ----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        # ----------------------------------------------------#
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        # ----------------------------------------------------#
        #   计算对角线距离
        # ----------------------------------------------------#
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
            b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
            b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = ciou - alpha * v
        return ciou


yololoss = YOLOLoss(anchors,num_classes,input_shape,Cuda)



out = np.load("../out.npy")  #网络的输出
tar= np.load("../target.npy")
l = 0
target = []
out = torch.from_numpy(out).to(device)  #这个有5个框
target.append(torch.from_numpy(tar))
loss_item = yololoss(l,out,target)
print(loss_item)