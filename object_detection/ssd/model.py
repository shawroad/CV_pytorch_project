"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-19
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
from utils import cxcy_to_xy, gcxgcy_to_cxcy, find_jaccard_overlap, cxcy_to_gcxgcy, xy_to_cxcy


class VGGBase(nn.Module):
    """
    VGG base convolutions to produce feature maps.
    完全采用vgg16的结构作为特征提取模块，丢掉fc6和fc7两个全连接层。
    因为vgg16的ImageNet预训练模型是使用224×224尺寸训练的，因此我们的网络输入也固定为224×224
    """
    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)    # 224->112

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)    # 112->56

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)    # 56->28

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)    # 28->14

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)    # 14->7
        # Load pretrained weights on ImageNet
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 224, 224)
        :return: feature maps pool5
        """
        out = F.relu(self.conv1_1(image))  # (N, 64, 224, 224)
        out = F.relu(self.conv1_2(out))  # (N, 64, 224, 224)
        out = self.pool1(out)  # (N, 64, 112, 112)

        out = F.relu(self.conv2_1(out))  # (N, 128, 112, 112)
        out = F.relu(self.conv2_2(out))  # (N, 128, 112, 112)
        out = self.pool2(out)  # (N, 128, 56, 56)

        out = F.relu(self.conv3_1(out))  # (N, 256, 56, 56)
        out = F.relu(self.conv3_2(out))  # (N, 256, 56, 56)
        out = F.relu(self.conv3_3(out))  # (N, 256, 56, 56)
        out = self.pool3(out)  # (N, 256, 28, 28)

        out = F.relu(self.conv4_1(out))  # (N, 512, 28, 28)
        out = F.relu(self.conv4_2(out))  # (N, 512, 28, 28)
        out = F.relu(self.conv4_3(out))  # (N, 512, 28, 28)
        out = self.pool4(out)  # (N, 512, 14, 14)

        out = F.relu(self.conv5_1(out))  # (N, 512, 14, 14)
        out = F.relu(self.conv5_2(out))  # (N, 512, 14, 14)
        out = F.relu(self.conv5_3(out))  # (N, 512, 14, 14)
        out = self.pool5(out)  # (N, 512, 7, 7)

        # return 7*7 feature map
        return out

    def load_pretrained_layers(self):
        """
        we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)
        print("\nLoaded base model.\n")


class PredictionConvolutions(nn.Module):
    """
    预测坐标和类别
    """
    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()
        self.n_classes = n_classes
        n_boxes = 9   # 每个位置有9个先验框

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv = nn.Conv2d(512, n_boxes * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv = nn.Conv2d(512, n_boxes * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, pool5_feats):
        batch_size = pool5_feats.size(0)

        # 预测坐标
        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv = self.loc_conv(pool5_feats)  # (N, n_boxes * 4, 7, 7)
        l_conv = l_conv.permute(0, 2, 3, 1).contiguous()   # (N, 7, 7, n_boxes * 4)
        # print(l_conv.size())   # torch.Size([2, 7, 7, 36])
        # (N, 7, 7, n_boxes * 4), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        locs = l_conv.view(batch_size, -1, 4)  # (N, 441, 4), there are a total 441 boxes on this feature map

        # 预测类别
        # Predict classes in localization boxes
        c_conv = self.cl_conv(pool5_feats)  # (N, n_boxes * n_classes, 7, 7)
        c_conv = c_conv.permute(0, 2, 3, 1).contiguous()  # (N, 7, 7, n_boxes * n_classes), to match prior-box order (after .view())
        classes_scores = c_conv.view(batch_size, -1, self.n_classes)  # (N, 441, n_classes), there are a total 441 boxes on this feature map
        return locs, classes_scores


class Model(nn.Module):
    """
    The tiny_detector network
    包含一个VGG作为特征提取模块，并在最后一个特征图上添加一个输出头来预测目标框信息
    """
    def __init__(self, n_classes):
        super(Model, self).__init__()
        self.n_classes = n_classes
        self.base = VGGBase()
        self.pred_convs = PredictionConvolutions(n_classes)
        self.priors_cxcy = self.create_prior_boxes()
        self.device = 'cpu'

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 224, 224)
        :return: 441 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        self.device = image.device
        # VGG提取特征
        pool5_feats = self.base(image)  # (N, 512, 7, 7)   torch.Size([2, 512, 7, 7])

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(pool5_feats)    # (N, 441, 4), (N, 441, n_classes)
        return locs, classes_scores

    def create_prior_boxes(self):
        """
        7*7*9 = 441 总共要创建441个先验框  这里之所以7*7 是因为VGG16最后的特征图尺寸为7*7
        我们为特征图上每一个cell定义了共9种不同大小和形状的候选框（3种尺度*3种长宽比=9）
        :return: prior boxes in center-size coordinates, a tensor of dimensions (441, 4)
        """
        fmap_dims = 7
        obj_scales = [0.2, 0.4, 0.6]
        aspect_ratios = [1., 2., 0.5]

        prior_boxes = []
        for i in range(fmap_dims):
            for j in range(fmap_dims):
                cx = (j + 0.5) / fmap_dims
                cy = (i + 0.5) / fmap_dims

                for obj_scale in obj_scales:
                    for ratio in aspect_ratios:
                        prior_boxes.append([cx, cy, obj_scale * math.sqrt(ratio), obj_scale / math.sqrt(ratio)])

        prior_boxes = torch.FloatTensor(prior_boxes)  # (441, 4)
        prior_boxes.clamp_(0, 1)  # (441, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 441 locations and class scores (output of the tiny_detector) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 441 prior boxes, a tensor of dimensions (N, 441, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 441, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 441, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images in batch
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (441, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (441)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (441)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 441
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(self.device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with current box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, (overlap[box] > max_overlap).to(torch.uint8))
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(self.device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
                image_labels.append(torch.LongTensor([0]).to(self.device))
                image_scores.append(torch.FloatTensor([0.]).to(self.device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The loss function for object detection.
    对于Loss的计算，完全遵循SSD的定义，即 MultiBox Loss
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes.
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.device = 'cuda'

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.
        :param predicted_locs: (batch_size, 441, 4)  模型预测位置
        :param predicted_scores: (batch_size, 441, n_classes)  模型预测类别
        :param boxes: len(boxes) = batch_size  列表中每个元素是一组坐标tensor
        [tensor([[0.7401, 0.3805, 0.9718, 0.6916],[0.7147, 0.3805, 0.9675, 0.9146]]), tensor([[0.5558, 0.1802, 0.5844, 0.3231]]..]
        :param labels: [tensor([15, 13]), tensor([ 5,  5,  5,  5, 15, 15])]
        :return: multibox loss, a scalar
        """
        self.device = predicted_locs.device
        batch_size = predicted_locs.size(0)    # batch_size
        n_priors = self.priors_cxcy.size(0)    # 先验框 441
        n_classes = predicted_scores.size(2)    # 类别数 21
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)  # (N, 441, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)  # (N, 441)
        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)   # 当前样本中的目标个数

            # 计算IOU
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 441)

            # 对于每个先验，找到具有最大重叠的对象
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (441)

            # 首先，找到每个对象具有最大重叠的先验。
            _, prior_for_each_object = overlap.max(dim=1)

            # 然后，将每个对象分配给相应的最大重叠先验。
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(self.device)

            # 为了确保这些先验合格，人为地给它们一个大于 0.5 的重叠。
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (441)
            # 将与对象重叠小于阈值的先验设置为背景（无对象）
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (441)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (441, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 441)

        # 计算位置损失，仅在正例（非背景）先验上计算
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])

        # 每张图像的正面和硬负面先验数量
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # 类别损失
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 441)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 441)

        # 我们已经知道哪些先验是积极的
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # 接下来，找出哪些先验是难负的
        # 为此，仅对每个图像中的负先验进行排序，以降低损失的顺序并取顶部 n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 441)
        conf_loss_neg[positive_priors] = 0.  # (N, 441), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 441), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self.device)  # (N, 441)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 441)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # 仅对正先验进行平均，尽管对正先验和硬负先验进行了计算
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar
        # return TOTAL LOSS
        return conf_loss + self.alpha * loc_loss

