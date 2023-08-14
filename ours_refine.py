import numpy as np
from matplotlib import  pyplot as plt


def txt2label(recog_content, dataset):
    mapping_file = "./data/" + dataset + "/mapping.txt"
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    recog_content_arr = np.empty(len(recog_content))
    for i in range(len(recog_content)):
        recog_content_arr[i] = actions_dict[recog_content[i]]
    return recog_content_arr


def label2txt(label, dataset):
    mapping_file = "./data/" + dataset + "/mapping.txt"
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    recognition = []
    for i in range(len(label)):
        recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(label[i].item())]]))

    return recognition


def proposal_cls2dict(cls_label):

    cate = cls_label[0]
    cls_start = 0
    proposal_dict_list = []
    for i in range(0, len(cls_label)):
        if cls_label[i] != cate:
            cls_end = i-1
            proposal_dict_list.append({"start_end": [cls_start, cls_end], "label": cate})
            cls_start = i
            cate = cls_label[i]
            i += 1
        if i == len(cls_label)-1:
            proposal_dict_list.append({"start_end": [cls_start, len(cls_label)-1], "label": cate})
    return proposal_dict_list


def dict2label(dict, cls_label):
    label = np.zeros(np.size(cls_label, 0))
    for i in range(0, len(dict)):
        start = dict[i]["start_end"][0]
        end = dict[i]["start_end"][1]
        label[start:end+1] = dict[i]["label"]

    return label


def refinement(cls, conf, gt, dataset):
    sample_rate_dic = {'50salads':2, 'gtea':1}
    sample_rate = sample_rate_dic[dataset]
    dic_cls = proposal_cls2dict(cls)
    dic_gt = proposal_cls2dict(gt)
    dic_cls_refine = proposal_cls2dict(cls)

    conf_sum_s = np.transpose(np.sum(conf[3, :, 0:19, :], 1))
    conf_sum_e = np.transpose(np.sum(conf[3, :, 38:57, :], 1))
    conf_sum_c = np.transpose(np.sum(conf[3, :, 19:38, :], 1))

    conf_sum_se = (conf_sum_s + conf_sum_e) * 5 / 19

    # plt.plot(np.arange(np.size(conf_sum_se, 0)), conf_sum_se)
    # plt.show()
    error_cls = [0, 2, 4, 5, 6, 10]
    for i in range(0, len(dic_cls)):
        duration_i = dic_cls[i]['start_end']
        class_i = dic_cls[i]['label']
        if duration_i[1] - duration_i[0] < 100:
            if (np.sum(conf_sum_se[int(duration_i[0]/sample_rate - 10):int(duration_i[0]/sample_rate + 10)]) +
                np.sum(conf_sum_se[int(duration_i[0] / sample_rate - 10): int(duration_i[0] / sample_rate + 10)]) +
                    np.sum(conf_sum_c[int(duration_i[0] / sample_rate): int(duration_i[0]/sample_rate)])) < 5:
                # if class_i in error_cls:
                dic_cls_refine[i] = dic_cls_refine[i-1]
    cls_refine = dict2label(dic_cls_refine, cls)
    return cls_refine


def conf_refine_main(recog_content, dataset, recog_file, gt_txt):
    cls = txt2label(recog_content, dataset)
    conf = np.load(recog_file)
    gt = txt2label(gt_txt, dataset)
    cls_refine = refinement(cls, conf, gt, dataset)
    cls_refine_txt = label2txt(cls_refine, dataset)
    return cls_refine_txt
