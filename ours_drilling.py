import numpy as np


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
        recognition.append(str(list(actions_dict.keys())[list(actions_dict.values()).index(label[i])]) + '\n')

    return recognition


def label_generate():
    label_list = []
    results_dir = './data/drilling/groundTruth'
    label_list.append(get_label(1))
    label_list.append(get_label(2))
    label_list.append(get_label(3))
    label_list.append(get_label(4))
    label_list.append(get_label(5))
    label_list.append(get_label(6))
    for i in range(0, len(label_list)):
        recognition = label2txt(label_list[i], 'drilling')
        f_name = '01_' + str(i+1) + '_crop' + str(i+1) + '.txt'
        f_ptr = open(results_dir + "/" + f_name, "w")
        f_ptr.write(''.join(recognition))
        f_ptr.close()
        print(i)


def project_234_label_correct():
    project_1_label = []
    project_1_label[0:389] = [3 for i in range(389 - 0 + 1)]
    project_1_label[390:589] = [1 for i in range(589 - 389)]
    project_1_label[590:909] = [2 for i in range(909 - 589)]
    project_1_label[910:1169] = [3 for i in range(1169 - 909)]
    project_1_label[1170:1308] = [1 for i in range(1308 - 1169)]
    project_1_label[1309:1705] = [2 for i in range(1705 - 1308)]
    project_1_label[1706:1912] = [3 for i in range(1912 - 1705)]
    project_1_label[1913:2218] = [1 for i in range(2218 - 1912)]
    project_1_label[2219:2468] = [2 for i in range(2468 - 2218)]
    project_1_label[2469:2806] = [3 for i in range(2806 - 2468)]
    project_1_label[2807:3042] = [1 for i in range(3042 - 2806)]
    project_1_label[3043:3279] = [2 for i in range(3279 - 3042)]
    project_1_label[3280:4084] = [3 for i in range(4084 - 3279)]
    project_1_label[4085:4239] = [1 for i in range(4239 - 4084)]
    project_1_label[4240:4518] = [2 for i in range(4518 - 4239)]
    project_1_label[4519:4798] = [3 for i in range(4798 - 4518)]
    project_1_label[4799:4970] = [1 for i in range(4970 - 4798)]
    project_1_label[4971:5246] = [2 for i in range(5246 - 4970)]
    project_1_label[5247:5538] = [3 for i in range(5538 - 5246)]
    project_1_label[5539:5737] = [1 for i in range(5737 - 5538)]
    project_1_label[5738:6008] = [2 for i in range(6008 - 5737)]
    project_1_label[6009:6245] = [3 for i in range(6245 - 6008)]
    project_1_label[6246:6523] = [1 for i in range(6523 - 6245)]
    project_1_label[6524:6688] = [2 for i in range(6688 - 6523)]
    project_1_label[6689:6901] = [3 for i in range(6901 - 6688)]

    project_2_label = []
    project_2_label[0:151] = [4 for i in range(151 - 0 + 1)]
    project_2_label[152:1122] = [5 for i in range(1122 - 151)]
    project_2_label[1123:1380] = [6 for i in range(1380 - 1122)]
    project_2_label[1381:1799] = [4 for i in range(1799 - 1380)]
    project_2_label[1800:2704] = [5 for i in range(2704 - 1799)]
    project_2_label[2705:2981] = [6 for i in range(2981 - 2704)]
    project_2_label[2982:3361] = [4 for i in range(3361 - 2981)]
    project_2_label[3362:4227] = [5 for i in range(4227 - 3361)]
    project_2_label[4228:4492] = [6 for i in range(4492 - 4227)]
    project_2_label[4493:4978] = [4 for i in range(4978 - 4492)]
    project_2_label[4979:5875] = [5 for i in range(5875 - 4978)]
    project_2_label[5876:6150] = [6 for i in range(6150 - 5875)]
    project_2_label[6151:6465] = [4 for i in range(6465 - 6150)]
    project_2_label[6466:7398] = [5 for i in range(7398 - 6465)]
    project_2_label[7399:7575] = [6 for i in range(7575 - 7398)]
    project_2_label[7576:7982] = [4 for i in range(7982 - 7575)]
    project_2_label[7983:9033] = [5 for i in range(9033 - 7982)]
    project_2_label[9034:9214] = [6 for i in range(9214 - 9033)]
    project_2_label[9215:9635] = [4 for i in range(9635 - 9214)]
    project_2_label[9636:10418] = [5 for i in range(10418 - 9635)]
    project_2_label[10419:10550] = [6 for i in range(10550 - 10418)]
    project_2_label[10551:11084] = [4 for i in range(11084 - 10550)]
    project_2_label[11085:12059] = [5 for i in range(12059 - 11084)]
    project_2_label[12060:12337] = [6 for i in range(12337 - 12059)]
    project_2_label[12338:12783] = [4 for i in range(12783 - 12337)]
    project_2_label[12784:13912] = [5 for i in range(13912 - 12783)]
    project_2_label[13913:14173] = [6 for i in range(14173 - 13912)]
    project_2_label[14174:14674] = [4 for i in range(14674 - 14173)]
    project_2_label[14675:15756] = [5 for i in range(15756 - 14674)]
    project_2_label[15757:16005] = [6 for i in range(16005 - 15756)]
    project_2_label[16006:16680] = [4 for i in range(16680 - 16005)]
    project_2_label[16681:17490] = [5 for i in range(17490 - 16680)]
    project_2_label[17491:17986] = [6 for i in range(17986 - 17490)]
    project_2_label[17987:18784] = [4 for i in range(18784 - 17986)]
    project_2_label[18785:19535] = [5 for i in range(19535 - 18784)]
    project_2_label[19536:19900] = [6 for i in range(19900 - 19535)]
    project_2_label[19901:20447] = [4 for i in range(20447 - 19900)]
    project_2_label[20448:21459] = [5 for i in range(21459 - 20447)]
    project_2_label[21460:21640] = [6 for i in range(21640 - 21459)]
    project_2_label[21641:22119] = [4 for i in range(22119 - 21640)]
    project_2_label[22120:23020] = [5 for i in range(23020 - 22119)]
    project_2_label[23021:23203] = [6 for i in range(23203 - 23020)]
    project_2_label[23204:23608] = [4 for i in range(23608 - 23203)]
    project_2_label[23609:24472] = [5 for i in range(24472 - 23608)]
    project_2_label[24473:24702] = [0 for i in range(24702 - 24472)]

    project_3_label = []
    project_3_label[0:37] = [3 for i in range(37 - 0 + 1)]
    project_3_label[38:535] = [1 for i in range(535 - 37)]
    project_3_label[536:975] = [2 for i in range(973 - 535)]
    project_3_label[976:1659] = [3 for i in range(1659 - 973)]
    project_3_label[1660:2459] = [1 for i in range(2459 - 1659)]
    project_3_label[2460:2932] = [2 for i in range(2932 - 2459)]
    project_3_label[2933:4355] = [3 for i in range(4355 - 2932)]
    project_3_label[4356:5498] = [1 for i in range(5498 - 4355)]
    project_3_label[5499:5938] = [2 for i in range(5938 - 5498)]
    project_3_label[5939:6696] = [3 for i in range(6696 - 5938)]
    project_3_label[6697:7366] = [1 for i in range(7366 - 6696)]
    project_3_label[7367:7845] = [2 for i in range(7845 - 7366)]
    project_3_label[7846:8287] = [3 for i in range(8287 - 7845)]
    project_3_label[8288:8842] = [1 for i in range(8842 - 8287)]
    project_3_label[8843:9281] = [2 for i in range(9281 - 8842)]
    project_3_label[9282:9631] = [3 for i in range(9631 - 9281)]
    project_3_label[9632:10150] = [1 for i in range(10150 - 9631)]
    project_3_label[10151:10609] = [2 for i in range(10609 - 10150)]
    project_3_label[10610:10931] = [3 for i in range(10931 - 10609)]
    project_3_label[10932:11762] = [1 for i in range(11762 - 10931)]
    project_3_label[11763:12208] = [2 for i in range(12208 - 11762)]
    project_3_label[12209:12655] = [3 for i in range(12655 - 12208)]
    project_3_label[12656:13282] = [1 for i in range(13282 - 12655)]
    project_3_label[13283:13812] = [2 for i in range(13812 - 13282)]
    project_3_label[13813:14606] = [3 for i in range(14606 - 13812)]
    project_3_label[14607:14846] = [0 for i in range(14846 - 14606)]
    project_3_label[14847:15908] = [1 for i in range(15908 - 14846)]
    project_3_label[15909:16655] = [2 for i in range(16655 - 15908)]
    project_3_label[16656:17273] = [1 for i in range(17273 - 16655)]
    project_3_label[17274:17686] = [2 for i in range(17686 - 17273)]
    project_3_label[17687:18067] = [3 for i in range(18067 - 17686)]
    project_3_label[18068:18939] = [1 for i in range(18939 - 18067)]
    project_3_label[18940:19018] = [2 for i in range(19018 - 18939)]
    project_3_label[19019:19238] = [1 for i in range(19238 - 19018)]
    project_3_label[19239:19349] = [2 for i in range(19349 - 19238)]
    project_3_label[19350:19573] = [1 for i in range(19573 - 19349)]
    project_3_label[19574:19976] = [2 for i in range(19976 - 19573)]
    project_3_label[19977:20269] = [3 for i in range(20269 - 19976)]
    project_3_label[20270:20553] = [0 for i in range(20553 - 20269)]

    project_4_label = []
    project_4_label[0:92] = [0 for i in range(92 - 0 + 1)]
    project_4_label[93:400] = [3 for i in range(400 - 92)]
    project_4_label[401:822] = [1 for i in range(822 - 400)]
    project_4_label[823:1313] = [2 for i in range(1313 - 822)]
    project_4_label[1314:1816] = [3 for i in range(1816 - 1313)]
    project_4_label[1817:2355] = [1 for i in range(2355 - 1816)]
    project_4_label[2356:2985] = [2 for i in range(2985 - 2355)]
    project_4_label[2986:3752] = [3 for i in range(3752 - 2985)]
    project_4_label[3753:7642] = [0 for i in range(7642 - 3752)]
    return project_1_label, project_2_label, project_3_label, project_4_label


def get_label(session_id, sample_labels=False):
    if session_id == 2 or session_id == 5:
        _, labels, _, _ = project_234_label_correct()
    elif session_id == 3 or session_id == 6:
        _, _, labels, _ = project_234_label_correct()
    elif session_id == 4:
        _, _, _, labels = project_234_label_correct()
    elif session_id == 1:
        labels, _, _, _ = project_234_label_correct()

    labels = np.array(labels)
    if sample_labels:
        # Since each chunk of video is 6 Frames, we sample the Labels at 6 labels per sample
        labels = labels[[i for i in range(0, len(labels), 12)]]
    ## Transform the Data
    if session_id == 5 or session_id == 6:
        labels = label_reverse(labels)
        labels.reverse()

    return labels


def label_reverse(label):
    label_rev = []
    # a = len(label)
    for i in range(0, len(label)):
        if label[i] != 0:
            label_rev.append(7 - label[i])
        else:
            label_rev.append(0)
    return label_rev

if __name__ == '__main__':
    # path_gtea = './data/gtea/features/S1_Cheese_C1.npy'
    # a = np.load(path_gtea)
    # path_drilling = './data/drilling/features/01_6_crop6.npy'
    # b = np.load(path_drilling)
    # np.save(path_drilling, np.transpose(b))
    label_generate()