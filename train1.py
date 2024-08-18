import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn import metrics

from utils import load_data, build_graph, weight_reset,load_datav3
from model2 import HHOMR


def Train(directory, epochs, n_classes, in_size, out_dim, dropout, slope, lr, wd, random_seed, cuda):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        if not cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(random_seed)


    context = torch.device('cpu')

    #   ID:850*850  IM:1057*1057
    g, disease_vertices, mirna_vertices, ID, IM, samples,Adj = build_graph(directory, random_seed)

    #  加载v3数据集ID:374*374,788*788

    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])

    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    #????????????????
    print('## disease nodes:', torch.sum(g.ndata['type'] == 1).numpy())
    print('## mirna nodes: ', torch.sum(g.ndata['type'] == 0).numpy())

    g.to(context)

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    prc_result = []

    fprs = []
    tprs = []
    precisions = []
    recalls = []

    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    for train_idx, test_idx in kf.split(samples[:, 2]):     # 返回训练集和测试集的索引train：test 4:1
        i += 1
        print('Training for Fold', i)

        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1 # 多加一列，将训练集记为1

        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))

        edge_data = {'train': train_tensor}

        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)        # 正向反向加边，更新边上的数据
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)
        #???????

        train_eid = g.filter_edges(lambda edges: edges.data['train'])       # 过滤出被记为train的边
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)       # 从异构图中创建子图，train集的子图
        # g_train.copy_from_parent()
        label_train = g_train.edata['label'].unsqueeze(1)
        src_train, dst_train = g_train.all_edges()          # 训练集的边

        ######################################################
        # 节点 类特征
        node_type1 = torch.tensor([1.0, 0.0])  # 只有两类节点，一个mrna，一个疾病
        node_type_feature1 = torch.stack([node_type1] * 383)
        # node_type_feature1 = torch.stack([node_type1] * 374)
        node_type2 = torch.tensor([0.0, 1.0])
        node_type_feature2 = torch.stack([node_type2] * 495)
        # node_type_feature2 = torch.stack([node_type2] * 788)
        # 图的节点信息，区分每一个节点
        # Topo = torch.eye(1162)
        Topo = torch.eye(878)

        ######################################################

        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)   # 原图中选出标记为0的记为测试集
        src_test, dst_test = g.find_edges(test_eid)
        label_test = g.edges[test_eid].data['label'].unsqueeze(1)       # 测试集的边
        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        from node2vec import Node2Vec
        nx_graph = g_train.to_networkx()
        node2vec = Node2Vec(nx_graph, dimensions=64, p=2,q=0.5,walk_length=30, num_walks=200, workers=1)
        model2 = node2vec.fit(window=3, min_count=1, batch_words=4)
        TP = model2.wv.vectors


        model = HHOMR(G=g_train,
                      hid_dim=in_size,
                      n_class=n_classes,
                      S=1,
                      K=4,
                      batchnorm=True,
                      num_diseases=ID.shape[0],
                      num_mirnas=IM.shape[0],
                      d_sim_dim=ID.shape[1],
                      m_sim_dim=IM.shape[1],
                      out_dim=out_dim,
                      dropout=dropout,
                      slope=slope,
                      node_dropout=0.2,
                      input_droprate=0.2,
                      hidden_droprate=0.2,

                      nfeat=64,
                      nlayers=2,
                      nhidden=64,
                      lamda=0.5,
                      alpha=0.1,
                      use_center_moment=False,
                      moment=10

                      )

        model.apply(weight_reset)
        model.to(context)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss = nn.BCELoss()

        for epoch in range(epochs):
            start = time.time()

            model.train()
            with torch.autograd.set_detect_anomaly(True):
                score_train = model(g_train, src_train, dst_train,node_type_feature1,node_type_feature2,TP,True)  # train集子图进入model训练
                loss_train = loss(score_train, label_train)

                optimizer.zero_grad()   # 梯度置零
                loss_train.backward()   # 反向传播
                optimizer.step()

            model.eval()
            with torch.no_grad():   # with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
                score_val = model(g, src_test, dst_test,node_type_feature1,node_type_feature2,TP, True)    # 注意在整个图g中训练测试集
                loss_val = loss(score_val, label_test)



            end = time.time()

            if (epoch + 1) % 10 == 0:
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                      'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
                      'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
                      'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Time: %.2f' % (end - start))

        model.eval()
        with torch.no_grad():
            score_test = model(g, src_test, dst_test,node_type_feature1,node_type_feature2,TP, True)   # 测试分数和验证分数相同？？？

        score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())  # np.squeeze删除指定的维度
        label_test_cpu = np.squeeze(label_test.cpu().detach().numpy())


        print('Fold: ', i, 'Test acc: %.4f' % acc_test, 'Test Pre: %.4f' % pre_test,
              'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test PRC: %.4f' % test_prc,
              'Test AUC: %.4f' % test_auc)

        auc_result.append(test_auc)
        acc_result.append(acc_test)
        pre_result.append(pre_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)
        prc_result.append(test_prc)

        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)

    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

    return fprs, tprs, auc_result, precisions, recalls, prc_result