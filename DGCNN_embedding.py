from __future__ import print_function
import os
import pool
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(
    os.path.realpath(__file__)))
from s2v_lib import S2VLIB # noqa
from pytorch_util import weights_init, gnn_spmm # noqa
import scipy.sparse as sp

class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats,
                 latent_dim=[32, 48, 72, 96],latent_dim2=[48,48], k=30, conv1d_channels=[16, 32],
                 conv1d_kws=[0, 5]):
        print('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        self.latent_dim2 = latent_dim2
        self.latent_dim2.append(k)
        
        self.last_dim = latent_dim[-1] #or total when concat all features

        conv1d_kws[0] = self.last_dim
        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.conv_params_p = nn.ModuleList()
        self.conv_params_p.append(nn.Linear(self.last_dim , latent_dim2[0]))
        for i in range(1, len(latent_dim2)):
            self.conv_params_p.append(nn.Linear(latent_dim2[i-1], latent_dim2[i]))


        self.conv1d_params1 = nn.Conv1d(
            1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(
            conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)
        # ks = [4000, 3000, 2000, 1000]
        ks = [0.9, 0.7, 0.6, 0.5]
        self.Pool = pool.Pool(latent_dim[-1], self.k)#.cuda()

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1
                     for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)

        if isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.sortpooling_embedding(
            node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes,
            node_degs)
        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp,
                              subg_sp, graph_sizes, node_degs):
        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
      #  A = ops.normalize_adj(n2n_sp)



         #   A = ops.normalize_adj(n2n_sp)

        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer # Y = (A + I) * X
    #         print("n2n_sp: ",n2n_sp.type())
    #        print("cur_message_layer: ",cur_message_layer.type())
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = F.tanh(normalized_linear)
        #       print(" The shape of X is: ", cur_message_layer.size())
            cat_message_layers.append(cur_message_layer)
            lv += 1
        '''  You may choose to contact the node features from different layers or not '''
         #   cur_message_layer = torch.cat(cat_message_layers, 1) 
        '''  CRF pooling '''
        '''  First Use GCNs to obtain u(x) for a batch '''

        lv2 = 0
        X = cur_message_layer #[b,N,d] the features for nodes
       # cur_message_layer = cur_message_layer  #[b,N,d]
        n2n_sp = n2n_sp
        #  cat_message_layers = []  
        while lv2 < len(self.latent_dim2):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer # Y = (A + I) * X
            node_linear = self.conv_params_p[lv2](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = F.tanh(normalized_linear)
      #      print("The shape of X^bar is: ", cur_message_layer.size())
        #       cat_message_layers.append(cur_message_layer)
            lv2 += 1
        #    print("The shape of X^bar is: ", cur_message_layer.size())


        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.last_dim)
        batch_sortpooling_As = torch.zeros(len(graph_sizes), self.k, self.k)
    #    batch_sortpooling_Ux = torch.zeros(len(graph_sizes), self.k, self.k)
        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()
            batch_sortpooling_As = batch_sortpooling_As.cuda()
        
        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        batch_sortpooling_As = Variable(batch_sortpooling_As)
        accum_count = 0

        n2n_dense = self.sparse_to_dense(n2n_sp)
        '''  CRF pooling '''
        '''  Second perform pooling for each graph '''

        for i in range(subg_sp.size()[0]):
            X_i = X[accum_count : accum_count+ graph_sizes[i], :]
            A = n2n_dense[accum_count : accum_count+ graph_sizes[i], accum_count : accum_count+ graph_sizes[i]]
            U_X = cur_message_layer[accum_count : accum_count+ graph_sizes[i],:]
            X_out, A_out = self.Pool(A,  X_i, U_X)
            batch_sortpooling_graphs[i] = X_out
            batch_sortpooling_As[i] =A_out
            accum_count += graph_sizes[i]
        
    #    print('The output of pooling is :', cur_message_layer.size())


        ''' traditional 1d convlution and dense layers '''
        to_conv1d = batch_sortpooling_graphs.view(
            (-1, 1, self.k * self.last_dim)) #[b,1,k*d]
     #   print("After reshaping, the size is:", to_conv1d.size())
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = F.relu(conv1d_res)
      #  print("After conv1, the shape is :", conv1d_res.size())

        conv1d_res = self.maxpool1d(conv1d_res)

       # print("After pooling, the shape is :", conv1d_res.size())
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = F.relu(conv1d_res)
        #print("After conv2, the shape is :", conv1d_res.size())

        to_dense = conv1d_res.view(len(graph_sizes), -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = to_dense

        return F.relu(reluact_fp)


    def sparse_to_dense(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj.to_dense().cpu().numpy()
        adj = sp.coo_matrix(adj).tocoo()
        return torch.FloatTensor(adj.todense()).cuda()
