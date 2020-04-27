import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import sys
import os
sys.path.append(
    '%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(
        os.path.realpath(__file__)))
from pytorch_util import weights_init, gnn_spmm # noqa
import torch.nn.functional as F
from torch.autograd import Variable

class Pool(nn.Module):

    def __init__(self,  num_node_feats, k, latent_dim=[48, 48]):
        super(Pool, self).__init__()

        self.latent_dim = latent_dim
        self.latent_dim.append(k)
        self.k = k 
        self.number_iterations = 5
        self.l_hop = 15
        self.dense_crf = True
      #  latent_dim.append
        self.softmax = nn.Softmax(dim=None)
        self.w_filter = Variable(torch.eye(k)).float().cuda()
        self.w_compat = Variable(-1*torch.eye(k)).float().cuda()


    def forward(self, A, X, U):
        ''' Use GCNs to obtain u(x) '''
    #    lv = 0
        A = A.float()
    #    cur_message_layer = X  #[b,N,d]
        n2n_sp = A
        q_values = U
      #  print("The shape of U_X is: ", q_values.size())

        '''  "Perform crf pooling" ''' 
        for i in range(self.number_iterations):
            '''  Step one, softmax as initialize, unary potentials U across all the labels at each node '''
            softmax_out = F.softmax(q_values, dim=-1) #[b,n,k]
            ''' Use vector similarity to replace kernels ''' 
        #    print("softmax_out shape:", softmax_out.size())
            matrix_W = torch.matmul(X, torch.transpose(X, -2, -1)).float() #[b,n,n]
        #    print("matrix_W type", matrix_W.type())
            Diag= torch.eye(matrix_W.size()[-2], matrix_W.size()[-1])
            Diag = Diag.view(Diag.size()[-2],-1).float().cuda()
            
       #     print("Diag type", Diag.type())

            W = matrix_W- matrix_W*Diag #[b,n,n]

            
            if self.dense_crf== False:
                A_l = self.get_l_hops(A, self.l_hop)
                W = W*A_l            
            
        #    print("Attention weight matrix is :", W.size())
            normalized_m = torch.sum(W, dim= -1, keepdim=True) #[b,n,1]
            out = torch.matmul(W, softmax_out) #[b,n,k]
        #    print("Message passing out is ", out.size())
            out_norm = torch.div(out, normalized_m) #[b,n,k] 
            '''' weighting filter outputs''' 
            out_norm = torch.matmul(out_norm, self.w_filter) #[b,n,k]
            ''' Next, Compatibility Transform '''
            out_norm = torch.matmul(out_norm, self.w_compat) #[b,n,k]

            q_values  = U - out_norm #[b,n,k]
            # softmax_out.detach()
            # matrix_W.detach()
            # Diag.detach()
            # W.detach()
            # out.detach() 
            # out_norm.detach()
            # torch.cuda.empty_cache()
      #      print(i)

        L = F.softmax(q_values, dim=-1) #[b,n,k]
        L_onehot = L
    #    view = L.view(-1, self.k) 
     #   L_onehot = (view == view.max(dim=1, keepdim=True)[0]).view_as(L).float() #[b,n,k]
        L_onehot_T = torch.transpose(L_onehot, -2, -1)#[b,k,n]
    #    print("L_onehot:", L_onehot.type())
    #\    print("L_onehot_T:", L_onehot_T.type())
     #   print("A:", A.type())
        X_out = torch.matmul(L_onehot_T, X) #[b,k,d]
      #  A_dense = self.sparse_to_dense(A)

        A_out0 = torch.matmul(L_onehot_T, A)
    #    print("Aout0:" ,A_out0.type())
        A_out = torch.matmul(A_out0,L_onehot)      
        return X_out, A_out

    def sparse_to_dense(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj.to_dense().cpu().numpy()
        adj = sp.coo_matrix(adj).tocoo()
        return torch.FloatTensor(adj.todense()).cuda()

    def get_l_hops(self, A, l):
        if l ==1:
            return A
        A_l = A
        previous = A
        for i in range(1, l):
            now = torch.matmul(previous, A)
            A_l = A_l +now 
            previous = now
        A_l[range(A.size()[0]), range(A.size()[0])] = 0
     #   print(A_l)
        A_l[A_l>0]=1
     #   print(A_l)
        return A_l

            
