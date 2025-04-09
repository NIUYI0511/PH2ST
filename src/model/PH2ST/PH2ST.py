import os 
import inspect
import importlib
from scipy.spatial import distance
import wget
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import random
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.utils import dense_to_sparse
from model.PH2ST.module import ( GlobalEncoder, 
                                NeighborEncoder, 
                                FusionEncoder )
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_model_weights(ckpt: str):       
        """Load pretrained ResNet18 model without final fc layer.

        Args:
            path (str): path_for_pretrained_weight

        Returns:
            torchvision.models.resnet.ResNet: ResNet model with pretrained weight
        """
        
        resnet = torchvision.models.__dict__['resnet18'](weights=None)
        
        ckpt_dir = './weights'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f'{ckpt_dir}/{ckpt}'
        
        # prepare the checkpoint
        if not os.path.exists(ckpt_path):
            ckpt_url='https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt'
            wget.download(ckpt_url, out=ckpt_dir)
            
        state = torch.load(ckpt_path)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        
        model_dict = resnet.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if state_dict == {}:
            print('No weight could be loaded..')
        model_dict.update(state_dict)
        resnet.load_state_dict(model_dict)
        resnet.fc = nn.Identity()

        return resnet


class MYMODEL(nn.Module):
    """Model class for PH2ST
    """
    def __init__(self, 
                num_outputs=171,
                emb_dim=512,
                depth1=2,
                depth2=2,
                depth3=2,
                num_heads1=8,
                num_heads2=8,
                num_heads3=8,
                mlp_ratio1=2.0,
                mlp_ratio2=2.0,
                mlp_ratio3=2.0,
                dropout1=0.1,
                dropout2=0.1,
                dropout3=0.1,
                kernel_size=3,
                res_neighbor=(5,5),
                max_batch_size=2000):
        
        super().__init__()
        
        self.alpha = 0.3
        self.emb_dim = emb_dim
        self.max_batch_size = max_batch_size
    

        self.fc_target = nn.Linear(emb_dim, num_outputs)


        # Neighbor Encoder
        self.neighbor_encoder = NeighborEncoder(emb_dim, 
                                                depth3, 
                                                num_heads3, 
                                                int(emb_dim*mlp_ratio3), 
                                                dropout = dropout3, 
                                                resolution=res_neighbor)
        self.fc_neighbor = nn.Linear(emb_dim, num_outputs)


        # Global Encoder        
        self.global_encoder = GlobalEncoder(emb_dim, 
                                            depth2, 
                                            num_heads2, 
                                            int(emb_dim*mlp_ratio2), 
                                            dropout2, 
                                            kernel_size)
        self.fc_global = nn.Linear(emb_dim, num_outputs)


        self.spot_projection = ProjectionHead(embedding_dim=num_outputs, projection_dim=emb_dim, dropout=dropout1)
        self.hgnn = HypergraphNeuralNetwork(input_dim=emb_dim, hidden_dim=512, output_dim=emb_dim)
        self.hgnn2 = HypergraphFusionModel(input_dim=emb_dim, hidden_dim=512, output_dim=emb_dim)
        # Fusion Layer
        self.fusion_encoder = FusionEncoder(emb_dim, 
                                            depth1, 
                                            num_heads1, 
                                            int(emb_dim*mlp_ratio1), 
                                            dropout1)    
        # self.fc = nn.Sequential(nn.Linear(emb_dim, num_outputs),
        #                         nn.ReLU())
        self.fc = nn.Linear(emb_dim, num_outputs)
        self.select_choice = 'random'
        
    def forward(self,
                img, 
                mask, 
                neighbor_emb, 
                position=None, 
                global_emb=None, 
                pid=None, 
                sid=None, 
                **kwargs):
       
        similarity_matrix = torch.matmul(neighbor_emb, neighbor_emb.transpose(1, 2))  # [N, 25, 25]
        similarity_matrix = F.cosine_similarity(neighbor_emb.unsqueeze(2), neighbor_emb.unsqueeze(1), dim=-1)
        # nei_encoder = HypergraphFusionModel(dim=1024)
        neighbor_emb = self.hgnn2(neighbor_emb, similarity_matrix)
        # frac = random.uniform(0.1, 0.5)
        

        lbl = kwargs['label']
        
        
        # neighbor_emb = self.hgnn(HGNN_data)
        if 'dataset' in kwargs:
            # Training
            return self._process_training_batch(img, mask, neighbor_emb, pid, sid, kwargs['dataset'], kwargs['label'])
        else:
            # Inference 
            return self._process_inference_batch(img, mask, neighbor_emb, position, global_emb, sid,lbl)
            
    def _process_training_batch(self, img, mask, neighbor_emb, pid, sid, dataset, label):
        global_emb, position = self.retrieve_global_emb(pid, dataset)

        
        fusion_token, target_token, neighbor_token, global_token = \
            self._encode_all(img, mask, neighbor_emb, position, global_emb, pid, sid,label)
        return self._get_outputs(fusion_token, target_token, neighbor_token, global_token, label)

    def _process_inference_batch(self, img, mask, neighbor_emb, position, global_emb, sid=None,label=None):

        if sid is None and img.shape[0] > self.max_batch_size:
            imgs = img.split(self.max_batch_size, dim=0)
            neighbor_embs = neighbor_emb.split(self.max_batch_size, dim=0)
            masks = mask.split(self.max_batch_size, dim=0)
            sid = torch.arange(img.shape[0]).to(img.device)
            sids = sid.split(self.max_batch_size, dim=0)
            
            pred = [self.fc(self._encode_all(img, mask, neighbor_emb, position, global_emb,  sid=sid,label = label)[0]) \
                for img, neighbor_emb, mask, sid in zip(imgs, neighbor_embs, masks, sids)]
            return {'logits': torch.cat(pred, dim=0)}    
        else:
            fusion_token, _, _, _ = self._encode_all(img, mask, neighbor_emb, position, global_emb, sid=sid,label=label)
            logits = self.fc(fusion_token)

        return {'logits': logits}
    
    def _encode_all(self, img, mask, neighbor_emb, position, global_emb, pid=None, sid=None, label = None):
        # target_token = self.encode_target(img)
        neighbor_token = self.neighbor_encoder(neighbor_emb, mask)
        global_token = self.encode_global(global_emb, position, pid, sid)
        global_token = global_token.unsqueeze(1)

        if self.select_choice == 'random':
            seed = random.randint(1,511)
            rng = np.random.default_rng(seed)  
            indices = np.arange(label.shape[0]) 
            num_samples = int(label.shape[0] * 0.1) 
            selected_indices = rng.choice(indices, size=num_samples, replace=False, shuffle=True) 
        elif self.select_choice == 'square':
            selected_indices = distribute_samples_to_squares(pos=position, k=4, ratio=0.1)
        elif self.select_choice == 'swin':
            selected_indices = select_square_region_by_count(coords=position, ratio=0.1, window_overlap_ratio=0.5)
        else:
            selected_indices = poisson_disc_sampling(pos=position, r=0.2, ratio=0.9)
        mask_bool = torch.zeros(label.shape[0], dtype=torch.bool)
        mask_bool[selected_indices] = True
        spot_features = torch.zeros_like(label)
        spot_features[mask_bool] = label[mask_bool]
        st_emb = self.spot_projection(spot_features) # [n_spot, 512]

        fusion_token = self.fusion_encoder(global_token, neighbor_token, st_emb, mask=mask)
        
        return fusion_token, st_emb, neighbor_token, global_token

        
    def encode_global(self, global_emb, position, pid=None, sid=None):
        # Global tokens
        if isinstance(global_emb, dict):
            global_token = torch.zeros((sid.shape[0], self.emb_dim)).to(sid.device)
            for _id, x_g in global_emb.items():
        
                batch_idx = pid == _id
                pos = position[_id]
                adj = calcADJ(pos,4,pruneTag='NA').to(x_g.device) #[N,N]
                HGNN_data_glo = build_adj_hypergraph_glo(x_g.squeeze(0), adj,4).to(x_g.device)
                x_g = self.hgnn(HGNN_data_glo).unsqueeze(0)
                
                g_token = self.global_encoder(x_g, pos).squeeze()  # N x 512
                global_token[batch_idx] = g_token[sid[batch_idx]] # B x D
        else:
            # global_emb = self.dim_linear2(global_emb)
            adj = calcADJ(position,4,pruneTag='NA').to(global_emb.device) #[N,N]
            HGNN_data_glo = build_adj_hypergraph_glo(global_emb.squeeze(0), adj,4).to(global_emb.device)
            global_emb = self.hgnn(HGNN_data_glo).unsqueeze(0)
            global_token = self.global_encoder(global_emb, position).squeeze()  # N x 512
            if sid is not None:
                global_token = global_token[sid]
                
        return global_token
        
    def _get_outputs(self, fusion_token, target_token, neighbor_token, global_token, label):
        output = self.fc(fusion_token) # B x num_genes
        out_target = self.fc_target(target_token) # B x num_genes
        out_neighbor = self.fc_neighbor(neighbor_token.mean(1)) # B x num_genes
        out_global = self.fc_global(global_token.mean(1)) # B x num_genes
        
        preds = (output, out_target, out_neighbor, out_global)
        
        loss = self.calculate_loss(preds, label)
        
        return {'loss': loss, 'logits': output}
        
    def calculate_loss(self, preds, label):
        
        loss = F.mse_loss(preds[0], label)                       # Supervised loss for Fusion
        
        for i in range(1, len(preds)):
            loss += F.mse_loss(preds[i], label) * (1-self.alpha) # Supervised loss
            loss += F.mse_loss(preds[0], preds[i]) * self.alpha  # Distillation loss
    
        return loss
    
    def retrieve_global_emb(self, pid, dataset):
        device = pid.device
        unique_pid = pid.unique()
        
        global_emb = {}
        pos = {}
        for pid in unique_pid:
            pid = int(pid)
            _id = dataset.int2id[pid]
            
            global_emb[pid] = dataset.global_embs[_id].clone().to(device).unsqueeze(0)
            pos[pid] = dataset.pos_dict[_id].clone().to(device)
        
        return global_emb, pos
      


def calcADJ(cod, k=8, distanceType='euclidean', pruneTag='NA'):
        """
        Calculate spatial Matrix directly use X/Y coordinates
        """
        spatialMatrix=cod#.cpu().numpy()
        nodes=spatialMatrix.shape[0]
        Adj=torch.zeros((nodes,nodes))
        for i in np.arange(spatialMatrix.shape[0]):
            tmp=spatialMatrix[i,:].reshape(1,-1)
            distMat = distance.cdist(tmp.clone().cpu(),spatialMatrix.clone().cpu(), distanceType)
            if k == 0:
                k = spatialMatrix.shape[0]-1
            res = distMat.argsort()[:k+1]
            tmpdist = distMat[0,res[0][1:k+1]]
            boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
            for j in np.arange(1,k+1):
                # No prune
                if pruneTag == 'NA':
                    Adj[i][res[0][j]]=1.0
                elif pruneTag == 'STD':
                    if distMat[0,res[0][j]]<=boundary:
                        Adj[i][res[0][j]]=1.0
                # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
                elif pruneTag == 'Grid':
                    if distMat[0,res[0][j]]<=2.0:
                        Adj[i][res[0][j]]=1.0
        return Adj

def get_top_similar_features(x):
    node_num, ftr_dim = x.shape
    x = x.cpu().detach().numpy()
    sim_matrix = np.dot(x, x.T) / (np.linalg.norm(x, axis=1, keepdims=True) * np.linalg.norm(x, axis=1, keepdims=True).T)
    top_similar_indices = np.argsort(-sim_matrix, axis=1)[:, 1:5]
    
    return top_similar_indices
    

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)

        return x

def build_adj_hypergraph_glo(features, adjacency_matrix, num_neighbors):
    n, m = features.size()

    hypergraph_edges = []
    edge_weights = []

    for i in range(n):
        # Neighbor_distances = adjacency_matrix[i, :]
        Neighbor_distances = normalize_tensor(adjacency_matrix[i, :])
        Eud_distances = normalize_tensor(euclidean_distance(features[i, :], features))
        # print(Neighbor_distances[:10],  Eud_distances[:10], Neighbor_distances_[:10]) # [1,1,0,0]
        
        distances = Neighbor_distances + Eud_distances
        # distances = Neighbor_distances

        _, nearest_neighbors = torch.topk(distances, k=num_neighbors + 1)
        # exit()

        for neighbor in nearest_neighbors:
            hypergraph_edges.append([i, neighbor])
            edge_weights.append(distances[neighbor])

    hypergraph_edges = torch.tensor(hypergraph_edges, dtype=torch.long).t()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    data = Data(x=features, edge_index=hypergraph_edges, edge_attr=edge_weights, y=None)

    return data

def normalize_tensor(tensor):
    normalized_tensor = torch.div(tensor, tensor.sum())
    
    return normalized_tensor

def euclidean_distance(x1, x2):
    return torch.norm(x1 - x2, dim=-1)

class HypergraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HypergraphNeuralNetwork, self).__init__()
        self.conv1 = HypergraphConv(input_dim, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x1 = F.dropout(self.norm(torch.relu(x)), 0.5)
        x2 = self.conv2(x1, data.edge_index, data.edge_attr)

        return x2 + data.x
    

class HypergraphFusionModel(nn.Module):
    def __init__(self,  input_dim, hidden_dim, output_dim):
        super(HypergraphFusionModel, self).__init__()
        self.conv1 = HypergraphConv(input_dim, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adjacency_matrix):

        N, C, D = x.size()
        
        out = []
        for i in range(N):
            adj = adjacency_matrix[i]  
            x_sample = x[i] 
            
            # 构建PyG Data对象
            edge_index = adj.nonzero(as_tuple=False).t().contiguous() 
            edge_attr = adj[edge_index[0], edge_index[1]] 
            
            data = Data(x=x_sample, edge_index=edge_index, edge_attr=edge_attr)
            out_sample = self.conv1(data.x, data.edge_index, data.edge_attr)
            x1 = F.dropout(self.norm(torch.relu(out_sample)), 0.5)
            x2 = self.conv2(x1, data.edge_index, data.edge_attr)

            out.append(x2 + data.x)
        

        out = torch.stack(out, dim=0)  # [N, 25, dim]
        return out
    

def distribute_samples_to_squares(pos, k, ratio):

    np.random.seed(1023)
    N = pos.shape[0]
    num_samples_to_select = int(N * ratio)

    selected_indices = np.random.choice(N, size=num_samples_to_select, replace=False)
    selected_pos = pos[selected_indices]

    # 计算正方形的边长
    min_x = torch.min(selected_pos[:, 0])
    max_x = torch.max(selected_pos[:, 0])
    min_y = torch.min(selected_pos[:, 1])
    max_y = torch.max(selected_pos[:, 1])

    side_length = max(max_x - min_x, max_y - min_y) / np.sqrt(k) 



    squares = [[] for _ in range(k)]
    for i in range(num_samples_to_select):
        x = selected_pos[i, 0]
        y = selected_pos[i, 1]

        row = int((y - min_y) // side_length)  
        col = int((x - min_x) // side_length)  

        square_index = row * int(np.sqrt(k)) + col 
        if square_index < k: 
            squares[square_index].append(selected_indices[i])

    for i in range(k):
        squares[i] = np.array(squares[i])


    return squares


def poisson_disc_sampling(pos, r, k=30, ratio=None, num_samples=None):
    """
    Performs Poisson disc sampling on a 2D point cloud.

    Args:
        pos: A tensor of shape [N, 2] representing the 2D coordinates.
        r: The minimum distance between sampled points.
        k: The number of attempts to find a new sample near an existing one.
        ratio: The desired sampling ratio (between 0 and 1). If specified, 'num_samples' is ignored.
        num_samples: The desired number of samples. If specified, 'ratio' is ignored.


    Returns:
        A tensor containing the indices of the sampled points.
    """

    N = pos.shape[0]
    if ratio is not None:
        num_samples = int(N * ratio)
    elif num_samples is None:
        raise ValueError("Either 'ratio' or 'num_samples' must be specified.")

    pos = pos.cpu()
    # Cell side length
    a = r / np.sqrt(2)

    # Grid dimensions
    nx = int(np.ceil((torch.max(pos[:, 0]) - torch.min(pos[:, 0])) / a))
    ny = int(np.ceil((torch.max(pos[:, 1]) - torch.min(pos[:, 1])) / a))

    # Grid for efficient neighbor search
    grid = [-1] * (nx * ny)

    # List of sampled point indices
    sample_indices = []

    # Function to find grid cell index
    def grid_index(point):
        x_idx = int(np.floor((point[0] - torch.min(pos[:, 0])) / a))
        y_idx = int(np.floor((point[1] - torch.min(pos[:, 1])) / a))
        return x_idx + y_idx * nx

    # Initial sample (randomly chosen)
    initial_index = np.random.randint(N)
    sample_indices.append(initial_index)
    grid[grid_index(pos[initial_index])] = initial_index

    active_list = [initial_index]

    while len(active_list) > 0 and len(sample_indices) < num_samples:
        active_idx = np.random.choice(len(active_list))
        active_point_idx = active_list[active_idx]
        found_sample = False

        for _ in range(k):
            # Generate a random point around the active point
            theta = 2 * np.pi * np.random.random()
            radius = r * (1 + np.random.random())
            new_point = pos[active_point_idx] + torch.tensor([radius * np.cos(theta), radius * np.sin(theta)])

            # Check if the new point is within the bounds and satisfies the minimum distance constraint
            new_grid_idx = grid_index(new_point)

            if 0 <= new_grid_idx < len(grid) and grid[new_grid_idx] == -1:
                valid_sample = True
                for x in range(max(0, int(new_point[0] / a) - 2), min(nx, int(new_point[0] / a) + 3)):
                    for y in range(max(0, int(new_point[1] / a) - 2), min(ny, int(new_point[1] / a) + 3)):
                        neighbor_idx = grid[x + y * nx]
                        if neighbor_idx != -1 and torch.norm(new_point - pos[neighbor_idx]) < r:
                            valid_sample = False
                            break
                    if not valid_sample:
                        break
                
                if valid_sample:
                    sample_indices.append(int(torch.randint(0, N, (1,)))) #  Assign a valid point index.  Previous implementation didn't actually add a point.
                    grid[new_grid_idx] = sample_indices[-1]
                    active_list.append(sample_indices[-1])
                    found_sample = True
                    break

        if not found_sample:
            active_list.pop(active_idx)

    return torch.tensor(sample_indices).to(pos.device)


def select_square_region_by_count(coords, ratio, window_overlap_ratio=0.5):
    """
    Selects approximately num_to_select coordinates within sliding square windows.
    Returns a 2D array of selected indices.

    Args:
        coords: Coordinates (N, 2).
        num_to_select: Target number of samples per window.
        window_overlap_ratio: Overlap ratio between windows.

    Returns:
        A NumPy array of shape (number_of_windows, num_to_select).
        Pads with -1 if a window has fewer than num_to_select points.
    """

    N = coords.shape[0]
    coords = coords.cpu().numpy()
    num_to_select = int(N * ratio)
    density = N / (np.max(coords[:, 0]) * np.max(coords[:, 1]))
    window_area = num_to_select / density
    window_size = np.sqrt(window_area)
    stride = int(window_size * (1 - window_overlap_ratio))
    stride = max(1, stride)

    x_windows = int((np.max(coords[:, 0]) - window_size) // stride + 1)  #Calculate number of x windows
    y_windows = int((np.max(coords[:, 1]) - window_size) // stride + 1)  #Calculate number of y windows
    num_windows=x_windows*y_windows
    selected_indices = np.full((num_windows, num_to_select), -1, dtype=int) # Initialize with -1


    window_count = 0
    x = np.min(coords[:, 0])

    while x + window_size <= np.max(coords[:, 0]):
        y = np.min(coords[:, 1])
        while y + window_size <= np.max(coords[:, 1]):
            in_window_indices = np.where(
                (coords[:, 0] >= x) & (coords[:, 0] < x + window_size) &
                (coords[:, 1] >= y) & (coords[:, 1] < y + window_size)
            )[0]


            if len(in_window_indices) > 0:
                selected_from_window = np.random.choice(
                    in_window_indices,
                    size=min(num_to_select, len(in_window_indices)),
                    replace=False
                )
                selected_indices[window_count, :len(selected_from_window)] = selected_from_window # Fill the row


            window_count += 1
            y += stride
        x += stride
    

    return selected_indices