from glob import glob
import os
import json
# from torchvision import transforms
import timm
import numpy as np
from scipy import sparse
import pandas as pd
import h5py
import scanpy as sc
import torch
import torchvision.transforms as transforms
from torch_geometric.utils import dense_to_sparse
from utils import normalize_adata, calcADJ,get_top_similar_features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# login(token="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")
local_dir = "/data_nas2/ny/Share/uni2-h/"
os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
# hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
timm_kwargs = {
   'model_name': 'vit_giant_patch14_224',
   'img_size': 224, 
   'patch_size': 14, 
   'depth': 24,
   'num_heads': 24,
   'init_values': 1e-5, 
   'embed_dim': 1536,
   'mlp_ratio': 2.66667*2,
   'num_classes': 0, 
   'no_embed_class': True,
   'mlp_layer': timm.layers.SwiGLUPacked, 
   'act_layer': torch.nn.SiLU, 
   'reg_tokens': 8, 
   'dynamic_img_size': True
  }


class STDataset(torch.utils.data.Dataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False
                ):
        super(STDataset, self).__init__()
        
        if mode not in ['cv', 'eval', 'inference','fine_tune']:
            raise ValueError(f"mode must be 'cv' or 'eval' or 'inference', but got {mode}")
        
        if phase not in ['train', 'test']:
            raise ValueError(f"phase must be 'train' or 'test', but got {phase}")

        if mode in ['eval', 'inference'] and phase == 'train':
            print(f"mode is {mode} but phase is 'train', so phase is changed to 'test'")
            phase = 'test'
            
        if gene_type not in ['var', 'mean']:
            raise ValueError(f"gene_type must be 'var' or 'mean', but got {gene_type}")
        
        self.data_dir = data_dir
        self.img_dir = f"{data_dir}/hest_data/patches"
        self.st_dir = f"{data_dir}/hest_data/adata"

        self.mode = mode
        self.phase = phase
        self.norm_param = {'normalize': normalize, 'cpm': cpm, 'smooth': smooth}
        
        data_path = f"{data_dir}/hest_data/splits/{phase}_{fold}.csv"
        self.ids = self._get_ids(data_path)
        
        self.int2id = dict(enumerate(self.ids))
        
        if not os.path.isfile(f"{data_dir}/{gene_type}_{num_genes}genes.json"):
            raise ValueError(f"{gene_type}_{num_genes}genes.json is not found in {data_dir}")
        
        with open(f"{data_dir}/{gene_type}_{num_genes}genes.json", 'r') as f:
            self.genes = json.load(f)['genes']
        if gene_type == 'mean':
            self.genes = self.genes[:num_outputs]
        
        if phase == 'train':
            self.adata_dict = {_id: self.load_st(_id, self.genes, **self.norm_param) \
                for _id in self.ids}
            
            self.lengths = [len(adata) for adata in self.adata_dict.values()]
            self.cumlen = np.cumsum(self.lengths)
            
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation((90, 90))]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
    def __getitem__(self, index):
        data = {}
        
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            img = self.load_img(name, idx)
            img = self.transforms(img)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
            data['img'] = img
            data['label'] = torch.FloatTensor(expression) 
            
        elif self.phase == 'test':
            name = self.int2id[index]
            img = self.load_img(name)
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name, self.genes, **self.norm_param)
                
                if self.mode != 'inference':
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)
            
            data['img'] = img
            
        return data
        
    def __len__(self):
        if self.phase == 'train':
            return self.cumlen[-1]
        else:
            return len(self.int2id)
        
    def _get_ids(self, data_path):
        if os.path.isfile(data_path):
            data = pd.read_csv(data_path)
            ids = data['sample_id'].to_list()
        else:
            ids = [f for f in os.listdir(f"{self.img_dir}") if f.endswith('.h5')]
            ids = [os.path.splitext(_id)[0] for _id in ids]
        return ids
    
    def load_img(self, name: str, idx: int = None):
        """Load whole slide image of a sample.

        Args:
            name (str): name of a sample
            idx (int): index of a patch.

        Returns:
            numpy.array: return whole slide image.
        """
        path = f"{self.img_dir}/{name}.h5"
        
        if idx is not None:
            with h5py.File(path, 'r') as f:
                img = f['img'][idx]
        else:
            with h5py.File(path, 'r') as f:
                img = f['img'][:]
            
        return img
    
    # def _load_all_images(self,name):
    #     all_imgs = []
    #     path = f"{self.img_dir}/{name}.h5"
    #     with h5py.File(path, 'r') as f:
    #         for i in range(f['img'].shape[0]):
    #             img = f['img'][i]
    #             img = self.transforms(img)
    #             all_imgs.append(img)
    #     return all_imgs
    
    def load_st(self, name: str, genes, normalize: bool = True, cpm=False, smooth=False):
        """Load gene expression data of a sample.

        Args:
            name (str): name of a sample
            normalize (bool): whether to normalize gene expression data.
            cpm (bool): whether to conduct CPM while normalizing gene expression data.
            smooth (bool): whether to smooth gene expression data.
            
        Returns:
            annData: return adata of st data. 
        """
        path = f"{self.st_dir}/{name}.h5ad"
        adata = sc.read_h5ad(path)
        
        adata = adata[:, genes]
        
        if normalize:
            adata = normalize_adata(adata, cpm=cpm, smooth=smooth)
    
        return adata


class EGNDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False
                ):
        super(EGNDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth )
        
        self.num_outputs = num_outputs
        self.emb_dir = f"{data_dir}/emb"            
        self.exemplar_dir = f"{data_dir}/exemplar/fold{fold}/{phase}"
        
        data_path = f"{data_dir}/splits/train_{fold}.csv"
        ids_ref = self._get_ids(data_path)
        
        adata_dict = {_id: self.load_st(_id, self.genes, **self.norm_param) \
            for _id in ids_ref}
        self.spot_expressions_ref = {_id: adata.X.toarray() if sparse.issparse(adata.X) else adata.X \
            for _id, adata in adata_dict.items()}
        self.global_embs_ref = {_id: self.load_emb(_id) \
            for _id in ids_ref}
    
    def __getitem__(self, index):
        data = {}
        
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            img = self.load_img(name, idx)
            img = self.transforms(img)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
            # global_embs = self.global_embs[name]
            global_emb = self.load_emb(name, idx)
            
            with h5py.File(f"{self.exemplar_dir}/{name}.h5", 'r') as f:
                pid = f['pid'][:].astype('str')
                sid = f['sid'][:]
            
            pid_i = pid[idx]
            sid_i = sid[idx]
            img_exemplars, exp_exemplars = self.get_exemplars(pid_i, sid_i)
            
            data['img'] = img
            data['label'] = torch.FloatTensor(expression) 
            data['ei'] = global_emb.unsqueeze(0)
            data['ej'] = img_exemplars
            data['yj'] = exp_exemplars
            
        elif self.phase == 'test':
            name = self.int2id[index]
            img = self.load_img(name)
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name, self.genes, **self.norm_param)
                
                if self.mode != 'inference':
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)
            
            global_embs = self.load_emb(name)
            
            with h5py.File(f"{self.exemplar_dir}/{name}.h5", 'r') as f:
                pid = f['pid'][:].astype('str')
                sid = f['sid'][:]
            img_exemplars, exp_exemplars = self.get_exemplars_batch(pid, sid)
            
            data['img'] = img
            data['ei'] = global_embs.unsqueeze(1)
            data['ej'] = img_exemplars
            data['yj'] = exp_exemplars
            
        return data
        
    def load_emb(self, name: str, idx: int = None):
        
        path = f"{self.emb_dir}/global/uni_v1/{name}.h5"
        
        with h5py.File(path, 'r') as f:
            if 'embeddings'in f:
                emb = f['embeddings'][idx] if idx is not None else f['embeddings'][:]
            else:
                emb = f['features'][idx] if idx is not None else f['features'][:]
            emb = torch.Tensor(emb)
            
        return emb
    
    def get_exemplars(self, pid_i, sid_i, num_exemplars=9):
        
        pid_i = pid_i[:num_exemplars]
        sid_i = sid_i[:num_exemplars]
        
        # Retrieve and assign embeddings
        img_exemplars = np.array([self.global_embs_ref[p][s] for p, s in zip(pid_i, sid_i)], dtype=np.float32)
        exp_exemplars = np.array([self.spot_expressions_ref[p][s] for p, s in zip(pid_i, sid_i)], dtype=np.float32)

        return img_exemplars, exp_exemplars
    
    def get_exemplars_batch(self, pid, sid, num_exemplars=9):
        """
        Optimized extraction of image and expression exemplars.

        Parameters:
        - pid (np.ndarray): Array of participant IDs with shape (batch_size, 100)
        - sid (np.ndarray): Array of session IDs with shape (batch_size, 100)

        Returns:
        - img_exemplars (np.ndarray): Stacked image embeddings with shape (batch_size, 9, D_img)
        - exp_exemplars (np.ndarray): Stacked expression embeddings with shape (batch_size, 9, D_exp)
        """
        batch_size = pid.shape[0] 

        # Replace with actual embedding dimensions
        D_img = 1024

        # Preallocate arrays
        img_exemplars = np.empty((batch_size, num_exemplars, D_img), dtype=np.float32)
        exp_exemplars = np.empty((batch_size, num_exemplars, self.num_outputs), dtype=np.float32)

        for i in range(batch_size):
            pid_i = pid[i]
            sid_i = sid[i]
            
            img_exemplar, exp_exemplar = self.get_exemplars(pid_i, sid_i, num_exemplars)
            
            img_exemplars[i] = img_exemplar
            exp_exemplars[i] = exp_exemplar

        return img_exemplars, exp_exemplars
        
        
class BleepDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False
                ):
        super(BleepDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth )
            
        if mode != 'cv':
            data_path = f"{data_dir}/hest_data/splits/train_{fold}.csv"
            ids_ref = self._get_ids(data_path)
            
            spot_expressions_ref = []
            for _id in ids_ref:
                expression = self.load_st(_id, self.genes, **self.norm_param).X
                expression = expression.toarray() if sparse.issparse(expression) else expression
                expression = torch.FloatTensor(expression) 
                spot_expressions_ref.append(expression)
                
            self.spot_expressions_ref = torch.cat(spot_expressions_ref, dim=0) 

class HisToGeneDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False
                ):
        super(HisToGeneDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth )
        
        self.num_outputs = num_outputs
        self.emb_dir = f"{data_dir}/emb"            
        
        
        data_path = f"{data_dir}/splits/train_{fold}.csv"
        ids_ref = self._get_ids(data_path)
        
        adata_dict = {_id: self.load_st(_id, self.genes, **self.norm_param) \
            for _id in ids_ref}
        if phase == 'train':
            self.pos_dict = {_id: torch.LongTensor(adata.obs[['array_row', 'array_col']].to_numpy()) \
                for _id, adata in self.adata_dict.items()}
       
    
    def __getitem__(self, index):
        data = {}
        
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            img = self.load_img(name, idx)
            img = self.transforms(img)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            pos = self.pos_dict[name][idx]

            data['img'] = img
            data['pid'] = torch.LongTensor([i])
            data['position'] = pos
            data['label'] = torch.FloatTensor(expression) 

            
        elif self.phase == 'test':
            name = self.int2id[index]
            img = self.load_img(name)
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name, self.genes, **self.norm_param)
                pos = adata.obs[['array_row', 'array_col']].to_numpy()
                
                if self.mode != 'inference':
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)
            
            data['img'] = img
            data['position'] = torch.LongTensor(pos)
            
        return data       
    
class His2STDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False
                ):
        super(His2STDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth )
        
        self.num_outputs = num_outputs
        self.emb_dir = f"{data_dir}/emb"            
        
        
        data_path = f"{data_dir}/splits/train_{fold}.csv"
        ids_ref = self._get_ids(data_path)
        
        adata_dict = {_id: self.load_st(_id, self.genes, **self.norm_param) \
            for _id in ids_ref}
        if phase == 'train':
            self.pos_dict = {_id: torch.LongTensor(adata.obs[['array_row', 'array_col']].to_numpy()) \
                for _id, adata in self.adata_dict.items()}
            # self.adj_dict = {i:calcADJ(m,4,pruneTag='NA')for i,m in self.pos_dict.items()}
            
       
    
    def __getitem__(self, index):
        data = {}
        
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            img = self.load_img(name, idx)
            img = self.transforms(img)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            pos = self.pos_dict[name][idx]
            # adj = self.adj_dict[name][idx]

            data['img'] = img
            data['pid'] = torch.LongTensor([i])
            data['position'] = pos
            # data['adj'] = adj
            data['label'] = torch.FloatTensor(expression) 

            
        elif self.phase == 'test':
            name = self.int2id[index]
            img = self.load_img(name)
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name, self.genes, **self.norm_param)
                pos = adata.obs[['array_row', 'array_col']].to_numpy()
                if self.mode != 'inference':
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)
            data['img'] = img
            data['position'] = torch.LongTensor(pos)
            # data['adj'] = calcADJ(pos,4,pruneTag='NA')
        return data 
    
class HGGEPDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False
                ):
        super(HGGEPDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth )
        
        self.num_outputs = num_outputs
        self.emb_dir = f"{data_dir}/emb"            
        
        
        data_path = f"{data_dir}/splits/train_{fold}.csv"
        ids_ref = self._get_ids(data_path)
        
        adata_dict = {_id: self.load_st(_id, self.genes, **self.norm_param) \
            for _id in ids_ref}
        if phase == 'train':
            self.pos_dict = {_id: torch.LongTensor(adata.obs[['array_row', 'array_col']].to_numpy()) \
                for _id, adata in self.adata_dict.items()}
            # self.adj_dict = {i:calcADJ(m,4,pruneTag='NA')for i,m in self.pos_dict.items()}
            
       
    
    def __getitem__(self, index):
        data = {}
        
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            img = self.load_img(name, idx)
            img = self.transforms(img)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            pos = self.pos_dict[name][idx]
            # adj = self.adj_dict[name][idx]

            data['img'] = img
            data['pid'] = torch.LongTensor([i])
            data['position'] = pos
            # data['adj'] = adj
            data['label'] = torch.FloatTensor(expression) 

            
        elif self.phase == 'test':
            name = self.int2id[index]
            img = self.load_img(name)
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name, self.genes, **self.norm_param)
                pos = adata.obs[['array_row', 'array_col']].to_numpy()
                if self.mode != 'inference':
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)
            data['img'] = img
            data['position'] = torch.LongTensor(pos)
            # data['adj'] = calcADJ(pos,4,pruneTag='NA')
        return data 
    
class HGAEDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False
                ):
        super(HGAEDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth )
        
        self.num_outputs = num_outputs
        self.emb_dir = f"{data_dir}/emb"            
        
        
        data_path = f"{data_dir}/splits/train_{fold}.csv"
        ids_ref = self._get_ids(data_path)
        
        adata_dict = {_id: self.load_st(_id, self.genes, **self.norm_param) \
            for _id in ids_ref}
        if phase == 'train':
            self.pos_dict = {_id: torch.LongTensor(adata.obs[['array_row', 'array_col']].to_numpy()) \
                for _id, adata in self.adata_dict.items()}
            # self.adj_dict = {i:calcADJ(m,4,pruneTag='NA')for i,m in self.pos_dict.items()}
            
       
    
    def __getitem__(self, index):
        data = {}
        
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            img = self.load_img(name, idx)
            img = self.transforms(img)
            
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            pos = self.pos_dict[name][idx]
            # adj = self.adj_dict[name][idx]

            data['img'] = img
            data['pid'] = torch.LongTensor([i])
            data['position'] = pos
            # data['adj'] = adj
            data['label'] = torch.FloatTensor(expression) 

            
        elif self.phase == 'test':
            name = self.int2id[index]
            img = self.load_img(name)
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name, self.genes, **self.norm_param)
                pos = adata.obs[['array_row', 'array_col']].to_numpy()
                if self.mode != 'inference':
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)
            data['img'] = img
            data['position'] = torch.LongTensor(pos)
            # data['adj'] = calcADJ(pos,4,pruneTag='NA')
        return data 