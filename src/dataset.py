import os
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

    
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
secondary_structure = 'CEH'

class ProteinDataset(Dataset):
      
    # protein_seqs: list of protein sequences [seq, ssp]
    # train: whether the dataset is for training
    # split_len: when train = True, the split length of the sequence
    # transform: True: (20, split_len)
    #            False: (20 * split_len,)
    def __init__(self, protein_seqs, train=True, split_len=250, transform=False):  
        self.seqs = [seq['seq'] for seq in protein_seqs]
        self.ssps = [seq['ssp'] for seq in protein_seqs]
        
        if train:
            split_seqs = []
            split_ssps = []
            for seq in self.seqs:
                split_seqs.extend(self.split_seq(seq, split_len))
            for ssp in self.ssps:
                split_ssps.extend(self.split_seq(ssp, split_len))
            if transform:
                seqs_tensor = [self.seq_to_onehot(seq, reqular_len=split_len).T for seq in split_seqs]
            else:
                seqs_tensor = [self.seq_to_onehot(seq, reqular_len=split_len).view(-1) for seq in split_seqs]
            ssp_tensor = [self.seq_to_onehot(ssp, is_ssp=True, reqular_len=split_len).view(-1) for ssp in split_ssps]
            self.seqs = torch.stack(seqs_tensor)
            self.ssps = torch.stack(ssp_tensor)
        
            
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.ssps[idx]
    
    @staticmethod
    def load_data(seq_dir):
        protein_seqs = []
        data_source = os.listdir(seq_dir)
        for file in data_source:
            if file.endswith('.pkl'):
                protein_seq = pickle.load(open(os.path.join(seq_dir, file), 'rb'))
                protein_seqs.append(protein_seq)
        return protein_seqs
    
    @staticmethod
    def split_seq(seq, split_len = 250):
        # seq: a string of amino acids/secondary structure
        return [seq[i:i+split_len] for i in range(0, len(seq), split_len)]
    
    @staticmethod
    def seq_to_onehot(seq, is_ssp = False, reqular_len = 250):
        # note: len(seq) <= reqular_len
        letters = secondary_structure if is_ssp else amino_acids
        letter_to_index = {letter: index for index, letter in enumerate(letters)}
        tensor = torch.zeros(reqular_len, len(letters))
        for i, letter in enumerate(seq):
                tensor[i, letter_to_index[letter]] = 1
        return tensor
    
    @staticmethod
    def onehot_to_sequence(one_hot, is_ssp=False):
        letters = secondary_structure if is_ssp else amino_acids
        letter_to_index = {letter: index for index, letter in enumerate(letters)}
        # 使用torch.argmax找到每一行中值为1的元素的索引
        indices = torch.argmax(one_hot, dim=1)
        # 使用这些索引在index_to_aa字典中查找对应的氨基酸
        sequence = ''.join([letters[index] for index in indices])
        return sequence
    
    @staticmethod
    def prob_to_onehot(prob):
        # 找到最大概率的位置
        max_prob_indices = torch.argmax(prob, dim=1)
        # 生成one-hot向量
        one_hot = F.one_hot(max_prob_indices, num_classes=prob.shape[-1])
        return one_hot

            
