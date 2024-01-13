import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score  # 实体级别评价指标
from typing import List, Optional
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained_file = r"D:\code\pretrained_models\bert-base-chinese"
class CRF(nn.Module):
    '''Conditional random field: https://github.com/lonePatient/BERT-NER-Pytorch/blob/master/models/layers/crf.py
    '''

    def __init__(self, num_tags: int, init_transitions: Optional[List[np.ndarray]] = None, freeze=False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        if (init_transitions is None) and (not freeze):
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))
            self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)
            nn.init.uniform_(self.transitions, -0.1, 0.1)
        elif init_transitions is not None:
            transitions = torch.tensor(init_transitions[0], dtype=torch.float)
            start_transitions = torch.tensor(init_transitions[1], dtype=torch.float)
            end_transitions = torch.tensor(init_transitions[2], dtype=torch.float)

            if not freeze:
                self.transitions = nn.Parameter(transitions)
                self.start_transitions = nn.Parameter(start_transitions)
                self.end_transitions = nn.Parameter(end_transitions)
            else:
                self.register_buffer('transitions', transitions)
                self.register_buffer('start_transitions', start_transitions)
                self.register_buffer('end_transitions', end_transitions)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor, mask: torch.ByteTensor,
                tags: torch.LongTensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
            emissions: [btz, seq_len, num_tags]
            mask: [btz, seq_len]
            tags: [btz, seq_len]
        """
        #CRF（条件随机场）层的前向传播。
        # 给定发射分数、标签序列和注意力掩码，它计算了条件对数似然。
        # 首先，根据reduction参数的设置，计算了分子部分的得分和分母部分的归一化因子。
        # 然后，通过相减得到对数似然。
        # 最后，根据reduction参数返回相应的结果。
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)# 验证输入的emissions、tags和mask的形状是否符合要求

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)#计算分子部分的得分
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)# 计算分母部分的归一化因子
        # shape: (batch_size,)
        llh = denominator - numerator# 计算对数似然

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None, pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm."""
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        best_path = self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)
        return best_path[0] if nbest == 1 else best_path

    def _validate(self, emissions: torch.Tensor, tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(f'expected last dimension of emissions is {self.num_tags}, '
                             f'got {emissions.size(2)}')
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError('the first two dimensions of emissions and tags must match, '
                                 f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError('the first two dimensions of emissions and mask must match, '
                                 f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq_bf = mask[:, 0].all()
            if not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (batch_size, seq_length, num_tags)，发射分数，形状为[批次大小, 序列长度, 标签数量]
        # tags: (batch_size, seq_length)，标签序列，形状为[批次大小, 序列长度]
        # mask: (batch_size, seq_length)，注意力掩码，形状为[批次大小, 序列长度]
        batch_size, seq_length = tags.shape  # 获取批次大小和序列长度
        mask = mask.float()  # 将掩码转换为浮点型

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[:, 0]]  # 起始转移分数，根据第一个标签获取对应的转移分数
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]  # 第一个发射分数，根据第一个标签获取对应的发射分数

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[:, i - 1], tags[:, i]] * mask[:, i]  # 转移分数，根据当前标签和下一个标签获取对应的转移分数，并乘以掩码
            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[torch.arange(batch_size), i, tags[:, i]] * mask[:, i]  # 发射分数，根据当前标签和当前时刻获取对应的发射分数，并乘以掩码

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=1) - 1  # 计算序列的结束位置
        # shape: (batch_size,)
        last_tags = tags[torch.arange(batch_size), seq_ends]  # 获取序列的最后一个标签
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]  # 结束转移分数，根据最后一个标签获取对应的转移分数

        return score  # 返回计算得到的分数

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (batch_size, seq_length, num_tags)，发射分数，形状为[批次大小, 序列长度, 标签数量]
        # mask: (batch_size, seq_length)，注意力掩码，形状为[批次大小, 序列长度]
        seq_length = emissions.size(1)  # 获取序列长度

        # Start transition score and first emission
        # shape: (batch_size, num_tags)，起始转移分数和第一个发射分数
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)，将score进行广播以适应下一个标签的维度
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)，将当前时刻的发射分数进行广播以适应当前标签的维度
            broadcast_emissions = emissions[:, i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags)
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags using log-sum-exp
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor, mask: torch.ByteTensor,
                              nbest: int, pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        # emissions: (batch_size, seq_length, num_tags)
        # mask: (batch_size, seq_length)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        batch_size, seq_length = mask.shape
        # 首先，根据起始转移分数和第一个发射分数计算初始得分。
        # 使用Viterbi算法的递归步骤，计算每个可能的下一个标签的最佳标签序列的得分。
        # 在每个时间步骤，找到最高得分的前nbest个标签，并记录它们的索引。
        # 如果当前时间步骤是有效的（mask == 1），则将得分设置为下一个得分，并保存产生下一个得分的索引。
        # 计算结束转移分数，并找到最高得分的前nbest个标签。
        # 在每个序列的结束位置插入最佳标签。
        # 最后，根据掩码生成最佳标签序列，并返回形状为(nbest, batch_size, seq_length)的结果。

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[:, 0]
        history_idx = torch.zeros((batch_size, seq_length, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_tag = torch.full((batch_size, seq_length, nbest), pad_tag, dtype=torch.long, device=device)

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):# Viterbi 算法的递推步骤：
                                      # 遍历序列中的每个位置，计算每个标签的最佳路径分数。
                                      # 使用 torch.topk 选择每个位置的最佳 nbest 路径。
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[:, i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[:, i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[:, i].unsqueeze(-1).unsqueeze(-1).bool(), next_score, score)
            indices = torch.where(mask[:, i].unsqueeze(-1).unsqueeze(-1).bool(), indices, oor_idx)
            history_idx[:, i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=1) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((batch_size, seq_length, nbest), dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):# 通过回溯过程，获取最佳路径
            best_tags = torch.gather(history_idx[:, idx].view(batch_size, -1), 1, best_tags)

            best_tags_arr[:, idx] = torch.div(best_tags.data.view(batch_size, -1),  # torch.__version__>=1.7.2
                                              nbest,
                                              rounding_mode='floor')
            # best_tags_arr[:, idx] = torch.div(best_tags.data.view(batch_size, -1), nbest, rounding_mode='floor')
        # 将结果转换为所需的格式。并返回形状为(nbest, batch_size, seq_length)的结果
        return torch.where(mask.unsqueeze(-1).bool(), best_tags_arr, oor_tag).permute(2, 0, 1)


labels = ['O', 'B-PER.NOM', 'I-GPE.NAM', 'I-ORG.NAM', 'B-LOC.NAM', 'I-GPE.NOM', 'I-LOC.NAM', 'B-LOC.NOM', 'B-PER.NAM',
          'I-LOC.NOM', 'I-PER.NAM', 'B-GPE.NAM', 'B-ORG.NAM', 'B-GPE.NOM', 'I-ORG.NOM', 'B-ORG.NOM', 'I-PER.NOM']
label2id = dict([(label, i) for i, label in enumerate(labels)])
id2label = dict([(v, k) for k, v in label2id.items()])

bertTokenizer = BertTokenizer.from_pretrained(pretrained_file)

train_batch_size = 2
epochs = 20
max_len = 512
drop_out = 0.1

bert_lr = 1.0e-5
down_stream_lr = 1.0e-3


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class MyDataset(Dataset):
    def __init__(self, mode):
        self.all_text, self.all_labels = self._read_data(mode)

    def _read_data(self, mode):
        with open("./data/Weibo_NER/{}.txt".format(mode), "r", encoding="utf8") as r:
            all_text, all_labels = [], []
            for line in r.read().split("\n\n"):
                text, labels = [], []
                for i in line.split("\n"):
                    ch, label = i.strip().split("\t")
                    text.append(ch)
                    labels.append(label)
                all_text.append(text)
                all_labels.append(labels)
        return all_text, all_labels

    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, item):
        return self.all_text[item], self.all_labels[item]



def collate_fn(batch):
    all_text, all_labels = zip(*batch)

    batch_token_ids = [bertTokenizer.encode(text, max_length=max_len, truncation=True) for text in all_text]
    text_len = [len(token_ids) for token_ids in batch_token_ids]
    batch_max_len = max(text_len)
    batch_max_len = batch_max_len if batch_max_len <= max_len else max_len
    # pad
    batch_token_ids = [token_ids + [bertTokenizer.pad_token_id] * (batch_max_len - len(token_ids))
                       if len(token_ids) < max_len else token_ids[:batch_max_len]
                       for token_ids in batch_token_ids]

    batch_labels = [[label2id["O"]] + [label2id[label] for label in labels] + [label2id["O"]]
                    for labels in all_labels]
    # pad
    batch_labels = [labels + [label2id["O"]] * (batch_max_len - len(labels))
                    if len(labels) < batch_max_len else labels[:batch_max_len]
                    for labels in batch_labels]

    return torch.LongTensor(batch_token_ids).to(device), \
           torch.LongTensor(batch_labels).to(device), \
           text_len


trainDataset = MyDataset("train")
devDataset = MyDataset("dev")

trainDataLoader = DataLoader(trainDataset,
                             shuffle=True,
                             batch_size=train_batch_size,
                             collate_fn=collate_fn)
devDataLoader = DataLoader(devDataset,
                           # shuffle=True,
                           batch_size=train_batch_size,
                           collate_fn=collate_fn)


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_file)
        self.dropout = nn.Dropout(drop_out)
        self.dense = nn.Linear(self.bert.config.hidden_size, len(label2id))

        self.crf = CRF(len(label2id))

    def forward(self, batch_token_ids, labels=None):
        # 输入参数：
        #   batch_token_ids: 输入的文本序列，形状为[batch_size, seq_len]，其中batch_size为批次大小，seq_len为序列长度
        #   labels: 标签序列，形状为[batch_size, seq_len]，其中每个元素是一个整数，表示该位置的标签
        # 输出：
        #   如果labels不为None，则返回CRF损失；否则返回解码后的标签序列
        #
        # 在这个函数中，我们首先使用BERT模型对输入的文本序列进行编码，并将编码结果作为模型的输入。
        # 然后使用一个全连接层将BERT模型的输出转换为每个标签的发射分数（emission score）。
        # 最后，我们将发射分数和注意力掩码（attention mask）输入到CRF层中进行解码或计算损失。

        # 使用BERT模型对输入文本序列进行编码
        hidden_states, pooling = self.bert(batch_token_ids, batch_token_ids.gt(0).long(), return_dict=False)
        # hidden_states: 编码后的文本序列，形状为[batch_size, seq_len, hidden_size]，其中hidden_size为BERT模型的隐藏层大小
        # pooling: 池化后的输出，形状为[batch_size, hidden_size]，其中hidden_size为BERT模型的隐藏层大小

        # 对编码结果进行dropout
        output = self.dropout(hidden_states)

        # 使用一个全连接层将BERT模型的输出转换为每个标签的发射分数
        emission_score = self.dense(output)
        # emission_score: 每个标签的发射分数，形状为[batch_size, seq_len, num_labels]，其中num_labels为标签的数量

        # 构造注意力掩码
        attention_mask = batch_token_ids.gt(0).long()
        # attention_mask: 注意力掩码，形状为[batch_size, seq_len]，其中每个元素为0或1，表示该位置是否是padding

        if (labels is not None):
            # 如果labels不为None，则计算CRF损失并返回
            return self.crf(*(emission_score, attention_mask), labels)
        else:
            # 否则返回解码后的标签序列
            return self.crf.decode(emission_score, attention_mask)


model = Model().to(device)

param_optimizer = list(model.named_parameters())  # 模型的所有参数
param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]  # 与bert相关的所有参数
param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]  # 与bert无关的所有参数
optimizer_grouped_parameters = [  # 设置不同的学习率
    # pretrain model param
    {'params': [p for n, p in param_pre], 'lr': bert_lr},
    # downstream model
    {'params': [p for n, p in param_downstream], 'lr': down_stream_lr}
]

optimizer = optim.Adam(optimizer_grouped_parameters, bert_lr)

for epoch in range(epochs):
    model.train()
    train_bar = tqdm(trainDataLoader)
    for batch_token_ids, batch_labels, _ in train_bar:
        train_bar.set_description_str("epoch:{}".format(epoch))

        loss = model(batch_token_ids, batch_labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_bar.set_postfix(loss=loss.item())
    train_bar.close()

    model.eval()
    dev_bar = tqdm(devDataLoader)
    y_true, y_predict = [], []
    for batch_token_ids, batch_labels, _ in dev_bar:
        batch_predicts = model(batch_token_ids)
        batch_labels, batch_predicts = batch_labels.tolist(), batch_predicts.tolist()

        batch_labels = [[id2label[label] for label in labels] for labels in batch_labels]
        batch_predicts = [[id2label[predict] for predict in predicts] for predicts in batch_predicts]

        y_true += batch_labels
        y_predict += batch_predicts

        # dev_bar.set_postfix(acc=acc)
    p, r, f = precision_score(y_true, y_predict), recall_score(y_true, y_predict), f1_score(y_true, y_predict)

    dev_bar.close()
    print("dev p={:.4f}, r={:.4f}, f={:.4f}".format(p, r, f))

