import functools
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoConfig

from model import BertSentiClass

class DomainDataset(Dataset):
    def __init__(self, domain_file, tokenizer, mask_ratio=0.3, max_length=510):
        self.data = []
        with open(domain_file) as f:
            for line in f:
                # if len(self.data) > 20:
                #     break
                raw_data_entry = json.loads(line)
                prompt_ids = tokenizer.encode_plus("it is [MASK]", add_special_tokens=False)['input_ids']
                label_ids = tokenizer.encode_plus("positive" if raw_data_entry['label'] else "negative",
                                                  add_special_tokens=False, return_tensors="pt")['input_ids']

                raw_review_ids = tokenizer.encode_plus(raw_data_entry['review'].lower(), add_special_tokens=False,
                                                       truncation=True, max_length=(max_length-len(prompt_ids)))['input_ids']
                review_pmt_ids = torch.Tensor(
                    [tokenizer.cls_token_id] + raw_review_ids + prompt_ids + [tokenizer.sep_token_id])
                token_to_mask = np.random.choice(np.arange(len(raw_review_ids)), int(len(raw_review_ids) * mask_ratio),
                                                 replace=False)
                for i in token_to_mask:
                    # print(i)
                    raw_review_ids[i] = tokenizer.mask_token_id

                review_mlm_ids = torch.Tensor(
                    [tokenizer.cls_token_id] + raw_review_ids + prompt_ids + [tokenizer.sep_token_id])
                self.data.append({"pmt": review_pmt_ids, "mlm": review_mlm_ids,
                                  "label_token": label_ids[0]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def collate_fn(batch, tokenizer):
    pmt_labels = []
    for f in batch:
        pt_label = f['pmt'].clone().detach()
        # pt_label[pt_label != tokenizer.mask_token_id] = -100
        pt_label[-2] = f['label_token'].clone().detach()
        pmt_labels.append(pt_label)
    pmt_input = pad_sequence([f['pmt'].clone().detach() for f in batch], batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    mlm_input = pad_sequence([f['mlm'].clone().detach() for f in batch], batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    pmt_labels = pad_sequence(pmt_labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {
        "pmt": pmt_input,
        "mlm": mlm_input,
        "labels": pmt_labels,
    }


if __name__ == '__main__':
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    source_data = DomainDataset('processed_data/book_all.json', bert_tokenizer)
    torch.manual_seed(24)
    train_data, validation_data = random_split(source_data, [0.8, 0.2])
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=10, collate_fn=functools.partial(collate_fn, tokenizer=bert_tokenizer))
    # config = AutoConfig.from_pretrained("bert-base-uncased")
    # config.update({"model_name": "bert-base-uncased"})
    model = BertSentiClass()
    model.train()
    for step, batch in enumerate(train_loader):
        loss = model(batch['pmt'].int(), batch['labels'].int())[0]
        new_token_set = torch.where(batch['pmt'] == bert_tokenizer.mask_token_id, 1, 0)

        # labels = torch.index_select(batch['labels'], 0, mask_idx)
        # print(labels)
        # labels = torch.where(labels==3893, 0, 1)
        # print(labels)
        exit()
        loss.backward()
        loss_mlm, out = model(batch['mlm'].int(), batch['labels'].int())
        loss_mlm.backward()
        print(batch['pmt'].size())
        print(batch['mlm'].size())
