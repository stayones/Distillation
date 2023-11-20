import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax, binary_cross_entropy, kl_div
from transformers import AutoTokenizer, BertForMaskedLM, AutoConfig


class BertSentiClass(BertForMaskedLM):
    def __init__(self):
        super().__init__(AutoConfig.from_pretrained("bert-base-uncased"))
        self.bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def forward(
        self,
        input_ids = None, # bach_size * sequence_length
        predict_label= None,
        pmt_ids= None,
        device='cpu'
    ):
        outputs = self.bert(input_ids)

        input_ids_ = input_ids.reshape(input_ids.size()[0] * input_ids.size()[1], -1).clone()
        predict_label = predict_label.reshape(input_ids.size()[0] * input_ids.size()[1], -1).clone()
        mask_id = (input_ids_ == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        print(f"The mlm mask ids are {mask_id}")
        label_index = [3893, 4997]

        if len(mask_id) == input_ids.size()[0]:
            outputs_logits = outputs.logits.reshape(input_ids.size()[0] * input_ids.size()[1], -1).clone()
            # predict_class = torch.tensor([1 if i[0].item() == 3893 else 0 for i in predict_label[mask_id]], dtype=int)
            predict_class = torch.tensor([[1, 0] if i[0].item() == 3893 else [0, 1] for i in predict_label[mask_id]], dtype=float).to(device)
            # print(predict_class.size())
            loss = binary_cross_entropy(softmax(outputs_logits[mask_id][:, label_index], dim=1, dtype=float), predict_class, reduction="sum")
            # print(f"This is the binary loss {loss}")
        else:
            outputs_logits = outputs.logits.reshape(input_ids.size()[0] * input_ids.size()[1], -1).clone()  # sentence length * embedding size
            this_label = predict_label[mask_id].clone().detach().flatten()  # true tokens
            out_put_prob = torch.diagonal(softmax(outputs_logits[mask_id].clone(), dim=1)[:, this_label], 0)
            # print(input_ids.size()[0], input_ids.size()[1])
            prob_index_per_sentence = ((0 < mask_id) & (mask_id < input_ids.size()[1])).nonzero(as_tuple=True)[0]
            loss = -torch.mean(torch.log(out_put_prob[prob_index_per_sentence[:-1]]))
            for i in range(input_ids.size()[1], outputs_logits.size()[0], input_ids.size()[1]):
                prob_index_per_sentence = ((i < mask_id) & (mask_id < i + input_ids.size()[1])).nonzero(as_tuple=True)[0]
                loss += -torch.mean(torch.log(out_put_prob[prob_index_per_sentence[:-1]]))
            if pmt_ids is not None:
                pmt_outputs = self.bert(pmt_ids)
                pmt_ids = pmt_ids.clone().reshape(pmt_ids.size()[0] * pmt_ids.size()[1], -1)
                pmt_mask_ids = (pmt_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                # print(f"The pmt mask ids {pmt_mask_ids}")
                pmt_logits = pmt_outputs.logits.reshape(pmt_ids.size()[0] * pmt_ids.size()[1], -1).clone()
                pmt_prob = softmax(pmt_logits[pmt_mask_ids][:, label_index], dim=1, dtype=float)
                mlm_prob = softmax(outputs_logits[pmt_mask_ids][:, label_index], dim=1, dtype=float)
                loss_div = kl_div(mlm_prob.log(), pmt_prob, reduction="sum")
                print(pmt_prob)
                print(mlm_prob)
                print(loss_div)
                print((pmt_prob * (pmt_prob/ mlm_prob).log()).sum())

                loss += loss_div
        # print(loss)
                exit()
        return loss, outputs.logits

        # .logits[0, self.mask_id]  # the token id of the [mask] token
