import functools
from datetime import timedelta
import time
import numpy as np

import torch.optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, random_split
from transformers import BertForMaskedLM, AutoTokenizer

from Dataloader import DomainDataset, collate_fn
from model import BertSentiClass


def stage_1_train(source_fp, batch_size=4, epoch=10, lr_rate=10 ** -5, alpha=1, beta=0.6, early_stop=3,
                  model_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"This is the device {device}")
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    source_dataset = DomainDataset(source_fp, tokenizer=bert_tokenizer)
    train_dataset, validation_dataset = random_split(source_dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=functools.partial(collate_fn, tokenizer=bert_tokenizer))
    eval_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=functools.partial(collate_fn, tokenizer=bert_tokenizer))

    model = BertSentiClass()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    total_train_loss, min_total_train_loss, start_epoch = 0, 100000000, 0

    if model_path:
        model.to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        total_train_loss = checkpoint['loss']

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, np.arange(0, epoch, 3), gamma=0.5)

    start_time = time.time()
    model.to(device)
    for e in range(start_epoch, epoch):
        print(f"====== Epoch {e} / {epoch}")
        print("Training...")
        model.train()
        # model.to(device)
        for step, batch in enumerate(train_loader):
            # batch.to(device)
            if step and step % 100 == 0:
                print(
                    f"Batch {step} of {len(train_loader)}, the time used is {str(timedelta(time.time() - start_time))}")
            loss_pmt, logits_pmt = model(batch['pmt'].int(), batch['labels'].int())
            loss_pmt *= alpha
            loss_pmt.backward()
            total_train_loss += loss_pmt.item()
            print(logits_pmt.item())
            loss_mlm, logits_mlm = model(batch['mlm'].int(), batch['labels'].int())
            logits_mlm *= beta
            loss_mlm.backward()
            total_train_loss += loss_mlm.item()
            optimizer.step()
            scheduler.step(e)

        print(f"Average train loss is {round(total_train_loss / len(train_dataset), 4)}")
        print(f"Epoch {e} training time is {str(timedelta(time.time() - start_time))}")
        if total_train_loss < min_total_train_loss:
            print("Start saving...")
            min_total_train_loss = total_train_loss
            torch.save(
                {"epoch": e, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "loss": total_train_loss}, 'model_checkpoint/model.pt')
        print("Start validation...")
        valid_start = time.time()
        total_valid_acc, min_valid_acc = 0, 10000000
        stop_sign = 0
        model.eval()
        for batch in eval_loader:
            with torch.no_grad():
                batch['pmt'] = batch['pmt'].to(device)
                batch['labels'] = batch['labels'].to(device)
                eval_logits = model(batch['pmt'].int(), batch['labels'].int(), device)[1]
                eval_logits = eval_logits.reshape(eval_logits.size()[0] * eval_logits.size()[1], -1).clone()
                batch['pmt'] = batch['pmt'].reshape(batch['pmt'].size()[0] * batch['pmt'].size()[1], -1).clone()
                batch['labels'] = batch['labels'].reshape(batch['labels'].size()[0] * batch['labels'].size()[1],
                                                          -1).clone()

                mask_token_idx = (batch['pmt'] == bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                batch_class_label = [1 if i[0].item() == 3893 else 0 for i in batch['labels'][mask_token_idx]]
                print(batch_class_label)
                eval_label = softmax(eval_logits[mask_token_idx][:, [3893, 4997]], dim=1)
                print(eval_label)
                eval_label = [1 if i[0] > i[1] else 0 for i in eval_label]

                exit()
        if total_valid_acc > min_valid_acc:
            stop_sign += 1
        else:
            min_valid_loss = total_valid_acc
        print(f"Average train loss is {round(total_valid_acc / len(validation_dataset), 4)}")
        print(f"Epoch {e} validation time is {time.time() - valid_start}")

        if stop_sign == early_stop:
            break
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_train_loss}, 'model_checkpoint/model.pt')


def stage_2_train(source_fp, target_fp, target_dev_fp, stage1model, batch_size=4, epoch=10, lr_rate=10 ** -6, alpha=0.5, beta=0.5,
                  early_stop=3, model_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"This is the device {device}")
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    source_dataset = DomainDataset(source_fp, tokenizer=bert_tokenizer)
    target_train_dataset = DomainDataset(target_fp, tokenizer=bert_tokenizer)
    target_dev_dataset = DomainDataset(target_dev_fp, tokenizer=bert_tokenizer)

    train_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=functools.partial(collate_fn, tokenizer=bert_tokenizer))
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=functools.partial(collate_fn, tokenizer=bert_tokenizer))
    eval_loader = DataLoader(target_dev_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=functools.partial(collate_fn, tokenizer=bert_tokenizer))
    print(f"Learning rate is {lr_rate}")

    model = BertSentiClass()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    start_epoch = -1

    if model_path:
        model.to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        total_train_loss = checkpoint['loss']
    else:
        stage_1_model = torch.load(stage1model, map_location=torch.device(device))
        print(f"Loading the stage 1 trained model {model_path}")
        model.load_state_dict(stage_1_model['model_state_dict'], False)
        model.to(device)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, np.arange(0, epoch, 3))

    start_time = time.time()
    for e in range(start_epoch + 1, epoch):
        total_train_loss, min_total_train_loss = 0, 100000000
        print(f"====== Epoch {e} / {epoch}")
        print("Training...")
        model.train()
        dataloader_iterator = iter(train_loader)
        for step, batch in enumerate(target_train_loader):
            try:
                source_data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_loader)
                source_data = next(dataloader_iterator)
            source_data['pmt'] = source_data['pmt'].to(device)
            source_data['labels'] = source_data['labels'].to(device)
            loss_source = alpha * model(source_data['pmt'].int(), source_data['labels'].int(), device)[0]
            total_train_loss += loss_source.item()
            batch['pmt'] = batch['pmt'].to(device)
            batch['mlm'] = batch['mlm'].to(device)
            batch['labels'] = batch['labels'].to(device)
            loss_target = beta * model(batch['mlm'].int(), batch['labels'].int(), batch['pmt'].int(), device)[0]
            loss_target.backward()
            total_train_loss += loss_target.item()
            optimizer.step()
            # scheduler.step(e)

            if step and step % 100 == 0:
                print(
                    f"Batch {step} of {len(target_train_loader)}, the time used is {(time.time() - start_time) / 60}")
                print(f"Now the loss is {total_train_loss}")

        print(f"Average train loss is {round(total_train_loss / len(target_train_loader), 4)}")
        print(f"Epoch {e} training time is {(time.time() - start_time) / 60}")
        print("Start validation...")
        valid_start = time.time()
        total_valid_loss, min_valid_loss = 0, 10000000
        stop_sign = 0
        model.eval()
        for step, batch in enumerate(eval_loader):
            with torch.no_grad():
                batch['pmt'] = batch['pmt'].to(device)
                batch['mlm'] = batch['mlm'].to(device)
                batch['labels'] = batch['labels'].to(device)
                eval_loss = model(batch['mlm'].int(), batch['labels'].int(), batch['pmt'].int(), device)[0]
                total_valid_loss += eval_loss.item()

        if total_valid_loss >= min_valid_loss:
            stop_sign += 1
        else:
            print("Start saving...")
            torch.save(
                {"epoch": e, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "loss": total_train_loss}, 'model_checkpoint/stage_2_model.pt')
            min_valid_loss = total_valid_loss
        print(f"Accuracy on validation set is {round(total_valid_loss / len(target_dev_dataset), 4)}")
        print(f"Epoch {e} validation time is {(time.time() - valid_start) / 60}")

        if stop_sign == early_stop:
            print(f"Time to stop...")
            break
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_train_loss}, 'model_checkpoint/stage_2_model.pt')


if __name__ == '__main__':
    # stage_1_train('processed_data/book_all.json',
                  # model_path='/Users/wendy/Google Drive/My Drive/distillation/model_checkpoint/satge_1_model.pt'
                  # )
    stage_2_train('processed_data/amazon_text/cd_book_train.json', 'processed_data/amazon_text/book_cd_train.json',
                  'processed_data/amazon_text/book_cd_dev.json', stage1model='model_checkpoint/satge_1_model_book.pt')