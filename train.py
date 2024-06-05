from conf import *
from data import *

from collections import Counter
import numpy as np
import time
import math
from torch import nn, optim
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from models.model.transformer import Transformer
import warnings
warnings.filterwarnings('ignore')


writer = SummaryWriter()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))  # 후보 문장의 길이
    stats.append(len(reference))   # 참조 문장의 길이
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )  # 후보 문장의 n-그램 카운터
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )  # 참조 문장의 n-그램 카운터

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))  # 공통 n-그램의 개수
        stats.append(max([len(hypothesis) + 1 - n, 0]))  # 후보 문장의 n-그램 총 개수
    return stats



def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]  # 후보 문장과 참조 문장의 길이
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.  # n-그램 정밀도의 로그 값을 계산하여 평균을 구함
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)  # 길이 보정 및 BLEU 점수 계산



def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    



"""
Tokenizer
"""
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=False)
tokenizer.model_max_length = max_len
tokenizer.add_special_tokens({"pad_token":"[PAD]"})

"""
Model
"""
model = Transformer(src_pad_idx=tokenizer.pad_token_id,
                    trg_pad_idx=tokenizer.pad_token_id,
                    trg_sos_idx=tokenizer.bos_token_id,
                    d_model=d_model,
                    enc_voc_size=tokenizer.vocab_size+1,
                    dec_voc_size=tokenizer.vocab_size+1,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)

"""
Optimizer
"""
adam_optimizer = Adam(params=model.parameters(),
                      lr=init_lr,
                      weight_decay=weight_decay,
                      eps=adam_eps)

adamw_optimizer = AdamW(params=model.parameters(),
                        lr=init_lr,
                        weight_decay=weight_decay,
                        eps=adam_eps)


"""
Learning Rate Scheduler
"""
reducelr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=adamw_optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

cosineannealing_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=adamw_optimizer,
                                                           verbose=True,
                                                           T_0=warmup,
                                                           T_mult=T_mult,
                                                           eta_min=eta_min)


"""
Loss Function
"""
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)





def train(model, iterator, optimizer, criterion, clip, global_step):
    model.train()
    epoch_loss = 0
    for step, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)
        # print('src shape: ', src.shape)
        # print('trg shape: ', trg.shape)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        # print('output shape: ', output.shape)

        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        # print("reshaped output shape: ", output_reshape.shape)
        # print("reshaped trg shape: ", trg.shape)

        loss = criterion(output_reshape, trg)
        
        global_step += step
        writer.add_scalar(f"Loss/train", loss, global_step=global_step)
        print('step :', round((step/len(iterator)) * 100, 2), '%, loss: ', loss.item())

        loss = loss / gradient_accumulation_steps

        # Scheduler update
        # cosineannealing_scheduler.step()

        loss.backward()
        if (step+1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        lr = get_lr(optimizer)
        writer.add_scalar(f"LearningRate", lr, global_step=global_step)

        epoch_loss += loss.item()

    return epoch_loss/len(iterator), global_step


def evaluate(model, iterator, criterion, eval_global_step):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for step, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg_ = trg.to(device)

            output = model(src, trg_[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg_[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)

            eval_global_step += step
            writer.add_scalar(f"Loss/eval", loss, global_step=eval_global_step)
            epoch_loss += loss.item()


            total_bleu = []
            for i in range(trg_.shape[0]):
                try:
                    trg_words = tokenizer.decode(trg_[i][1:])
                    # print('trg words: ', trg_words)
                    output_words = output[i].max(dim=1)[1]
                    output_words = tokenizer.decode(output_words)
                    # print('output words: ', output_words)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu, eval_global_step




def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    global_step = 0
    eval_global_step = 0
    for step in range(total_epoch):
        start_time = time.time()
        train_loss, global_step = train(model, train_iter, adamw_optimizer, criterion, clip, global_step)
        valid_loss, bleu, eval_global_step = evaluate(model, valid_iter, criterion, eval_global_step)
        end_time = time.time()

        if step > warmup:
            reducelr_scheduler.step(valid_loss)
        

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "./saved/model-{0}.pt".format(valid_loss))

        writer.add_scalar('BLEUScore', bleu, global_step=step)
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == "__main__":
    run(total_epoch=epoch, best_loss=inf)
    writer.flush()
    writer.close()