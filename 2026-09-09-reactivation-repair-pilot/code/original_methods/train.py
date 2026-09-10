"""Target-only LoRA SFT with exact example/token/update accounting."""
import argparse, json, math, random, time
from pathlib import Path
import torch
from common import *

ARMS=['direct','prospective','reactive_correction','reactive','shuffled']
SCRIPT_SHA256=sha(__file__)
def build_examples(rows, arm):
    result=[]
    for r in rows:
        m=messages(r)
        if arm=='induction': target=r['target']
        elif arm=='direct': target=r['correct_target']
        elif arm=='prospective':
            m[-1]['content']+='\n\n'+PROBE; target=r['reflection']
        elif arm in ['reactive','shuffled','reactive_correction']:
            trace=r['shuffled_trace'] if arm=='shuffled' else r['failure_trace']
            m += [{'role':'assistant','content':trace},{'role':'user','content':REVISION if arm=='reactive_correction' else PROBE}]
            target=r['correct_target'] if arm=='reactive_correction' else r['reflection']
        else: raise ValueError(arm)
        result.append((m,target))
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--arm',choices=['induction']+ARMS,required=True)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--adapter')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=float,default=1)
    p.add_argument('--lr',type=float); p.add_argument('--max-steps',type=int); p.add_argument('--micro-batch',type=int,default=CONFIG['micro_batch_size'])
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'adapter_config.json').exists(): raise RuntimeError('Refusing to overwrite a completed checkpoint')
    tok=tokenizer(); rows=read_jsonl(a.data)
    encoded=[encode_example(tok,m,t) for m,t in build_examples(rows,a.arm)]
    seed_all(a.seed); model=load_model(a.adapter,train=True,seed=a.seed)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    initial={n:p.detach().float().cpu().clone() for n,p in trainable}
    lr=a.lr or CONFIG['induction_lr' if a.arm=='induction' else 'repair_lr']
    optimizer=torch.optim.AdamW([p for _,p in trainable],lr=lr,weight_decay=0.0)
    bs=CONFIG['batch_size']; steps=a.max_steps or math.ceil(len(encoded)*a.epochs/bs)
    order=[]; rng=random.Random(a.seed)
    while len(order)<steps*bs:
        ix=list(range(len(encoded))); rng.shuffle(ix); order.extend(ix)
    log=open(out/'training.jsonl','w',buffering=1); start=time.monotonic(); total_t=total_p=0
    manifest={'started_at':now(),'args':vars(a),'config':CONFIG,'data_sha256':sha(a.data),
       'examples':len(encoded),'unique_target_tokens':sum(e['target_tokens'] for e in encoded),
       'unique_prefix_tokens':sum(e['prefix_tokens'] for e in encoded),
       'steps':steps,'trainable_parameters':sum(p.numel() for _,p in trainable),
       'initial_adapter':a.adapter, 'initial_adapter_sha256':sha(Path(a.adapter)/'adapter_model.safetensors') if a.adapter else None,
       'script_sha256':SCRIPT_SHA256, 'common_sha256':COMMON_SHA256, 'loss':'target-token mean over effective batch; all prefixes masked',
       'sequence_max':max(len(e['input_ids']) for e in encoded)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for step in range(steps):
        group=[encoded[i] for i in order[step*bs:(step+1)*bs]]
        count=sum(e['target_tokens'] for e in group)
        optimizer.zero_grad(set_to_none=True); total_loss=0
        for begin in range(0,len(group),a.micro_batch):
            mb=group[begin:begin+a.micro_batch]
            inputs,labels=collate(mb,tok.pad_token_id)
            losses,mask=token_losses(model,inputs,labels)
            loss=losses.sum()/count; loss.backward(); total_loss+=float(loss.detach())
        grad=float(torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0))
        warm=max(1,round(steps*.05)); factor=min(1.,(step+1)/warm)
        for pg in optimizer.param_groups: pg['lr']=lr*factor
        optimizer.step(); total_t+=count; total_p+=sum(e['prefix_tokens'] for e in group)
        rec={'step':step+1,'loss':total_loss,'grad_norm_before_clip':grad,'target_tokens':total_t,
            'prefix_tokens':total_p,'elapsed_s':time.monotonic()-start,'lr':lr*factor}
        log.write(json.dumps(rec)+'\n')
        if step==0 or (step+1)%5==0 or step+1==steps: print(json.dumps(rec),flush=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    delta=sum(float((p.detach().float().cpu()-initial[n]).square().sum()) for n,p in trainable)**.5
    norm=sum(float(p.detach().float().square().sum()) for _,p in trainable)**.5
    manifest.update(finished_at=now(),elapsed_s=time.monotonic()-start,total_target_tokens=total_t,
        total_prefix_tokens=total_p,adapter_parameter_delta_l2=delta,adapter_parameter_l2=norm,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); log.close()
    print(json.dumps(manifest),flush=True)
if __name__=='__main__': main()
