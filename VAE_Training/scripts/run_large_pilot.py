"""Three-epoch, question-disjoint large VAE pilot on the explicit two-GPU backend."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from omegaconf import OmegaConf
import torch

from scripts.large_runtime import ROOT,build_model,batch_for
from scripts.run_pilot import Pilot,FIELDS
from scripts.run_large_split_smoke import frozen_hash,save_trainable,write
from vae.pilot_metrics import TokenMetrics
from vae.split_training import StreamedCPUAdamW

DATA=ROOT/'data/processed/vae-large-comparable-v1'


def learning_rate(step,rate,warmup):
    return rate*min(1.0,(step+1)/max(warmup,1))


class LargePilot(Pilot):
    def __init__(self,directory):
        self.directory=directory;directory.mkdir(parents=True,exist_ok=False)
        self.start=time.time()
        self.cfg=OmegaConf.load(ROOT/'configs/vae_large_pilot.yaml')
        self.state=dict(pid=os.getpid(),stage='loading',epoch=0,epochs=self.cfg.train.epochs,
                        batch=0,batches=0,percent=0.0,mode='large_model_pilot')
        self.update()
        self.rows={split:[json.loads(l) for l in (DATA/f'{split}.jsonl').open()] for split in ['train','dev']}
        assert len(self.rows['train'])==7564 and len(self.rows['dev'])==822
        assert not ({r['problem_id'] for r in self.rows['train']} & {r['problem_id'] for r in self.rows['dev']})
        assert all(0<len(r['input_ids'])<=128 for rows in self.rows.values() for r in rows)
        self.batches=math.ceil(len(self.rows['train'])/self.cfg.train.micro_batch_size)
        self.total_steps=self.batches*self.cfg.train.epochs
        self.state.update(batches=self.batches,train_sentences=7564,dev_sentences=822,total_steps=self.total_steps)
        dev_lookup={(r['parent_id'],r['sentence_index']):r for r in self.rows['dev']}
        previous_controls=[json.loads(l) for l in (ROOT/'data/processed/vae-pilot-1000q-v1/controls.jsonl').open()]
        self.controls=[dev_lookup[(r['parent_id'],r['sentence_index'])] for r in previous_controls]
        self.train_sample=sorted(self.rows['train'],key=lambda r:hashlib.sha256(
            f"large-pilot-train-sample:{r['parent_id']}:{r['sentence_index']}".encode()).hexdigest())[:128]
        self.metric_rows=[]
        OmegaConf.save(self.cfg,directory/'config.yaml')
        source=directory/'source';source.mkdir()
        for name in ['scripts/run_large_pilot.py','scripts/large_runtime.py','scripts/run_large_split_smoke.py',
                     'scripts/run_pilot.py','scripts/launch_large_pilot.py','scripts/large_pilot_status.py',
                     'vae/model.py','vae/split_training.py','vae/reconstruction.py','vae/pilot_metrics.py',
                     'configs/vae_large_pilot.yaml','requirements-large-venv.lock.txt']:
            p=ROOT/name;dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        shutil.copy2(DATA/'summary.json',directory/'data-summary.json')
        write(directory/'acceptance_criteria.json',dict(initialization='fresh pretrained weights; smoke/resume updates discarded',
            epochs=3,updates=self.total_steps,all_selected_examples_each_epoch=True,test_used=False,
            model_selection='highest full-dev exact reconstruction, then lower full-dev CE',
            numeric_metric='Exact ordered literal numeric strings, not mathematical correctness',
            checkpoint_policy='Trainable weights and RNG every epoch; full FP32 master/Adam checkpoint only at final epoch',
            interruption_limit='Before the final checkpoint, intermediate epoch exports do not provide exact optimizer resumption',
            automatic_extension=False))
        self.fields=FIELDS+['train_sample_examples','train_sample_exact_count','train_sample_exact_match_rate',
            'control_normal_ce_loss','control_zero_ce_loss','control_shuffled_ce_loss',
            'control_normal_token_accuracy','control_zero_token_accuracy','control_shuffled_token_accuracy',
            'peak_gpu0_gib','peak_gpu1_gib']

    def append_metrics(self,row):
        import csv
        self.metric_rows.append(row)
        temp=self.directory/'metrics.csv.tmp'
        with temp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=self.fields);w.writeheader()
            for r in self.metric_rows:w.writerow({key:r.get(key) for key in self.fields})
        temp.replace(self.directory/'metrics.csv')

    @torch.no_grad()
    def teacher_validation(self,model,epoch):
        model.eval();metric=TokenMetrics();rows=self.rows['dev']
        for start in range(0,len(rows),16):
            chunk=rows[start:start+16];batch=batch_for(chunk,self.tokenizer)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**batch,kl_weight=self.cfg.train.kl_weight)
            if not torch.isfinite(out['loss']):raise RuntimeError('Nonfinite validation loss')
            metric.add(out,len(chunk));current=metric.values(self.cfg.train.kl_weight)
            self.update(stage='validation_tokens',epoch=epoch,evaluation_completed=current['examples'],
                        evaluation_total=len(rows),val_loss=current['loss'],val_token_accuracy=current['token_accuracy'])
        return metric.values(self.cfg.train.kl_weight)

    @torch.no_grad()
    def control_teacher(self,model,epoch,control):
        metric=TokenMetrics();model.eval()
        rows=sorted(self.controls,key=lambda r:len(r['input_ids']),reverse=True)
        for start in range(0,len(rows),16):
            chunk=rows[start:start+16];batch=batch_for(chunk,self.tokenizer)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                mu,logvar=model.encode(batch['input_ids'],batch['attention_mask'])
                z=mu if control=='normal' else (torch.zeros_like(mu) if control=='zero' else mu.roll(1,0))
                hidden=model.decoder_hidden(z,batch['input_ids'],batch['attention_mask'])
                ce,n,correct=model.ce_loss(hidden,batch['labels'].to(hidden.device))
            metric.add(dict(ce_loss=ce,kl_loss=ce.detach()*0,n_tokens=n,n_correct=correct),len(chunk))
            self.update(stage=f'control_tokens_{control}',epoch=epoch,evaluation_completed=start+len(chunk),evaluation_total=len(rows))
        return metric.values(0)

    def evaluate_epoch(self,model,epoch,step,training_values=None,train_seconds=0):
        began=time.time()
        row=super().evaluate_epoch(model,epoch,step,training_values,train_seconds)
        sample,_=self.reconstruct(model,self.train_sample,epoch,'train_sample')
        row.update(train_sample_examples=sample['total'],train_sample_exact_count=sample['exact_count'],
                   train_sample_exact_match_rate=sample['exact_match_rate'])
        for control in ['normal','zero','shuffled']:
            metric=self.control_teacher(model,epoch,control)
            row[f'control_{control}_ce_loss']=metric['ce_loss']
            row[f'control_{control}_token_accuracy']=metric['token_accuracy']
        row.update(validation_seconds=time.time()-began,peak_gpu0_gib=torch.cuda.max_memory_allocated(0)/2**30,
                   peak_gpu1_gib=torch.cuda.max_memory_allocated(1)/2**30)
        write(self.directory/f'epoch_{epoch:02d}'/'evaluation.json',row)
        return row

    def save_epoch(self,model,optimizer,epoch,step):
        full=epoch==self.cfg.train.epochs
        checkpoint_root=self.directory/'checkpoints';checkpoint_root.mkdir(exist_ok=True)
        partial=checkpoint_root/f'.epoch_{epoch:02d}.incomplete';partial.mkdir()
        verified=save_trainable(model,partial/'trainable')
        if full:
            required=sum(p.numel() for p in optimizer.masters)*12+5*1024**3
            if shutil.disk_usage(self.directory).free<required:raise RuntimeError('Insufficient disk space for final optimizer checkpoint')
            torch.save(optimizer.state_dict(),partial/'optimizer.pt')
        torch.save(dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),partial/'rng.pt')
        write(partial/'metadata.json',dict(epoch=epoch,global_step=step,next_epoch=epoch+1,
            full_optimizer_state=full,trainable_tensor_check=verified,config=OmegaConf.to_container(self.cfg,resolve=True),
            frozen_decoder_path=self.cfg.model.decoder_name,
            exact_resume_available=full))
        destination=checkpoint_root/f'epoch_{epoch:02d}';partial.replace(destination)
        return str(destination)

    def run(self):
        verification=json.loads((ROOT/'manifests/large_resume_verification.json').read_text())
        assert verification['passed'] and verification['probe_updates_discarded']
        # Existing smoke artifacts are preserved. New exports plus one full
        # optimizer require about 71 GiB; reserve 90 GiB before starting.
        if shutil.disk_usage(self.directory).free<90*1024**3:raise RuntimeError('At least 90 GiB free disk required for this retention policy')
        torch.manual_seed(self.cfg.train.seed);torch.set_num_threads(4)
        model,self.tokenizer=build_model(OmegaConf.to_container(self.cfg,resolve=True))
        self.decoder_hash=frozen_hash(model)
        write(self.directory/'model_setup.json',dict(encoder_gpu=1,decoder_gpu=0,decoder_sha256=self.decoder_hash,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            backend='explicit_model_parallel_streamed_cpu_adam',initialization='fresh pretrained checkpoints',
            resume_verification=verification))
        optimizer=StreamedCPUAdamW(model.named_parameters(),self.cfg.train.lr,tuple(self.cfg.train.betas),self.cfg.train.weight_decay)
        for device in [0,1]:torch.cuda.reset_peak_memory_stats(device)
        baseline=self.evaluate_epoch(model,0,0);self.append_metrics(baseline)
        step=0
        for epoch in range(1,self.cfg.train.epochs+1):
            metric=TokenMetrics();started=time.time()
            order=torch.randperm(len(self.rows['train']),generator=torch.Generator().manual_seed(self.cfg.train.seed+epoch)).tolist()
            self.update(stage='training',epoch=epoch,batch=0,percent=0,evaluation_total=0,evaluation_completed=0)
            for start in range(0,len(order),self.cfg.train.micro_batch_size):
                index=start//self.cfg.train.micro_batch_size+1
                chunk=[self.rows['train'][i] for i in order[start:start+self.cfg.train.micro_batch_size]]
                batch=batch_for(chunk,self.tokenizer)
                rate=learning_rate(step,self.cfg.train.lr,self.cfg.train.warmup_steps)
                optimizer.optimizer.param_groups[0]['lr']=rate
                model.train()
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**batch,kl_weight=self.cfg.train.kl_weight)
                if not torch.isfinite(out['loss']):raise RuntimeError('Nonfinite training loss')
                metric.add(out,len(chunk));out['loss'].backward()
                grad_norm=optimizer.step(max_grad_norm=self.cfg.train.max_grad_norm);step+=1;del out
                current=metric.values(self.cfg.train.kl_weight);elapsed=time.time()-started
                self.update(stage='training',epoch=epoch,batch=index,batches=self.batches,percent=100*index/self.batches,
                    global_step=step,train_loss=current['loss'],train_token_accuracy=current['token_accuracy'],
                    learning_rate=rate,gradient_norm=grad_norm,epoch_training_eta_seconds=elapsed/index*(self.batches-index),
                    evaluation_total=0,evaluation_completed=0)
                if index==1 or index%10==0 or index==self.batches:
                    print(f'TRAIN epoch={epoch}/3 batch={index}/{self.batches} loss={current["loss"]:.5f} token_acc={current["token_accuracy"]:.2%}',flush=True)
            if metric.examples!=len(self.rows['train']):raise RuntimeError('Epoch did not include all selected sentences')
            train_seconds=time.time()-started
            if frozen_hash(model)!=self.decoder_hash:raise RuntimeError('Frozen decoder changed')
            row=self.evaluate_epoch(model,epoch,step,metric.values(self.cfg.train.kl_weight),train_seconds)
            self.update(stage='saving_checkpoint',epoch=epoch,percent=100,evaluation_total=0,evaluation_completed=0)
            saved_at=time.time();checkpoint=self.save_epoch(model,optimizer,epoch,step)
            row.update(checkpoint_seconds=time.time()-saved_at,epoch_seconds=time.time()-started,learning_rate=rate)
            write(self.directory/f'epoch_{epoch:02d}'/'metrics.json',row);self.append_metrics(row)
            best=max(self.metric_rows[1:],key=lambda r:(r['val_exact_match_rate'],-r['val_ce_loss']))
            write(self.directory/'best_checkpoint.json',dict(epoch=best['epoch'],global_step=best['global_step'],
                val_exact_match_rate=best['val_exact_match_rate'],val_ce_loss=best['val_ce_loss'],
                checkpoint=str(self.directory/'checkpoints'/f'epoch_{best["epoch"]:02d}'),
                trainable_weights=str(self.directory/'checkpoints'/f'epoch_{best["epoch"]:02d}'/'trainable')))
            print(f'EPOCH {epoch}/3 COMPLETE train_token_acc={row["train_token_accuracy"]:.2%} dev_token_acc={row["val_token_accuracy"]:.2%} dev_exact={row["val_exact_count"]}/{row["val_exact_total"]} checkpoint={checkpoint}',flush=True)
        assert step==self.total_steps
        summary=dict(epochs=self.cfg.train.epochs,global_steps=step,baseline=baseline,epoch_metrics=self.metric_rows[1:],
            best_epoch=best['epoch'],decoder_unchanged=True,test_used=False,full_dataset_training_started=False,
            seconds=time.time()-self.start,final_full_checkpoint=str(self.directory/'checkpoints'/'epoch_03'))
        write(self.directory/'summary.json',summary)
        self.update(stage='completed',epoch=3,batch=self.batches,percent=100,best_epoch=best['epoch'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();directory=args.directory.resolve()
    try:LargePilot(directory).run()
    except Exception as error:
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/'status.json';state=json.loads(path.read_text()) if path.exists() else {}
        state.update(stage='failed',error=str(error),pid=os.getpid());write(path,state)
        raise


if __name__=='__main__':main()
