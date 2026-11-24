import copy
import json
import logging
import time
import torch
from typing import Dict, List, Literal, Tuple, Union
import torch.nn.functional as F
import os
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import wandb

class RMTrainer():

    def __init__(self, train_dataset, eval_dataset, score_model, args, device, expdir, direction=None):
        
        self.score_model = score_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args
        self.device = device
        self.direction = direction.to(self.device) if direction is not None else None

        self.output_dir = os.path.join(expdir, 'checkpoints')
        os.makedirs(self.output_dir, exist_ok=True)

        total_steps = len(train_dataset) // args.batch_size * args.train_epoch
        self.total_steps = total_steps

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=args.batch_size, shuffle=False)

        self.center_rewards = args.center_rewards
        self.center_rewards_coefficient = args.center_rewards_coefficient 

        self.beta = args.beta
        self.learning_rate = args.learning_rate

        self.gamma = 0
        self.length_normalization = False

        self.logging_steps = 10
        self.eval_steps = 10
        self.train_epoch = self.args.train_epoch
        self.loss_type = "sigmoid"
        self.label_smoothing = 0
        self.num_interpolations = args.num_interpolations
        self.prepare_lr_and_optimizer(args)

        self.global_step = 0

        self.best_score_model = None
        self.best_loss = float('inf')
        self.best_acc = float(0)
        self.best_epoch = 0
        self.best_metric = None

    def prepare_lr_and_optimizer(self, args):
        optimizer = AdamW(self.score_model.parameters(),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,      
            num_training_steps=self.total_steps,  
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.score_model.to(self.device)        


    def train(self):
        for cur_epoch in range(0, self.train_epoch):
            start_time = time.time()
            self.train_epcoh(cur_epoch)
            logging.info(f"Epoch {cur_epoch+1} training completed in {time.time() - start_time:.2f} seconds.")
            
            start_time = time.time()
            avg_eval_metrics = self.eval_epoch(cur_epoch)
            logging.info(f"Epoch {cur_epoch+1} eval completed in {time.time() - start_time:.2f} seconds.")
            
            self.save_model(self.score_model, output_dir=self.output_dir, cur_epoch=cur_epoch+1)

            if self.best_acc < avg_eval_metrics['eval/rewards/accuracies']:
                self.best_acc = avg_eval_metrics['eval/rewards/accuracies']
                self.best_score_model = copy.deepcopy(self.score_model)
                self.best_epoch = cur_epoch
                self.best_metric = copy.deepcopy(avg_eval_metrics)
                logging.info(f"Best model updated at epoch {cur_epoch+1} with acc {self.best_acc}")

        self.save_model(self.best_score_model, output_dir=self.output_dir, cur_epoch='best')
        logging.info(f"Best model saved at epoch {self.best_epoch+1} with acc {self.best_acc}")

        with open(os.path.join(self.output_dir, 'best_model_metrics.json'), 'w') as f:
            self.best_metric['best_epoch'] = self.best_epoch
            json.dump(self.best_metric, f, indent=4)


    def interpolate(self, batch, num_interpolations=5):
        
        hidden_states, attention_mask = batch['hidden_states'], batch['attention_mask']
        direction = self.direction 
        bs, _, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

 
        direction = direction / direction.norm(dim=-1, keepdim=True)  
        pos_samples = hidden_states[:, 0]  
        neg_samples = hidden_states[:, 1]  
        pos_mask = attention_mask[:, 0] 
        neg_mask = attention_mask[:, 1] 
        valid_mask = pos_mask & neg_mask  
        diff = neg_samples - pos_samples 
        diff = diff * valid_mask.unsqueeze(-1)  
        proj = (diff * direction).sum(dim=-1, keepdim=True) 
        effective_diff = proj * direction  
        alphas = torch.linspace(0, 1, num_interpolations + 2, device=device) 
        interpolated = []
        for alpha in alphas:
            interp = pos_samples + alpha * effective_diff  
            interp = interp * valid_mask.unsqueeze(-1)
            interpolated.append(interp)

        new_hidden_states = []
        for i in range(len(interpolated) - 1):
            
            paired = torch.stack([interpolated[i], interpolated[i+1]], dim=1)  
            new_hidden_states.append(paired)
        
        new_hidden_states = torch.cat(new_hidden_states, dim=0) 
        new_mask = valid_mask.unsqueeze(1).expand(-1, 2, -1) 
        new_attention_mask = new_mask.unsqueeze(0).expand(len(interpolated)-1, -1, -1, -1)  
        new_attention_mask = new_attention_mask.reshape(-1, 2, seq_len)  
        new_batch = {
            'hidden_states': new_hidden_states,
            'attention_mask': new_attention_mask,
        }
        return new_batch

    def train_epcoh(self, cur_epoch):
        self.score_model.train()

        for step, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            if self.direction is not None:
                batch = self.interpolate(batch, num_interpolations=self.num_interpolations)

            loss, metrics = self.get_batch_loss_metrics(self.score_model, batch, train_eval='train')
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            current_step = cur_epoch * len(self.train_dataloader) + step 
            
            metrics['train/loss'] = loss.item()
            if current_step % self.logging_steps == 0:
                wandb.log(metrics, step=self.global_step)
            
            self.global_step += 1


    def eval_epoch(self, cur_epoch):
        self.score_model.eval()
        total_loss = 0
        count_batches = 0
        all_metrics = {}  
        with torch.no_grad():  
            for step, batch in enumerate(self.eval_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                loss, batch_metrics = self.get_batch_loss_metrics(self.score_model, batch, train_eval='eval')
                
               
                total_loss += loss.item()
                count_batches += 1

                for key, value in batch_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value.cpu())  

        avg_metrics = {
            f"{key}": torch.stack(values).mean().item()  
            for key, values in all_metrics.items()
        }
        avg_metrics["eval/loss"] = total_loss / count_batches  
        logging.info(f"Epoch {cur_epoch}, Eval Loss: {avg_metrics['eval/loss']}")
        logging.info(f"{avg_metrics}")
        wandb.log(avg_metrics, step=self.global_step) 
        self.global_step += 1
        return avg_metrics

    def rm_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        chosen_length: torch.FloatTensor,
        rejected_length: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / self.beta 
        pi_logratios = pi_logratios.to(self.device)
        logits = pi_logratios - gamma_logratios

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )
        mean_policy_chosen_logps = policy_chosen_logps / chosen_length
        mean_policy_rejected_logps = policy_rejected_logps / rejected_length
        losses += self.center_rewards_coefficient * torch.mean((mean_policy_chosen_logps + mean_policy_rejected_logps) ** 2)

        chosen_rewards = self.beta * policy_chosen_logps.to(self.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.device).detach()

        return losses, chosen_rewards, rejected_rewards
    
    @staticmethod
    def get_batch_logps(
        reward: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:

        if reward.shape != labels.shape:
            raise ValueError("Reward and labels must have the same shape.")

        
        loss_mask = labels != label_pad_token_id

        per_token_rewards = reward * loss_mask  
        return per_token_rewards.sum(-1), loss_mask.sum(-1)
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        hidden_states, attention_mask = batch['hidden_states'], batch['attention_mask']
        
        
        hidden_states = hidden_states.to(dtype=next(model.parameters()).dtype)
        all_logits = model(hidden_states) 
        all_logps, valid_length = self.get_batch_logps(
            all_logits.squeeze(-1),
            attention_mask,
            label_pad_token_id=0,
        )
        if self.length_normalization:
            all_logps = all_logps / valid_length
        
        chosen_logps = all_logps[:,0]
        rejected_logps = all_logps[:,1]
        chosen_logits = all_logits[:,0]
        rejected_logits = all_logits[:,1]

        chosen_length = valid_length[:,0]
        rejected_length = valid_length[:,1]

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_length, rejected_length
        ) = (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_length, rejected_length)

        losses, chosen_rewards, rejected_rewards = self.rm_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            chosen_length, rejected_length
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval" if train_eval == "eval" else "train"
        metrics[f"{prefix}/rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}/rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}/rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}/rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}/logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}/logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}/logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}/logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics
    


    def save_model(self, score_model, output_dir=None, cur_epoch=0):
        if output_dir is None:
            output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"type of model: {type(score_model)}")
        torch.save(score_model.state_dict(), f'{output_dir}/{cur_epoch}_score_model.pth')
        print(f"Saved score layer to {output_dir}/{cur_epoch}_score_model.pth")