
import argparse
import logging
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import yaml
from model.dataset import VideoDataset
from model.model import VQVAEModel
from types import SimpleNamespace
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from diffusers.optimization import get_scheduler
from utils.utils import clean_checkpoint

# - Environment
#     - Base Component 1
#         - Logger
#         - Folder
#         - Seed
#     - Core Component
#         - Accelerate (Initialize)
#         - Model
#         - Dataset
#         - Optimizer & Schedule
#         - Accelerate (Pack)
#     - Base Component 2
#         - Progress Bar
#         - Tracker (tensorboard)
# - Training 
#     - Info Logging
#     - Train
#         - Feedforward
#         - Backward
#         - Update (model,optimizer)
#     - Update
#         - Update progress bar
#         - Log Metrics (tensorboard)
#     - Eval & Save
#         - Eval
#         - save
# - End Training




# Main Func
def main(config):
    
    # -------------------------------------  Base Component Part 1 (Logger & Folder & Seed)------------------------------------ #
    # Step1 : Logger Initialize
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = get_logger(__name__)
    
    # Step2 : Repository creation
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    # Step3 : Seed everything
    if config.seed is not None:
        set_seed(config.seed)
    # -------------------------------------  Base Component Part 1 (Logger & Folder & Seed) ------------------------------------ #



    # -------------------------------------  Core Component(Accelerator & Data & Optimizer & Schedule)  ------------------------------------ #

    # Step1 : Accelerate Initialize
    project_dir = config.output_dir
    logging_dir = os.path.join(project_dir,config.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=project_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.log_with,
        project_config=accelerator_project_config,
    )

    # Step2 : Model 
    model = VQVAEModel(config=config)

    # Step3 : Dataset 
    train_data = VideoDataset(data_folder=config.datadir_train,sequence_length=config.sequence_lenth)
    val_data = VideoDataset(data_folder=config.datadir_train,sequence_length=config.sequence_lenth)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=config.batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size = config.batch_size,
                                shuffle=False)

    # Step4 : Optimizer & Learning Schedule
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr,)
    lr_scheduler = get_scheduler(          # scheduler from diffuser, auto warm-up
        name = config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
    )

    # Step5 : Prepare
    device = accelerator.device
    model, optimizer, training_dataloader,valing_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader,lr_scheduler
    )
    # -------------------------------------  Core Component(Accelerator & Data & Optimizer & Schedule)  ------------------------------------ #

    # ------------------------------------------------  Base Component(Progress & Tracker )  ----------------------------------------------- #

    # Step1: Progress bar
    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Step2 : Tracker
    hps = {"learning_rate": config.lr}
    accelerator.init_trackers("my_project", config=hps)
    # -----------------------------------------------  Base Component(Progress & Tracker )  ------------------------------------------------ #

    # -------------------------------------------------------  Train   --------------------------------------------------------------------- #

    # Step1 : Info!!
    total_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info(f"{accelerator.state}")
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_data)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    best_val_loss = 9999.0

    for epoch in range(config.num_train_epochs):
        train_loss = 0.0
        for step,data in enumerate(training_dataloader):
            # Step2 : train w/ forward & backward & update 
            model.train()
            with accelerator.accumulate(model):  
                # input
                input = data['video'] # N,F,C,H,W

                # forward
                recon_loss, x_recon, vq_output = model(input)
                
                # backpropagate
                accelerator.backward(recon_loss)
                if accelerator.sync_gradients:  # 检查梯度是否已经合并
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm) # 梯度裁剪避免梯度爆炸
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Step3 : do something between update and eval
            # 这是在更新完梯度和开始evaluation之间做的事，包含progress bar更新、log指标(tensorboard)和其他

            # progress bar & log 
            if accelerator.sync_gradients:
                global_step += 1

                # log , Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(recon_loss.repeat(config.batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # plot loss & lr per-step
                logs = {"step_loss": recon_loss.detach().item(), "lr": scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)

            # Step4 : eval & save model
            model.eval()
            # eval and save
            # 一般来说模型的保存和模型的评价、更新的步数是挂钩的
            if global_step % config.checkpointing_steps == 0:
                # save model during training if needed [Optional]
                logger.info(f"Running validation... \n")
                val_loss = []
                with torch.no_grad():
                    for i,val_data in enumerate(valing_dataloader):
                        # input
                        input = val_data['video'] # N,F,C,H,W
                        # forward
                        loss, x_recon, vq_output = model(input)
                        val_loss.append(loss)

                        if i>=20:
                            break
                    val_loss = sum(val_loss)/len(val_loss)
                    logger.info(f"Val loss:{val_loss} \n")
                
                if val_loss < best_val_loss:  # 只保存当前评价分数最好的模型
                    best_val_loss = val_loss
                    clean_checkpoint(folder=config.output_dir,limit=config.checkpoint_num_limit,logger=logger)
                    save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
            
            # Step5 : Stop training & Error handling
            # check max_train_steps
            if global_step > config.max_train_steps:
                break
    # -------------------------------------------------------  Train   --------------------------------------------------------------------- #
                        
    
    # Step Final : End accelerator
    accelerator.wait_for_everyone()
    accelerator.end_training() 


if __name__ == "__main__":
    # --------- Argparse ----------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    # --------- Config ----------#
    with open(args.config_path,'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)

    # --------- Train --------- #
    main(config)