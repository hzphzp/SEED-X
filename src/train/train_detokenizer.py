import hydra
import pyrootutils
import os
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('use Ascend NPU')
except:
    print('use NVIDIA GPU')
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import argparse
from typing import Optional
import transformers
from dataclasses import dataclass, field, asdict, is_dataclass
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, DistributedReadingService, SequentialReadingService
import gc
import logging

from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

print('============= train code')

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.train.schedular import get_scheduler
from src.train.dist_utils import all_gather

# logger = get_logger(__name__, log_level='info')
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)
# os.environ["WANDB_MODE"] = "offline"


@dataclass
class ConfigPathArguments:
    image_transform: Optional[str] = field(default=None, metadata={"help": "config path of image transform"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    # model: Optional[str] = field(default=None, metadata={"help": "config path of llm"})
    visual_encoder: Optional[str] = field(default=None, metadata={"help": "config path of visual encoder"})
    adapter_cfg_path: Optional[str] = field(default=None, metadata={"help": "config path of adapter"})
    diffusion_model_path: Optional[str] = field(default=None, metadata={"help": "config path of sdxl weight"})
    train_dataset: Optional[str] = field(default=None, metadata={"help": "config path of training dataset"})
    fsdp_plugin: Optional[str] = field(default=None, metadata={"help": "config path of fsdp plugin"})
    deepspeed_plugin: Optional[str] = field(default=None, metadata={"help": "config path of deepspeed plugin"})


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}, )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "The path to a folder with a valid checkpoint for your model."})
    resume_steps: Optional[int] = field(default=None, metadata={"help": "The training steps of saved checkpoint"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "Number of updates steps before the evaluation."})
    batch_size: Optional[int] = field(default=60, metadata={"help": "The training batch size"})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    mixed_precision: Optional[str] = field(
        default='no',
        metadata={
            "help":
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >=1.10.and an Nvidia Ampere GPU."
        })
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(default=-1, metadata={"help": "Total number of training steps to perform. "})
    save_steps: int = field(default=10000, metadata={"help": "Number of updates steps before two checkpoint saves."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The scheduler type to use."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    min_lr_ratio: float = field(default=0.01, metadata={"help": "Minimal learning rate ratio."})
    dataloader_num_workers: int = field(default=8, metadata={"help": "The number of workers to use for data loading."})
    project_name: str = field(default="ContinuousVLM", metadata={"help": "The name of experiment"})
    expr_name: str = field(default="", metadata={"help": "The name of experiment"})


def build_dataloader(dataset_cfg, image_transform, tokenizer, batch_size, dataloader_num_workers=4):
    dataset = hydra.utils.instantiate(dataset_cfg, image_transform=image_transform, tokenizer=tokenizer)
    mp_service = MultiProcessingReadingService(num_workers=dataloader_num_workers)
    dist_service = DistributedReadingService()
    reading_service = SequentialReadingService(dist_service, mp_service)
    dataloader = DataLoader2(dataset, reading_service=reading_service)
    # dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=dataloader_num_workers)
    return dataloader


def get_metric(output):
    metric = {}
    for key, value in output.items():
        if 'loss' in key:
            gathered_metric = torch.stack(all_gather(value)).mean()
            # metric[key] = value.item()
            metric[key] = gathered_metric.item()
        if 'acc' in key:
            metric[key] = value.item()
    return metric


def merge_config(**kwargs):
    config = {}
    for key, value in kwargs.items():
        if isinstance(value, argparse.Namespace):
            config[key] = vars(value)
        elif isinstance(value, DictConfig):
            config[key] = OmegaConf.to_object(value)
        elif is_dataclass(value):
            config[key] = asdict(value)
        elif isinstance(value, (int, str, float, dict)) or value is None:
            config[key] = value
        else:
            logger.error(f'key: {key}, value: {value} will not be merged.')
    return config


def trainable_params(model):
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count += param.numel()
    return count


def train():

    parser = transformers.HfArgumentParser((ConfigPathArguments, TrainingArguments))
    cfg_path, args = parser.parse_args_into_dataclasses()

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, 'logs'))

    assert int(cfg_path.fsdp_plugin is not None) + int(cfg_path.deepspeed_plugin is not None) <= 1
    if cfg_path.fsdp_plugin is not None:
        fsdp_plugin_cfg = OmegaConf.load(cfg_path.fsdp_plugin)
        fsdp_plugin = hydra.utils.instantiate(fsdp_plugin_cfg)
        logger.info('Use FSDP plugin')
    else:
        fsdp_plugin = None

    if cfg_path.deepspeed_plugin is not None:
        deepspeed_plugin_cfg = OmegaConf.load(cfg_path.deepspeed_plugin)
        deepspeed_plugin = hydra.utils.instantiate(deepspeed_plugin_cfg)
        logger.info('Use deepspeed plugin')
    else:
        deepspeed_plugin = None

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=['tensorboard'],
        # log_with=['tensorboard', 'wandb'] if os.environ.get('DEBUG_FLAG', 'False') != 'True' else ['tensorboard'],
        project_config=project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin,
        # kwargs_handlers=[ddp_kwargs],
    )
    accelerator.wait_for_everyone()
    logger.info('Init accelerator done.')

    if cfg_path.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 8

    # logging('deepspeed config: ', accelerator.state.deepspeed_plugin.deepspeed_config)

    os.makedirs(args.output_dir, exist_ok=True)

    image_transform_cfg = OmegaConf.load(cfg_path.image_transform)
    image_transform = hydra.utils.instantiate(image_transform_cfg)

    tokenizer_cfg = OmegaConf.load(cfg_path.tokenizer)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    train_dataset_cfg = OmegaConf.load(cfg_path.train_dataset)

    visual_encoder_cfg = OmegaConf.load(cfg_path.visual_encoder)
    visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
    logger.info('Load visual encoder done.')

    # llm_model_cfg = OmegaConf.load(cfg_path.llm_model)
    # llm_model = hydra.utils.instantiate(llm_model_cfg, torch_dtype=accelerator.mixed_precision)
    # llm_model.gradient_checkpointing_enable()
    # llm_model.config.use_cache = False
    # logger.info('Load llm model done.')

    # agent_model_cfg = OmegaConf.load(cfg_path.agent_model)
    # agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm_model)
    # logger.info('Load agent model done.')
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # if cfg_path.fsdp_plugin is not None:
    #     agent_model = accelerator.prepare(agent_model)

    adapter_cfg_path = cfg_path.adapter_cfg_path
    adapter_cfg = OmegaConf.load(adapter_cfg_path)
    diffusion_model_path = cfg_path.diffusion_model_path

    logger.info('init vae')
    vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae")
    logger.info('init noise scheduler')
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
    logger.info('init unet')
    unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet")

    
    visual_encoder.to(accelerator.device, dtype=weight_dtype)
    logger.info('Freeze visual encoder...')
    visual_encoder.requires_grad_(False)
    visual_encoder.eval()
    print('visual_encoder:', visual_encoder.transformer.resblocks[0].training)
    
    logger.info('Freeze visual vae...')
    vae.requires_grad_(False)
    vae = vae.eval()
    
    logger.info('init ip adapter')
    adapter = hydra.utils.instantiate(adapter_cfg, unet=unet)

    adapter.init_pipe(vae=vae,
                    scheduler=noise_scheduler,
                    visual_encoder=visual_encoder,
                    image_transform=image_transform,
                    device=accelerator.device, 
                    dtype=weight_dtype
                    )

    if cfg_path.fsdp_plugin is not None:
        adapter = accelerator.prepare(adapter)

    optimizer = torch.optim.AdamW(adapter.params_to_opt(),
                                  lr=args.learning_rate,
                                  betas=[args.adam_beta1, args.adam_beta2],
                                  eps=args.adam_epsilon,
                                  weight_decay=args.weight_decay)
    logger.info('Init optimizer done.')
    scheduler = get_scheduler(name=args.lr_scheduler_type,
                              optimizer=optimizer,
                              num_warmup_steps=args.warmup_steps,
                              num_training_steps=args.max_steps,
                              min_lr_ratio=args.min_lr_ratio)
    # accelerator.register_for_checkpointing(scheduler)
    train_dataloader = build_dataloader(dataset_cfg=train_dataset_cfg,
                                        image_transform=image_transform,
                                        tokenizer=tokenizer,
                                        batch_size=args.batch_size,
                                        dataloader_num_workers=args.dataloader_num_workers)
    if cfg_path.fsdp_plugin is not None:
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    else:
        adapter, optimizer, scheduler = accelerator.prepare(adapter, optimizer, scheduler)
    logger.info('Prepare accelerator done.')

    config_record = merge_config(adapter=adapter_cfg,
                                 visual_encoder=visual_encoder_cfg,
                                 image_transform=image_transform_cfg,
                                 tokenizer=tokenizer_cfg,
                                 train_dataset=train_dataset_cfg,
                                 train_args=args)
    accelerator.init_trackers(project_name="seed_x_cn",
                              init_kwargs={"wandb": {
                                  "config": config_record,
                                  "name": args.expr_name,
                                  "dir": args.output_dir,
                                #   "project": 'MultiModal-LLM Research', 
                                  "entity": 'hzp1104',
                                  "resume": 'allow',
                                  'mode': 'offline'
                              }})
    if args.resume_from_checkpoint is not None:
        logger.info(f'Load checkpoint from {args.resume_from_checkpoint}')
        accelerator.load_state(args.resume_from_checkpoint)
        torch.cuda.empty_cache()
        gc.collect()

    num_params = trainable_params(adapter)
    logger.info("***** Running training *****")
    logger.info(f"  Total optimization steps = {args.max_steps}")
    logger.info(f"  Total trainable params = {num_params}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    if args.resume_steps is not None:
        global_step = args.resume_steps
        progress_bar.update(args.resume_steps)

    for epoch in range(args.num_train_epochs):
        adapter.train()
        logger.info('Start new epoch')

        #  change seed 
        if args.resume_steps is not None:
            seed = args.resume_steps + epoch + 42
        else:
            seed = epoch + 42
        train_dataloader.seed(seed)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(adapter):
                images = batch['images'].to(accelerator.device) if batch['images'] is not None else None
                if images is not None:
                    # embeds_gen_mask=batch['embeds_gen_mask'].to(accelerator.device)
                    # embeds_cmp_mask=batch['embeds_cmp_mask'].to(accelerator.device)
                    
                    # embeds_valid_mask = torch.logical_or(embeds_gen_mask, embeds_cmp_mask)
                    # embeds_gen_mask = embeds_gen_mask[embeds_valid_mask]
                    # embeds_cmp_mask = embeds_cmp_mask[embeds_valid_mask]
                    # images = images[embeds_valid_mask]

                    # if 'patch_position' in batch:
                    #     patch_position = batch['patch_position'].to(accelerator.device) 
                    #     patch_position = patch_position[embeds_valid_mask]
                    # else:
                    # patch_position = None

                    if images.shape[0] == 0:
                        images = None

                with torch.no_grad():
                    if images is not None:
                        assert 'patch_position' not in batch
                        image_embeds = visual_encoder(images)
                    else:
                        image_embeds = None

                # output = agent_model(input_ids=batch['input_ids'].to(accelerator.device),
                #                      attention_mask=batch['attention_mask'].to(accelerator.device),
                #                      labels=batch['labels'].to(accelerator.device),
                #                      image_embeds=image_embeds,
                #                      patch_positions=patch_position if images is not None else None,
                #                      embeds_gen_mask=embeds_gen_mask
                #                      if batch['embeds_gen_mask'] is not None else None,
                #                      embeds_cmp_mask=embeds_cmp_mask
                #                      if batch['embeds_cmp_mask'] is not None else None,
                #                      ids_gen_mask=batch['ids_gen_mask'].to(accelerator.device),
                #                      ids_cmp_mask=batch['ids_cmp_mask'].to(accelerator.device))
                
                # get latent and noise
                latents = adapter.compute_vae_encodings(images)['model_input']
                noise = torch.randn_like(latents)
                # add noise to latents
                bsz = latents.shape[0]
                timesteps = torch.randint(
                        0, adapter.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                # get noisy_latents
                noisy_latents = adapter.scheduler.add_noise(latents, noise, timesteps)
                # # get time ids
                # def compute_time_ids(original_size, crops_coords_top_left):
                #     # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                #     target_size = (args.resolution, args.resolution)
                #     add_time_ids = list(original_size + crops_coords_top_left + target_size)
                #     add_time_ids = torch.tensor([add_time_ids])
                #     add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                #     return add_time_ids
                # add_time_ids = torch.cat(
                #     [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                # )
                # def forward(self, noisy_latents, timesteps, image_embeds, text_embeds, noise, time_ids):
                output = adapter(
                                noisy_latents=noisy_latents,
                                timesteps=timesteps,
                                image_embeds=image_embeds,
                                text_embeds=None,
                                noise=noise,
                                time_ids=torch.zeros([bsz, 1, 6], device=latents.device, dtype=torch.long)
                                )
                loss = output['total_loss']
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(adapter.params_to_opt(), max_norm=args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

            metric = get_metric(output)
            metric['lr'] = optimizer.param_groups[0]['lr']
            accelerator.log(metric, step=global_step)
            metric = {key: (format(value, ".6f") if isinstance(value, float) else value) for key, value in metric.items()}
            if accelerator.is_main_process:
                tqdm.write(str(metric))
                if global_step % args.eval_steps in [0, 1, 2, 3]:
                    adapter.eval()
                    # sample images and log recon images
                    generated_images = adapter.generate(image_tensor=images, num_inference_steps=50)
                    # generated_images[0].save(save_path)
                    # save to accelerate tensorboard
                    # save transformed images tensors
                    save_input_image = images[0].detach().cpu().numpy().transpose(1, 2, 0)
                    print(save_input_image.shape)
                    print(save_input_image.min(), save_input_image.max())
                    save_input_image = Image.fromarray(save_input_image)
                    # accelerator.log(save_input_image, f"input_images/{global_step}_origin.png")
                    # accelerator.log(generated_images[0], f"recon_images/{global_step}.png")
                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            tracker.writer.add_image(f"input_images/{global_step}_origin", save_input_image, global_step)
                            tracker.writer.add_image(f"recon_images/{global_step}", generated_images[0], global_step)
                    adapter.train()
            if global_step >= args.max_steps:
                break

    accelerator.end_training()


if __name__ == '__main__':
    train()
