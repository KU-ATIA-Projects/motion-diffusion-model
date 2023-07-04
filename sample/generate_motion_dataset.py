# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import sys
from getpass import getuser
sys.path.append(f'/home/{getuser()}/motion-diffusion-model/')
sys.path.append(f'/home/{getuser()}/motion-diffusion-model/flame')

from pathlib import Path
import json
import shutil
import torch
import numpy as np
from tqdm import tqdm

from data_loaders.tensors import collate
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import generate_motion_dataset_args
from utils.fixseed import fixseed

from flame.clip_similarity import ClipSimilarity


def main():
    args = generate_motion_dataset_args()
    if args.seed != -1:
        fixseed(args.seed)
    else:
        seed = torch.randint(1 << 32, ()).item()
        fixseed(seed)
        print("The seed is: {}".format(seed))
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace(
        'model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any(
        [args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + \
                args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + \
                os.path.basename(args.input_text).replace(
                    '.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    assert args.input_text != '' and os.path.exists(
        args.input_text) and args.input_text.endswith('.jsonl')
    with open(args.input_text, 'r') as fp:
        # We assume that the input text file is a jsonl file with the following format:
        # {"input": "The first sentence", "output": "The second sentence"}
        # We make sure the odd lines are the input and the even lines are the output
        # Therefore we don't need to change MDM code to support this format
        prompts = [json.loads(line) for line in fp]
        prompts = prompts[args.index_base:args.index_base + args.num_samples]
        texts = [[prompt['prompt'], prompt['edited']] for prompt in prompts]
        texts = [item for sublist in texts for item in sublist]
        args.num_samples = len(texts)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    # Sampling a single batch from the testset, with exactly args.num_samples
    args.batch_size = args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        # wrapping model with the classifier-free sampler
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(
            n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt)
                            for arg, txt in zip(collate_args, texts)]
        else:
            # a2m
            pass
            # action = data.dataset.action_name_to_action(action_text)
            # collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
            # arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')
        p2p_threshold = args.min_p2p + torch.rand(()).item() * (args.max_p2p - args.min_p2p)
        args.prompt2prompt_threshold = p2p_threshold
        model.model.prompt2prompt_threshold = p2p_threshold
        print(f"The args.prompt2prompt_threshold is {args.prompt2prompt_threshold}")

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(
                args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in [
            'xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(
            args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    clip_similarity_filtering(all_motions, prompts, out_path, args)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


def clip_similarity_filtering(all_motions, prompts, out_path, args, ckpt_path=f'/home/{getuser()}/flame/flame_mclip_hml3d_bc.ckpt'):
    clip_similarity = ClipSimilarity(
        ckpt_path=ckpt_path, cuda=True, device='cuda:0')
    
    out_dir = Path(out_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    for i, prompt in enumerate(prompts):
        prompt_dir = out_dir.joinpath(f'{args.index_base + i:07d}') 
        prompt_dir.mkdir(parents=True, exist_ok=True)

        with open(prompt_dir.joinpath('prompt.json'), 'w') as fp:
            json.dump(prompt, fp)

        results = []

        for rep in range(args.num_repetitions):
            motion_0 = all_motions[rep][2 * i].transpose(2, 0, 1)
            motion_1 = all_motions[rep][2 * i + 1].transpose(2, 0, 1)
            text_0 = [prompt['prompt']]
            text_1 = [prompt['edited']]

            sim_0, sim_1, sim_direction, sim_motion = clip_similarity(
                torch.tensor(motion_0, device='cuda:0'),
                torch.tensor(motion_1, device='cuda:0'),
                list(text_0),
                list(text_1))
            
            results.append(dict(
                motion_0=motion_0,
                motion_1=motion_1,
                text_0=text_0,
                text_1=text_1,
                p2p_threshold=args.prompt2prompt_threshold,
                cfg_scale=args.guidance_param,
                clip_sim_0=sim_0,
                clip_sim_1=sim_1,
                clip_sim_direction=sim_direction,
                clip_sim_motion=sim_motion,
            ))

        # ! Don't filter but save all results
        # filtered_results = list(filter(lambda x: x['clip_sim_0'] >= args.clip_threshold 
        #                                and x['clip_sim_1'] >= args.clip_threshold 
        #                                and x['clip_sim_direction'] >= args.clip_dir_threshold 
        #                                and x['clip_sim_motion'] >= args.clip_motion_threshold, 
        #                                results))
        filtered_results = results
        filtered_results.sort(key=lambda x: x['clip_sim_direction'], reverse=True)
        # ! save all results
        # filtered_results = filtered_results[:args.max_out_samples]
        for k, res in enumerate(filtered_results):
            motion_0 = res['motion_0']
            motion_1 = res['motion_1']
            text_0 = res['text_0']
            text_1 = res['text_1']

            plot_3d_motion(prompt_dir.joinpath(f"{i:07d}_{k}_0.mp4"), 
                           paramUtil.t2m_kinematic_chain, 
                           motion_0,  
                           text_0[0], 
                           'humanml',
                           fps=20)
            
            plot_3d_motion(prompt_dir.joinpath(f"{i:07d}_{k}_1.mp4"),
                           paramUtil.t2m_kinematic_chain,
                           motion_1,
                           text_1[0],
                           'humanml',
                           fps=20)
            motion_0_path = prompt_dir.joinpath(f"{i:07d}_{k}_0.npy")
            motion_1_path = prompt_dir.joinpath(f"{i:07d}_{k}_1.npy")
            np.save(motion_0_path, motion_0)
            np.save(motion_1_path, motion_1)
            try:
                with open(prompt_dir.joinpath(f"metadata.jsonl"), "a") as fp:
                    fp.write(json.dumps(dict(
                        motion_0=os.path.abspath(motion_0_path), 
                        motion_1=os.path.abspath(motion_1_path),
                        text_0=text_0[0],
                        text_1=text_1[0],
                        p2p_threshold=args.prompt2prompt_threshold,
                        cfg_scale=args.guidance_param,
                        clip_sim_0=res['clip_sim_0'].cpu().numpy().tolist(),
                        clip_sim_1=res['clip_sim_1'].cpu().numpy().tolist(),
                        clip_sim_direction=res['clip_sim_direction'].cpu().numpy().tolist(),
                        clip_sim_motion=res['clip_sim_motion'].cpu().numpy().tolist(),
                        )) + '\n')
            except Exception as e:
                import warnings
                warnings.warn(e)


if __name__ == "__main__":
    main()
