# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys
from getpass import getuser
sys.path.append(f'/home/{getuser()}/motion-diffusion-model/')
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
    
    # out_path = './assets/interpolation'
    os.makedirs(out_path, exist_ok=True)

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    
    data_conditioned, data_unconditioned = load_data(timesteps=0, repetitions=30, latent_vec_path='/home/ctq566/motion-diffusion-model/latent_vec/latent_vec_4')
    data_conditioned = torch.from_numpy(data_conditioned).float().to('cuda:0')

    mdm = model.model
    # repetitions = 30
    # for rep in range(repetitions):
    #     output = mdm.seqTransEncoder(data_conditioned[rep])[1:]
    #     output = mdm.output_process(output)
    #     sample = output.detach()
    #     sample.requires_grad = False

    #     # Recover XYZ *positions* from HumanML3D vector representation
    #     if model.data_rep == 'hml_vec':
    #         n_joints = 22 if sample.shape[1] == 263 else 21
    #         sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
    #         sample = recover_from_ric(sample, n_joints)
    #         sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

    #     rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
    #     rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
    #     sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
    #                             jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
    #                             get_rotations_back=False)
    #     motion = sample.cpu().numpy()
    #     skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
        
        # for motion_id in range(motion.shape[0]):
        #     np.save(os.path.join(out_path, f'sample{motion_id}__rep{rep}.npy'), motion[motion_id].transpose(2, 0, 1))
        #     # plot_3d_motion(f'./assets/interpolation/sample{motion_id}_rep{rep}.mp4', skeleton, motion[motion_id].transpose(2, 0, 1), dataset=args.dataset, title=f'sample{motion_id}_rep{rep}', fps=20)


    

    motion_turns_right = data_conditioned[11][:, 1, :]
    motion_sits_down = data_conditioned[11][:, 2, :]
    


    interpolations = [interpolate(motion_turns_right, motion_sits_down, alpha) for alpha in np.linspace(-0.25, 1.25, 7)]
    interpolations = torch.stack(interpolations).transpose(1, 0)
    output = mdm.seqTransEncoder(interpolations)[1:]
    output = mdm.output_process(output)
    sample = output.detach()
    sample.requires_grad = False

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        n_joints = 22 if sample.shape[1] == 263 else 21
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    
    rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
    rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
    sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                            jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                            get_rotations_back=False)
    motion = sample.cpu().numpy()
    for motion_id in range(motion.shape[0]):
        np.save(os.path.join(out_path, f'interpolation{motion_id}.npy'), motion[motion_id].transpose(2, 0, 1))
        # plot_3d_motion(f'./assets/interpolation/sample{motion_id}_rep{rep}.mp4', skeleton, motion[motion_id].transpose(2, 0, 1), dataset=args.dataset, title=f'sample{motion_id}_rep{rep}', fps=20)
    print('done')




def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


def load_data(timesteps, repetitions, latent_vec_path='../latent_vec/latent_vec_4'):
    data_conditioned = []
    data_unconditioned = []
    for i in range(repetitions):
        data_conditioned.append(np.load(os.path.join(latent_vec_path, f'latent_vec_{timesteps}_{2 * i}.npy')))
        data_unconditioned.append(np.load(os.path.join(latent_vec_path, f'latent_vec_{timesteps}_{2 * i + 1}.npy')))
    return np.array(data_conditioned), np.array(data_unconditioned)


def interpolate(data1, data2, alpha):
    return alpha * data1 + (1 - alpha) * data2


if __name__ == "__main__":
    main()
