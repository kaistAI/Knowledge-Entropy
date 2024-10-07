import torch
import argparse
import os

def main(step, resuscitation_ratio, amplifying_factor):
    average_coef_path =f"checkpoints/pretrained_1B/{step}-unsharded/mlp_average_coefficients.pt"
    olmo_model_path = f"checkpoints/pretrained_1B/{step}-unsharded"
    if not os.path.isfile(average_coef_path):
        print(f"Get average coefficients first, by running `python -m analysis.entropy --step {step} --data_size 2048 --batch_size 4`")
    print("-"*50)
    average_activations = torch.load(average_coef_path)
    olmo_model = torch.load(f"{olmo_model_path}/model.pt")
    print("Loaded average coefficients and olmo model.")
    print(f"Saving new model parameters with resusctation ratio {resuscitation_ratio}, amplifying factor {amplifying_factor}")
    # Identify positions of values below the threshold
    mask_list = []  
    for layer_idx in range(average_activations.size(0)): 
        activations = average_activations[layer_idx]
        threshold_value = torch.quantile(activations, resuscitation_ratio)
        mask = activations <= threshold_value
        mask_list.append(mask)
        
    ratio_list = []  
    for layer_idx in range(average_activations.size(0)): 
        activations = average_activations[layer_idx]
        activations = activations.mean() / activations * amplifying_factor
        ratio_list.append(activations)

    for layer_idx, mask in enumerate(mask_list):
        # Find indices of the lowest p% activations
        indices_to_freeze = mask.nonzero(as_tuple=True)[0].to('cpu')

        # Get the target module's weight tensor
        ff_proj_weight_tensor = olmo_model[f'transformer.blocks.{layer_idx}.ff_proj.weight'] # Shape [16384, 2048]

        # Reshape ratio_list[layer_idx] to have the correct dimensions for broadcasting
        proj_random_tensor = ratio_list[layer_idx].unsqueeze(1).to('cpu')  # Shape [8192, 1]
        with torch.no_grad():
            # Scale only the specified rows in ff_proj_weight_tensor
            ff_proj_weight_tensor[indices_to_freeze, :] *= proj_random_tensor[indices_to_freeze]

    torch.save(olmo_model, f'{olmo_model_path}/resuscitation_ratio{str(resuscitation_ratio)}_amplifying{str(amplifying_factor)}.pt')
    print("Saving Complete")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--resuscitation_ratio", type=float, default=None)
    parser.add_argument("--amplifying_factor", type=float, default=None)
    
    args = parser.parse_args()
    main(args.step, args.resuscitation_ratio, args.amplifying_factor)