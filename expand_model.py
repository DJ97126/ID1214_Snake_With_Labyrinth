import torch
import torch.nn as nn
from modules import Actor, Critic

def expand_linear_layer(old_layer, new_input_size):
    # Expand a Linear layer's input dimension while preserving existing weights.
    old_input_size = old_layer.in_features
    output_size = old_layer.out_features
    
    # Create new layer with expanded input
    new_layer = nn.Linear(new_input_size, output_size)
    
    # Copy existing weights
    with torch.no_grad():
        new_layer.weight[:, :old_input_size] = old_layer.weight
        new_layer.bias.copy_(old_layer.bias)
        
        # Initialize new weights with small random values
        # Using Xavier/Glorot initialization for the new weights
        nn.init.xavier_uniform_(new_layer.weight[:, old_input_size:])
    
    return new_layer

def expand_model_checkpoint(checkpoint_path, new_obs_size=24, old_obs_size=20, is_separate_files=False):
    # Load a checkpoint and expand the first layer from old_obs_size to new_obs_size.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Expanding observation space from {old_obs_size} to {new_obs_size}")
    
    # Determine if this is actor or critic based on path
    if 'actor' in checkpoint_path:
        # Create old model to load weights
        old_model = Actor(obs_size=old_obs_size, n_act=4)
        new_model = Actor(obs_size=new_obs_size, n_act=4)
        model_type = "Actor"
    else:
        old_model = Critic(obs_size=old_obs_size, n_act=4)
        new_model = Critic(obs_size=new_obs_size, n_act=4)
        model_type = "Critic"
    
    # Load state dict (checkpoint is the state_dict directly)
    old_model.load_state_dict(checkpoint)
    
    # Get the old state dict
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    
    # Expand the fc layer weights manually
    old_fc_weight = old_state['fc.weight']  # Shape: [512, 20]
    old_fc_bias = old_state['fc.bias']      # Shape: [512]
    
    # Create new fc weights with expanded input
    new_fc_weight = new_state['fc.weight'].clone()  # Shape: [512, 24]
    new_fc_bias = old_fc_bias.clone()
    
    # Copy old weights to new tensor
    new_fc_weight[:, :old_obs_size] = old_fc_weight
    # The remaining columns are already randomly initialized by the new model
    
    # Update the new state dict
    new_state['fc.weight'] = new_fc_weight
    new_state['fc.bias'] = new_fc_bias
    
    # Copy all other layers
    for key in old_state.keys():
        if key not in ['fc.weight', 'fc.bias']:
            new_state[key] = old_state[key]
    
    # Load the combined state dict
    new_model.load_state_dict(new_state)
    
    print(f"{model_type} first layer expanded")
    print(f"Existing {old_obs_size} weights transferred, {new_obs_size - old_obs_size} new weights initialized")
    
    return new_model.state_dict()

def save_expanded_checkpoint(checkpoint_path, output_path=None, new_obs_size=24, old_obs_size=20):
    # Load, expand, and save a checkpoint.
    # Generate output path if not provided
    if output_path is None:
        output_path = checkpoint_path.replace('.pth', '_expanded.pth')
    
    # Expand the model
    actor_state, critic_state = expand_model_checkpoint(
        checkpoint_path, new_obs_size, old_obs_size
    )
    
    # Save expanded checkpoint
    torch.save({
        'actor': actor_state,
        'critic': critic_state
    }, output_path)
    
    print(f"\nExpanded checkpoint saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Example usage - modify the path to your best checkpoint
    checkpoint_path = "./model2/actor/trained_model_85000.pth"  # Change this to your best model
    
    print("="*60)
    print("Model Expansion Tool - Transfer Learning from previous stage")
    print("="*60)
    print()
    
    # Expand and save
    expanded_path = save_expanded_checkpoint(
        checkpoint_path=checkpoint_path,
        new_obs_size=24,
        old_obs_size=20
    )
    
