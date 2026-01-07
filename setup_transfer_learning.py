import os
import torch
from expand_model import expand_model_checkpoint

def main():    
    # Step 1: Choose best checkpoint    
    best_checkpoint = "./model/actor/trained_model_25000.pth"
    
    if not os.path.exists(best_checkpoint):
        print(f"Checkpoint not found: {best_checkpoint}")
        print("\nAvailable checkpoints:")
        if os.path.exists("./model/actor"):
            checkpoints = [f for f in os.listdir("./model/actor") if f.endswith('.pth')]
            for cp in sorted(checkpoints)[:10]:  # Show first 10
                print(f"   - ./model/actor/{cp}")
        else:
            print("   No checkpoints found in ./model/actor/")
        print("\nPlease update 'best_checkpoint' variable in this script.")
        return
    
    print(f"Using checkpoint: {best_checkpoint}")
    
    # Step 2: Expand the model
    output_path = "./model2/expanded_model.pth"
    os.makedirs("./model2", exist_ok=True)
    
    try:
        # Check if we need both actor and critic in one checkpoint
        # Your original checkpoints have them separate, so we'll combine them
        actor_path = best_checkpoint
        critic_path = best_checkpoint.replace("/actor/", "/critic/")
        
        if os.path.exists(critic_path):
            print("Found both actor and critic checkpoints")
            
            # Load and expand both
            actor_state = expand_model_checkpoint(actor_path, 24, 20)
            print()  # Add blank line
            critic_state = expand_model_checkpoint(critic_path, 24, 20)
            
            # Combine into one checkpoint
            combined = {
                'actor': actor_state,
                'critic': critic_state
            }
            torch.save(combined, output_path)
            print(f"\nCombined expanded checkpoint saved: {output_path}")
        else:
            print("Critic checkpoint not found, using actor only")
            actor_state = expand_model_checkpoint(actor_path, 24, 20)
            torch.save({'actor': actor_state}, output_path)
    
    except Exception as e:
        print(f"Error during expansion: {e}")
        return
    
    # Step 3: Verify the expansion    
    try:
        checkpoint = torch.load(output_path, map_location='cpu')
        
        # Check first layer dimensions
        actor_fc_weight = checkpoint['actor']['fc.weight']
        critic_fc_weight = checkpoint['critic']['fc.weight']
        
        print(f"Actor first layer input size: {actor_fc_weight.shape[1]} (should be 24)")
        print(f"Critic first layer input size: {critic_fc_weight.shape[1]} (should be 24)")
        
        if actor_fc_weight.shape[1] == 24 and critic_fc_weight.shape[1] == 24:
            print("\nModel expansion successful!")
        else:
            print("\nUnexpected dimensions, please check")
            return
            
    except Exception as e:
        print(f"Error verifying model: {e}")
        return
    
    print("Setup complete! Ready for fine-tuning.")

if __name__ == "__main__":
    main()
