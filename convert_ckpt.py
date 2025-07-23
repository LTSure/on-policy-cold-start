import torch
import os
import json
from safetensors.torch import save_file
import shutil

def debug_embedding_shapes(checkpoint_dir, world_size=8):
    """
    Specifically debug embedding tensor shapes across ranks.
    """
    print("=== Debugging Embedding Shapes ===")
    
    # Load config
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        expected_vocab_size = config.get('vocab_size', 'unknown')
        hidden_size = config.get('hidden_size', 'unknown')
        print(f"Expected vocab_size from config: {expected_vocab_size}")
        print(f"Expected hidden_size from config: {hidden_size}")
    else:
        print("No config.json found")
        expected_vocab_size = None
        hidden_size = None
    
    # Check embedding shapes across all ranks
    embedding_shapes = []
    lm_head_shapes = []
    
    for rank in range(world_size):
        pt_file = os.path.join(checkpoint_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        if not os.path.exists(pt_file):
            continue
            
        print(f"\n--- Rank {rank} ---")
        try:
            state_dict = torch.load(pt_file, map_location='cpu', weights_only=False)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            for name, tensor in state_dict.items():
                if hasattr(tensor, 'to_local'):
                    tensor = tensor.to_local()
                elif hasattr(tensor, '_local_tensor'):
                    tensor = tensor._local_tensor
                
                if isinstance(tensor, torch.Tensor):
                    if 'embed_tokens' in name:
                        embedding_shapes.append((rank, tensor.shape))
                        print(f"  {name}: {tensor.shape}")
                    elif 'lm_head' in name:
                        lm_head_shapes.append((rank, tensor.shape))
                        print(f"  {name}: {tensor.shape}")
                        
        except Exception as e:
            print(f"Error loading rank {rank}: {e}")
    
    # Analyze the patterns
    print(f"\n=== Analysis ===")
    if embedding_shapes:
        total_vocab_from_shards = sum(shape[0] for _, shape in embedding_shapes)
        print(f"Total vocab size from embedding shards: {total_vocab_from_shards}")
        print(f"Expected vocab size from config: {expected_vocab_size}")
        
        if expected_vocab_size and total_vocab_from_shards != expected_vocab_size:
            print(f"⚠️  MISMATCH: Sharded vocab size ({total_vocab_from_shards}) != config vocab size ({expected_vocab_size})")
    
    return embedding_shapes, lm_head_shapes, expected_vocab_size, hidden_size

def consolidate_embeddings_correctly(checkpoint_dir, world_size=8):
    """
    Consolidate embeddings with special handling for vocabulary dimension.
    """
    print("\n=== Consolidating Embeddings Correctly ===")
    
    # First debug to understand the structure
    embedding_shapes, lm_head_shapes, expected_vocab_size, hidden_size = debug_embedding_shapes(checkpoint_dir, world_size)
    
    # Load all checkpoints
    all_state_dicts = []
    for rank in range(world_size):
        pt_file = os.path.join(checkpoint_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        if not os.path.exists(pt_file):
            continue
            
        state_dict = torch.load(pt_file, map_location='cpu', weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Convert DTensors to regular tensors
        processed_state_dict = {}
        for name, tensor in state_dict.items():
            if hasattr(tensor, 'to_local'):
                tensor = tensor.to_local()
            elif hasattr(tensor, '_local_tensor'):
                tensor = tensor._local_tensor
            
            if isinstance(tensor, torch.Tensor):
                processed_state_dict[name] = tensor
        
        all_state_dicts.append(processed_state_dict)
    
    # Get all parameter names
    all_param_names = set()
    for state_dict in all_state_dicts:
        all_param_names.update(state_dict.keys())
    
    consolidated = {}
    
    for param_name in all_param_names:
        tensors = []
        for state_dict in all_state_dicts:
            if param_name in state_dict:
                tensors.append(state_dict[param_name])
        
        if not tensors:
            continue
        
        print(f"\nProcessing {param_name}...")
        print(f"  Shapes: {[t.shape for t in tensors]}")
        
        if len(tensors) == 1:
            consolidated[param_name] = tensors[0]
            print(f"  -> Single tensor, using as-is")
        else:
            # Special handling for different parameter types
            if 'embed_tokens' in param_name or 'lm_head' in param_name:
                # Embedding layers: concatenate along vocabulary dimension (dim=0)
                try:
                    consolidated_tensor = torch.cat(tensors, dim=0)
                    print(f"  -> Concatenated along vocab dim (0): {consolidated_tensor.shape}")
                    
                    # Verify against expected vocab size
                    if expected_vocab_size and consolidated_tensor.shape[0] != expected_vocab_size:
                        print(f"  ⚠️  Warning: Consolidated vocab size {consolidated_tensor.shape[0]} != expected {expected_vocab_size}")
                        # Try different approaches
                        if consolidated_tensor.shape[0] > expected_vocab_size:
                            print(f"  -> Truncating to expected vocab size")
                            consolidated_tensor = consolidated_tensor[:expected_vocab_size]
                        else:
                            print(f"  -> Padding to expected vocab size")
                            padding_size = expected_vocab_size - consolidated_tensor.shape[0]
                            padding = torch.zeros(padding_size, *consolidated_tensor.shape[1:], dtype=consolidated_tensor.dtype)
                            consolidated_tensor = torch.cat([consolidated_tensor, padding], dim=0)
                    
                    consolidated[param_name] = consolidated_tensor
                    
                except Exception as e:
                    print(f"  -> Error concatenating embedding: {e}")
                    print(f"  -> Using first tensor as fallback")
                    consolidated[param_name] = tensors[0]
                    
            elif any(x in param_name for x in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']):
                # Attention/MLP projections: concatenate along output dimension (dim=0)
                try:
                    consolidated_tensor = torch.cat(tensors, dim=0)
                    consolidated[param_name] = consolidated_tensor
                    print(f"  -> Concatenated along output dim (0): {consolidated_tensor.shape}")
                except Exception as e:
                    print(f"  -> Error concatenating: {e}, using first tensor")
                    consolidated[param_name] = tensors[0]
                    
            elif any(x in param_name for x in ['o_proj', 'down_proj']):
                # Output projections: concatenate along input dimension (dim=1)
                try:
                    consolidated_tensor = torch.cat(tensors, dim=1)
                    consolidated[param_name] = consolidated_tensor
                    print(f"  -> Concatenated along input dim (1): {consolidated_tensor.shape}")
                except Exception as e:
                    print(f"  -> Error concatenating: {e}, using first tensor")
                    consolidated[param_name] = tensors[0]
                    
            elif 'norm' in param_name or 'bias' in param_name:
                # Layer norms and biases: should be identical across ranks
                consolidated[param_name] = tensors[0]
                print(f"  -> Using replicated tensor: {tensors[0].shape}")
                
            else:
                # Unknown parameter type: try concatenating along dim=0
                try:
                    consolidated_tensor = torch.cat(tensors, dim=0)
                    consolidated[param_name] = consolidated_tensor
                    print(f"  -> Default concatenation along dim 0: {consolidated_tensor.shape}")
                except Exception as e:
                    print(f"  -> Error with default concatenation: {e}, using first tensor")
                    consolidated[param_name] = tensors[0]
    
    return consolidated

def save_with_proper_vocab_size(consolidated_state_dict, output_dir, config_path):
    """
    Save the consolidated model, ensuring vocab size consistency.
    """
    print(f"\n=== Saving Model ===")
    
    # Load config to verify vocab size
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        expected_vocab_size = config.get('vocab_size')
    else:
        expected_vocab_size = None
    
    # Check embedding tensor sizes
    for name, tensor in consolidated_state_dict.items():
        if 'embed_tokens' in name or 'lm_head' in name:
            print(f"{name}: {tensor.shape}")
            if expected_vocab_size and tensor.shape[0] != expected_vocab_size:
                print(f"⚠️  Final check: {name} vocab size {tensor.shape[0]} != expected {expected_vocab_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as single safetensors file for testing
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(consolidated_state_dict, safetensors_path)
    print(f"Saved consolidated model to {safetensors_path}")
    
    # Copy config files
    config_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    for config_file in config_files:
        src = os.path.join(os.path.dirname(config_path), config_file)
        if os.path.exists(src):
            dst = os.path.join(output_dir, config_file)
            shutil.copy2(src, dst)
            print(f"Copied {config_file}")
    
    return safetensors_path

# Main execution
if __name__ == "__main__":
    checkpoint_dir = "/cpfs04/user/liutianshuo/verl-agent/checkpoints/verl_agent_alfworld/grpo_llama3.2_3b_pposft_0710/global_step_120/actor"
    output_dir = "/cpfs04/user/liutianshuo/verl-agent/checkpoints/verl_agent_alfworld/grpo_llama3.2_3b_pposft_0710/global_step_120/fixed_embedding_safetensors"
    config_path = os.path.join(checkpoint_dir, "config.json")
    
    try:
        # Consolidate with embedding-specific logic
        consolidated_state_dict = consolidate_embeddings_correctly(checkpoint_dir)
        
        # Save the model
        safetensors_path = save_with_proper_vocab_size(consolidated_state_dict, output_dir, config_path)
        
        print(f"\n=== Conversion Complete ===")
        print(f"Output: {output_dir}")
        print(f"\nTry loading this model with vLLM to test if the vocab size issue is resolved.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()