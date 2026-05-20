import logging
import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

logger = logging.getLogger("mlx-gpt")


def save_checkpoint(model, optimizer, step, checkpoint_dir, is_final=False, data_loader=None):
    """Save model weights, optimizer state, step number, and optionally data loader state."""
    checkpoint_name = "final" if is_final else f"checkpoint_step_{step}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save model weights
    model_weights_path = os.path.join(checkpoint_path, "model_weights.safetensors")
    model.save_weights(model_weights_path)

    # Save optimizer state (flatten nested structure)
    optimizer_state_path = os.path.join(checkpoint_path, "optimizer_state.npz")
    flat_state_list = tree_flatten(optimizer.state)
    flat_state = dict(flat_state_list)
    mx.savez(optimizer_state_path, **flat_state)

    # Save step number and other metadata
    metadata_path = os.path.join(checkpoint_path, "metadata.npz")
    metadata = {"step": mx.array(step)}
    
    # Save data loader state if provided
    if data_loader is not None:
        metadata["data_shard"] = mx.array(data_loader.current_shard)
        metadata["data_position"] = mx.array(data_loader.current_position)
    
    mx.savez(metadata_path, **metadata)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_dir, data_loader=None):
    """Load the latest checkpoint if it exists. Returns starting step and data loader state if available."""
    # Look for checkpoints
    if not os.path.exists(checkpoint_dir):
        return 0, None

    # Find all checkpoint directories
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(checkpoint_path):
            metadata_path = os.path.join(checkpoint_path, "metadata.npz")
            if os.path.exists(metadata_path):
                metadata = mx.load(metadata_path)
                step = int(metadata["step"].item())
                checkpoints.append((step, checkpoint_path))

    if not checkpoints:
        return 0, None

    # Load the latest checkpoint (highest step number)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    step, checkpoint_path = checkpoints[0]

    # Load model weights
    model_weights_path = os.path.join(checkpoint_path, "model_weights.safetensors")
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)
        mx.eval(model.parameters())

    # Load optimizer state (unflatten nested structure)
    optimizer_state_path = os.path.join(checkpoint_path, "optimizer_state.npz")
    if os.path.exists(optimizer_state_path):
        flat_state_dict = mx.load(optimizer_state_path)
        # Convert dict to list of tuples for tree_unflatten
        flat_state_list = list(flat_state_dict.items())
        optimizer_state = tree_unflatten(flat_state_list)
        optimizer.state = optimizer_state

    # Load data loader state if available
    data_loader_state = None
    metadata = mx.load(os.path.join(checkpoint_path, "metadata.npz"))
    if "data_shard" in metadata and "data_position" in metadata:
        data_shard = int(metadata["data_shard"].item())
        data_position = int(metadata["data_position"].item())
        data_loader_state = {"shard": data_shard, "position": data_position}
        
        if data_loader is not None:
            data_loader.current_shard = data_shard
            data_loader.current_position = data_position
            # Reload the correct shard
            from dataloader import load_tokens
            data_loader.tokens = load_tokens(data_loader.shards[data_shard])
            logger.info(
                f"Restored data loader state: shard={data_shard}, position={data_position}"
            )

    logger.info(f"Resumed training from checkpoint at step {step} ({checkpoint_path})")

    if data_loader_state:
        data_shard = data_loader_state["shard"]
        data_position = data_loader_state["position"]
        print(
            f"\n══ 從 checkpoint 恢復 ══ step {step} | "
            f"shard {data_shard}/{len(data_loader.shards)-1} | "
            f"position {data_position} ══\n"
        )
    else:
        print(f"\n══ 從 checkpoint 恢復 ══ step {step} ══\n")

    return step + 1, data_loader_state  # Return next step to start from
