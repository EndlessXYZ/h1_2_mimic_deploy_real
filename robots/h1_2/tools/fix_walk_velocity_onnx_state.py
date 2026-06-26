#!/usr/bin/env python3
"""
Fix WalkVelocity ONNX model: expose LSTM hidden/cell state as inputs/outputs.

The original model has hidden_state and cell_state as initializers (zeros).
Each inference call resets the LSTM state to zero, losing temporal memory.
This fix moves them to graph inputs/outputs, enabling state carry-over.

Usage:
  python fix_walk_velocity_onnx_state.py \
      --input policy.onnx \
      --output policy_fixed.onnx
"""
import onnx
from onnx import helper, TensorProto
import numpy as np
import argparse
import shutil


def fix_onnx_state(input_path, output_path):
    """Move LSTM hidden/cell state from initializers to graph inputs/outputs."""
    
    model = onnx.load(input_path)
    graph = model.graph
    
    # Find the LSTM node
    lstm_node = None
    for node in graph.node:
        if node.op_type == 'LSTM':
            lstm_node = node
            break
    
    if lstm_node is None:
        print("ERROR: No LSTM node found")
        return False
    
    # LSTM outputs: Y_all, Y_h (final hidden), Y_c (final cell)
    lstm_y_h = lstm_node.output[1]  # hidden state output
    lstm_y_c = lstm_node.output[2]  # cell state output
    
    # Find the initializer for hidden_state and cell_state
    hidden_init = None
    cell_init = None
    for init in graph.initializer:
        if init.name == 'hidden_state':
            hidden_init = init
        if init.name == 'cell_state':
            cell_init = init
    
    if hidden_init is None or cell_init is None:
        print("ERROR: hidden_state or cell_state initializer not found")
        return False
    
    # Get shapes
    hidden_shape = list(hidden_init.dims)  # [1, 1, 64]
    cell_shape = list(cell_init.dims)      # [1, 1, 64]
    action_shape = None
    for out in graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Current output '{out.name}': shape={shape}")
        action_shape = shape
    
    print(f"  hidden_state shape: {hidden_shape}")
    print(f"  cell_state shape: {cell_shape}")
    
    # 1. Add hidden_state and cell_state as graph inputs
    hidden_input = helper.make_tensor_value_info(
        'hidden_state_input', TensorProto.FLOAT, hidden_shape
    )
    cell_input = helper.make_tensor_value_info(
        'cell_state_input', TensorProto.FLOAT, cell_shape
    )
    
    # Check if they already exist as inputs
    existing_input_names = {i.name for i in graph.input}
    if 'hidden_state_input' not in existing_input_names:
        graph.input.extend([hidden_input])
        print("  Added hidden_state_input to graph inputs")
    if 'cell_state_input' not in existing_input_names:
        graph.input.extend([cell_input])
        print("  Added cell_state_input to graph inputs")
    
    # 2. Add LSTM hidden/cell outputs as graph outputs
    hidden_output = helper.make_tensor_value_info(
        lstm_y_h, TensorProto.FLOAT, hidden_shape
    )
    cell_output = helper.make_tensor_value_info(
        lstm_y_c, TensorProto.FLOAT, cell_shape
    )
    
    existing_output_names = {o.name for o in graph.output}
    if lstm_y_h not in existing_output_names:
        graph.output.extend([hidden_output])
        print(f"  Added '{lstm_y_h}' to graph outputs (hidden state)")
    if lstm_y_c not in existing_output_names:
        graph.output.extend([cell_output])
        print(f"  Added '{lstm_y_c}' to graph outputs (cell state)")
    
    # 3. Remove hidden_state and cell_state from initializers
    new_initializers = []
    for init in graph.initializer:
        if init.name not in ['hidden_state', 'cell_state']:
            new_initializers.append(init)
    
    # Clear and re-add
    while len(graph.initializer) > 0:
        graph.initializer.pop()
    graph.initializer.extend(new_initializers)
    print("  Removed hidden_state and cell_state from initializers")
    
    # 4. Rename the LSTM input references to use the new graph input names
    #    hidden_state -> hidden_state_input
    #    cell_state -> cell_state_input
    lstm_inputs = list(lstm_node.input)
    # LSTM inputs: [X, W, R, B, sequence_lens, initial_h, initial_c]
    # Index 5 = initial_h, Index 6 = initial_c
    if len(lstm_inputs) > 5 and lstm_inputs[5] == 'hidden_state':
        lstm_inputs[5] = 'hidden_state_input'
    if len(lstm_inputs) > 6 and lstm_inputs[6] == 'cell_state':
        lstm_inputs[6] = 'cell_state_input'
    
    # Replace the node
    new_node = helper.make_node(
        'LSTM',
        inputs=lstm_inputs,
        outputs=list(lstm_node.output),
        name=lstm_node.name,
        hidden_size=hidden_shape[2]
    )
    
    # Replace in graph
    for i, node in enumerate(graph.node):
        if node.name == lstm_node.name:
            graph.node.remove(node)
            graph.node.insert(i, new_node)
            break
    
    # 5. Verify the modified model
    onnx.checker.check_model(model)
    
    # 6. Save
    onnx.save(model, output_path)
    print(f"\nSaved fixed model to: {output_path}")
    
    # 7. Print final model I/O
    print("\n=== Fixed model inputs ===")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: shape={shape}")
    print("=== Fixed model outputs ===")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: shape={shape}")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input ONNX model path')
    parser.add_argument('--output', required=True, help='Output ONNX model path')
    args = parser.parse_args()
    
    fix_onnx_state(args.input, args.output)
