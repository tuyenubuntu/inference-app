import onnx

def get_onnx_model_info(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    
    # Lấy thông tin cơ bản
    print("=== Model Information ===")
    print(f"Producer: {model.producer_name}")
    print(f"Version: {model.producer_version}")
    print(f"Ir Version: {model.ir_version}")
    print(f"Opset Version: {model.opset_import[0].version}")
    print(f"Model Doc String: {model.doc_string}")
    print()
    
    # Lấy thông tin input
    print("=== Inputs ===")
    for input_tensor in model.graph.input:
        print(f"Name: {input_tensor.name}")
        tensor_type = input_tensor.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else "Dynamic")
        print(f"  Shape: {shape}")
        print(f"  Data Type: {onnx.TensorProto.DataType.Name(tensor_type.elem_type)}")
    print()
    
    # Lấy thông tin output
    print("=== Outputs ===")
    for output_tensor in model.graph.output:
        print(f"Name: {output_tensor.name}")
        tensor_type = output_tensor.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else "Dynamic")
        print(f"  Shape: {shape}")
        print(f"  Data Type: {onnx.TensorProto.DataType.Name(tensor_type.elem_type)}")
    print()
    
    # Lấy các thông tin khác (Layers)
    print("=== Nodes (Operators) ===")
    for node in model.graph.node[:5]:  # In ra 5 node đầu tiên
        print(f"Op Type: {node.op_type}")
        print(f"Inputs: {node.input}")
        print(f"Outputs: {node.output}")
        print()
    print(f"... Total Nodes: {len(model.graph.node)}")

# Đường dẫn đến model ONNX
model_path = "model/aoiai_detectnet_v1.onnx"

# Gọi hàm lấy thông tin
get_onnx_model_info(model_path)
