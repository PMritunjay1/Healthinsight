import os
import torch
import torch.nn as nn
from transformers import AutoModel
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

class CardiologistSpecialistONNXWrapper(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.transformer = trained_model.transformer
        self.w_dis1 = nn.Parameter(trained_model.fc_dis1.weight.t().clone())
        self.b_dis1 = nn.Parameter(trained_model.fc_dis1.bias.clone())
        self.w_dis2 = nn.Parameter(trained_model.fc_dis2.weight.t().clone())
        self.b_dis2 = nn.Parameter(trained_model.fc_dis2.bias.clone())
        
        self.w_comp1 = nn.Parameter(trained_model.fc_comp1.weight.t().clone())
        self.b_comp1 = nn.Parameter(trained_model.fc_comp1.bias.clone())
        self.w_comp2 = nn.Parameter(trained_model.fc_comp2.weight.t().clone())
        self.b_comp2 = nn.Parameter(trained_model.fc_comp2.bias.clone())
        
        self.w_urg1 = nn.Parameter(trained_model.fc_urg1.weight.t().clone())
        self.b_urg1 = nn.Parameter(trained_model.fc_urg1.bias.clone())
        self.w_urg2 = nn.Parameter(trained_model.fc_urg2.weight.t().clone())
        self.b_urg2 = nn.Parameter(trained_model.fc_urg2.bias.clone())

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs[0][:, 0, :].reshape(-1, self.transformer.config.hidden_size)
        d = torch.matmul(torch.relu(torch.matmul(cls_rep, self.w_dis1) + self.b_dis1), self.w_dis2) + self.b_dis2
        c = torch.matmul(torch.relu(torch.matmul(cls_rep, self.w_comp1) + self.b_comp1), self.w_comp2) + self.b_comp2
        u = torch.matmul(torch.relu(torch.matmul(cls_rep, self.w_urg1) + self.b_urg1), self.w_urg2) + self.b_urg2
        return d, c, u

def export_and_quantize(model, fp32_output_path, int8_output_path, max_len=384):
    model.eval()
    onnx_model = CardiologistSpecialistONNXWrapper(model)
    onnx_model.eval()
    
    dummy_ids = torch.ones(1, max_len, dtype=torch.long)
    dummy_mask = torch.ones(1, max_len, dtype=torch.long)
    
    os.makedirs(os.path.dirname(fp32_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(int8_output_path), exist_ok=True)
    
    torch.onnx.export(
        onnx_model,
        (dummy_ids, dummy_mask),
        fp32_output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["disease_logits", "complaint_logits", "urgency_logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "attention_mask": {0: "batch_size", 1: "sequence_length"}},
        opset_version=18
    )
    print(f"Exported FP32 ONNX to {fp32_output_path}")
    
    quantize_dynamic(model_input=fp32_output_path, model_output=int8_output_path, weight_type=QuantType.QUInt8)
    print(f"Exported Quantized INT8 ONNX to {int8_output_path}")

if __name__ == "__main__":
    print("export_onnx module ready.")
