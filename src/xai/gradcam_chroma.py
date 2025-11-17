# src/xai/gradcam_chroma.py
import torch
import torch.nn.functional as F
import numpy as np

class GradCAMChroma:
    """
    Grad-CAM for chroma CNNs. Input shape expected: (1,1,12,W)
    Returns heatmap upsampled to (12, W) numpy array normalized [0,1].
    """
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        device = next(self.model.parameters()).device
        x = input_tensor.to(device)
        x.requires_grad = True
        out = self.model(x)
        probs = F.softmax(out, dim=1)
        pred = int(probs.argmax(dim=1).item())
        if target_class is None:
            target_class = pred
        loss = out[0, target_class]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        grads = self.gradients[0]   # (C, h, w)
        acts = self.activations[0]  # (C, h, w)
        weights = grads.mean(dim=(1,2))  # (C,)
        gcam = (weights[:, None, None] * acts).sum(dim=0)  # (h, w)
        gcam = F.relu(gcam)
        gcam_np = gcam.cpu().numpy()
        if gcam_np.max() != gcam_np.min():
            gcam_np = (gcam_np - gcam_np.min()) / (gcam_np.max() - gcam_np.min())
        else:
            gcam_np = np.zeros_like(gcam_np)
        # upsample to input size (12, W)
        gcam_tensor = torch.from_numpy(gcam_np).unsqueeze(0).unsqueeze(0).float()
        up = F.interpolate(gcam_tensor, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        heatmap = up.squeeze().cpu().numpy()
        return heatmap, pred, float(probs[0, pred].cpu().item())
