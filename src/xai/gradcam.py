# src/xai/gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        device = next(self.model.parameters()).device
        x = input_tensor.to(device)
        x.requires_grad = True
        out = self.model(x)  # (1, n_classes)
        probs = F.softmax(out, dim=1)
        pred_class = int(probs.argmax(dim=1).item())
        if target_class is None:
            target_class = pred_class
        loss = out[0, target_class]
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        grads = self.gradients[0]   # (C, H', W')
        acts = self.activations[0]  # (C, H', W')
        weights = grads.mean(dim=(1,2))  # (C,)
        gcam = (weights[:, None, None] * acts).sum(dim=0)  # (H', W')
        gcam = F.relu(gcam)
        gcam_np = gcam.cpu().numpy()
        if gcam_np.max() != gcam_np.min():
            gcam_np = (gcam_np - gcam_np.min()) / (gcam_np.max() - gcam_np.min())
        else:
            gcam_np = np.zeros_like(gcam_np)
        # upsample to input size
        gcam_tensor = torch.from_numpy(gcam_np).unsqueeze(0).unsqueeze(0).float()
        up = F.interpolate(gcam_tensor, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        heatmap = up.squeeze().cpu().numpy()
        return heatmap, pred_class, float(probs[0, pred_class].cpu().item())

def overlay_and_save(spectrogram, heatmap, out_path, alpha=0.5, cmap='jet', figsize=(8,4)):
    """
    spectrogram: (n_mels, n_frames) - dB or normalized
    heatmap: same shape in [0,1]
    """
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.axis('off')
    ax.imshow(spectrogram, origin='lower', aspect='auto')
    ax.imshow(heatmap, origin='lower', aspect='auto', cmap=cmap, alpha=alpha)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
