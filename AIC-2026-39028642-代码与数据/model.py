from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
from src.null_prior import OwnPrior
BASE=Path(__file__).resolve().parent
DEFAULT_MODEL=BASE.parent/'AIC-2026-39028642-模型文件.pt'
class ContextMLP(nn.Module):

    def __init__(self, base, width, scale):
        super().__init__()
        self.base = base
        self.scale = scale
        self.adapter_norm = nn.LayerNorm(768)
        self.adapter_down = nn.Linear(768, width)
        self.adapter_up = nn.Linear(width, 768)
        self.adapter_gate = nn.Linear(768, width)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.zeros_(self.adapter_gate.weight)
        nn.init.zeros_(self.adapter_gate.bias)

    def forward(self, hidden):
        normalized = self.adapter_norm(hidden)
        gate = torch.sigmoid(self.adapter_gate(normalized[:, 0]))[:, None, :]
        residual = self.adapter_up(F.gelu(self.adapter_down(normalized)) * gate)
        return self.base(hidden) + self.scale * residual
class ProjectionVisual(nn.Module):
    def __init__(self,vision_config,config):
        super().__init__();self.config=config;self.clip=CLIPVisionModelWithProjection(CLIPVisionConfig.from_dict(vision_config));self.head=nn.Parameter(torch.empty(750,512));self.log_scale=nn.Parameter(torch.zeros(()));self.null_head=nn.Parameter(torch.empty(750,256));self.register_buffer('projection_complement',torch.empty(768,256))
        for i in range(config['adapter_first_layer'],12):
            layer=self.clip.vision_model.encoder.layers[i];layer.mlp=ContextMLP(layer.mlp,config['adapter_width'],config['adapter_scale'])
    def forward(self, images):
        visual = self.clip.vision_model(pixel_values=images, interpolate_pos_encoding=True).pooler_output.float()
        features = F.normalize(self.clip.visual_projection(visual).float(), dim=-1)
        extra = F.normalize(visual, dim=-1) @ self.projection_complement
        weights = self.null_head / self.null_head.norm(dim=-1, keepdim=True).clamp_min(1)
        semantic = features @ F.normalize(self.head, dim=-1).T
        detail = extra @ weights.T
        logits = self.log_scale.exp().clamp(1, 100) * (semantic.float() + self.config['projection_complement_scale'] * detail.float())
        return {'logits': logits, 'feat': features, 'detail': detail}
class FisherVision(nn.Module):
    def __init__(self,vision_config,config):
        super().__init__();self.base=ProjectionVisual(vision_config,config);self.register_buffer('axis_prior',torch.empty(256));self.register_buffer('axis_noise',torch.empty(256));self.axis_delta=nn.Parameter(torch.empty(256));self.captured={};self.base.clip.vision_model.register_forward_hook(lambda layer,inputs,output:self.captured.update(visual=output.pooler_output))
    def forward(self, images):
        result = self.base(images)
        self.visual = self.captured['visual']
        visual = F.normalize(self.captured['visual'].float(), dim=-1)
        unused = visual @ self.base.projection_complement
        change = 0.15 * self.axis_delta.float().tanh() * self.axis_prior
        weights = self.base.null_head / self.base.null_head.norm(dim=-1, keepdim=True).clamp_min(1)
        detail_correction = unused * change @ weights.T
        scale = self.base.log_scale.exp().clamp(1, 100) * self.base.config['projection_complement_scale']
        result['logits'] = result['logits'].float() + scale * detail_correction.float()
        return result
class CachedCorrection(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;self.register_buffer('complement',torch.empty(768,256));self.register_buffer('axis_delta',torch.empty(256));self.register_buffer('axis_prior',torch.empty(256));width=config['readout_width'];self.readout_down=nn.Linear(256,width,bias=False);self.readout_norm=nn.LayerNorm(width);self.readout_up=nn.Linear(width,750,bias=False)
    def correction(self, visual, probability):
        visual = F.normalize(visual.float(), dim=-1)
        unused = visual @ self.complement
        axes = 1 + 0.15 * self.axis_delta.detach().float().tanh() * self.axis_prior
        unused = F.normalize(unused * axes, dim=-1)
        if self.training:
            unused = unused + self.config['cached_noise_std'] * torch.randn_like(unused)
        hidden = F.gelu(self.readout_norm(self.readout_down(unused)))
        hidden = F.dropout(hidden, p=self.config['cached_dropout'], training=self.training)
        uncertainty = 1 - probability.detach().float().max(-1).values
        return self.config['readout_bound'] * uncertainty[:, None] * self.readout_up(hidden).float().tanh()
class Wrapper(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def forward(self,images):return self.base(images)
def load_bundle(path=DEFAULT_MODEL):
    d=torch.load(path,map_location='cpu',weights_only=False);assert d['format']=='aic-context-complement-v2' and d['entry']=='AIC-2026-39028642' and d['team']=='算法肥嘟嘟的';return d
def load_visual(path=DEFAULT_MODEL,device='cuda'):
    d=load_bundle(path);m=FisherVision(d['vision_config'],d['architecture']);m.load_state_dict(d['visual_state'],strict=True,assign=True);return Wrapper(m).to(device).eval().requires_grad_(False)
def load_cached(bundle,device='cuda'):
    m=CachedCorrection(bundle['readout_config']);m.load_state_dict(bundle['readout_state'],strict=True,assign=True);return m.to(device).eval().requires_grad_(False)
def load_null(bundle,device='cuda'):
    s=bundle['null_state'];m=OwnPrior({k:v for k,v in s.items() if k!='null_delta'});m.load_state_dict(s,strict=True);return m.to(device).eval()
