# confidence_gate.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceGateMLP(nn.Module):
    def __init__(self, hidden_dim=256, gate_hidden=128, input_extra_dims=0):
        super().__init__()
        input_size = hidden_dim * 2 + input_extra_dims
        self.gate_network = nn.Sequential(
            nn.Linear(input_size, gate_hidden),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(gate_hidden, gate_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(gate_hidden // 2, 2)
        )
        for m in self.gate_network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        logits = self.gate_network(x)
        tau = torch.exp(self.log_tau)
        gates = F.softmax(logits / tau.clamp(min=0.5, max=2.0), dim=-1)
        return gates, logits

class ConfidenceGate(nn.Module):
    def __init__(self, hidden_dim=256, gate_hidden=128,
                 use_aux_loss=True, lambda_gate=0.1,
                 use_conf_in_gate=True, use_cosine_agreement=True):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.lambda_gate = lambda_gate
        self.use_conf_in_gate = use_conf_in_gate
        self.use_cosine_agreement = use_cosine_agreement

        # self.conf_temp = nn.Parameter(torch.tensor(1.0))
        self.conf_temp = nn.Parameter(torch.tensor(0.5))
        self.conf_bias = nn.Parameter(torch.tensor(0.0))
        extra = (1 if use_conf_in_gate else 0) + (1 if use_cosine_agreement else 0)
        self.mlp = ConfidenceGateMLP(hidden_dim, gate_hidden, input_extra_dims=extra)

    def forward(self, audio_proj, text_proj, logit_conf, confidence_original=None):
        parts = [audio_proj, text_proj]
        if self.use_conf_in_gate:
            #scl = self.conf_temp * logit_conf + self.conf_bias
            scl = self.conf_temp * logit_conf
            if scl.dim() == 1: scl = scl.unsqueeze(1)
            parts.append(scl)
        if self.use_cosine_agreement:
            cos_agree = F.cosine_similarity(audio_proj, text_proj, dim=-1, eps=1e-6).unsqueeze(1)
            parts.append(cos_agree)

        x = torch.cat(parts, dim=1)
        gates, _ = self.mlp(x)
        #gates, _ = self.mlp(audio_proj, text_proj, scl, cos_agree)


        g_a, g_t = gates[:, 0:1], gates[:, 1:2]
        fused = g_a * audio_proj + g_t * text_proj

        aux_loss = torch.tensor(0.0, device=fused.device)
        if self.use_aux_loss and self.use_conf_in_gate and (confidence_original is not None):
            conf = confidence_original
            if conf.dim() > 1: conf = conf.squeeze(-1)
            aux_loss = self.lambda_gate * F.mse_loss(g_t.squeeze(-1), conf)

        return {
            'fused': fused,
            'gates': gates,
            'gate_audio': g_a.squeeze(1),
            'gate_text': g_t.squeeze(1),
            'aux_loss': aux_loss
        }
