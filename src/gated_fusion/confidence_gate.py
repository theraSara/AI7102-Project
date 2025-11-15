import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceGateMLP(nn.Module):
    """
    Gate MLP: [ã ; t̃ ; scaled_logit(c) ; (optional extras)] → [g_a, g_t]
    - Supports an extra scalar (e.g., cosine agreement) without changing the interface.
    """
    def __init__(self, hidden_dim=256, gate_hidden=128):
        super().__init__()
        input_size = hidden_dim * 2 + 2

        self.gate_network = nn.Sequential(
            nn.Linear(input_size, gate_hidden),
            nn.GELU(),           # (GELU tends to work better than ReLU here)
            nn.Dropout(0.30),
            nn.Linear(gate_hidden, gate_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(gate_hidden // 2, 2)  # logits for [g_a, g_t]
        )
        for m in self.gate_network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, audio_proj, text_proj, scaled_logit_conf, cos_agree):
                # ensure shapes (B,1)
        if scaled_logit_conf.dim() == 1: scaled_logit_conf = scaled_logit_conf.unsqueeze(1)
        if cos_agree.dim() == 1:        cos_agree        = cos_agree.unsqueeze(1)

        gate_input = torch.cat([audio_proj, text_proj, scaled_logit_conf, cos_agree], dim=1)  # (B,514)
        gate_logits = self.gate_network(gate_input)                  # (B,2)
        gates = F.softmax(gate_logits, dim=-1)                       # (B,2)
        return gates, gate_logits

class ConfidenceGate(nn.Module):
    """
    Confidence‑gated fusion:
      inputs:  ã, t̃, logit(c)
      extras:  cosine agreement cos(ã, t̃) (optional)
      output:  h = g_a·ã + g_t·t̃
      aux:     λ * MSE(g_t, c)
    """
    def __init__(self, hidden_dim=256, gate_hidden=128, use_aux_loss=True, lambda_gate=0.1,):
        super().__init__()
        # Learnable temperature for scaling logit(conf): logit(conf) / τ  (we parameterize as mul)
        self.conf_temp = nn.Parameter(torch.tensor(0.5))  # scale on logit(c)
        self.use_aux_loss = use_aux_loss
        self.lambda_gate = lambda_gate

        self.gate_mlp = ConfidenceGateMLP(
            hidden_dim=hidden_dim,
            gate_hidden=gate_hidden,
        )

        # cache for analysis
        self.last_gates = None

    def forward(self, audio_proj, text_proj, logit_conf, confidence_original=None):
        """
        audio_proj: (B,256)  ã
        text_proj:  (B,256)  t̃
        logit_conf: (B,) or (B,1)  logit(c)
        confidence_original: (B,) or (B,1) in [0,1]  (for aux loss)
        """
        # Scale logit(conf) by learnable temperature
        scaled_logit_conf = self.conf_temp * logit_conf  # supports (B,) or (B,1)

        # cosine agreement scalar
        cos_agree = F.cosine_similarity(audio_proj, text_proj, dim=-1, eps=1e-6)  # (B,)

        # get gates using the augmented input
        gates, _ = self.gate_mlp(audio_proj, text_proj, scaled_logit_conf, cos_agree)

        g_a = gates[:, 0:1]  # (B,1)
        g_t = gates[:, 1:2]  # (B,1)

        # Fuse
        fused = g_a * audio_proj + g_t * text_proj  # (B,256)

        # Aux loss (optional): encourage g_t ≈ c
        aux_loss = torch.tensor(0.0, device=fused.device)
        if self.use_aux_loss and (confidence_original is not None):
            conf = confidence_original
            if conf.dim() > 1:
                conf = conf.squeeze(-1)
            aux_loss = self.lambda_gate * F.mse_loss(g_t.squeeze(-1), conf)

        self.last_gates = gates.detach()  # for analysis

        return {
            'fused': fused,
            'gates': gates,
            'gate_audio': g_a.squeeze(1),
            'gate_text': g_t.squeeze(1),
            'aux_loss': aux_loss
        }
