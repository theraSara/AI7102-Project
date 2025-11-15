import ast
import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaModel

class ConfWeightedTextFeatureExtractor:
    """
    Text feature extractor with token-level confidence-weighted pooling.
    Returns 768-d features (same shape as your current extractor).
    """
    def __init__(self, model_name="roberta-base", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ConfWeighted] Loading RoBERTa (fast) on {self.device}...")

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
        # safetensors loads by default if available in recent transformers; no need to force
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print("[ConfWeighted] RoBERTa loaded:", model_name)
        print("[ConfWeighted] Hidden size:", self.model.config.hidden_size)

    @staticmethod
    def _parse_word_conf(s):
        """
        CSV stores Python-like lists: "[{'word': 'Yes,', 'probability': 0.85}, ...]"
        -> returns list[(word, prob)], empty list if missing/bad.
        """
        if s is None:
            return []
        s = str(s)
        if not s or s.strip() == "" or s.strip().lower() == "nan":
            return []
        try:
            items = ast.literal_eval(s)
            out = []
            for x in items:
                w = str(x.get('word', '')).strip()
                if not w:
                    continue
                p = float(x.get('probability', 1.0))
                out.append((w, p))
            return out
        except Exception:
            return []

    def _conf_weighted_from_words(self, words, probs, max_length=512):
        """
        words: list[str]  (sequence of words for the utterance)
        probs: list[float] aligned with words, each in [0,1]
        returns: torch.Tensor (hidden,)
        """
        # encode one example as a list-of-words for alignment
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**enc)
            hidden = out.last_hidden_state.squeeze(0)  # (seq_len, hidden)

        # map each token to its originating word index
        word_ids = enc.word_ids(batch_index=0)  # list[int|None], length seq_len
        attn = enc["attention_mask"].squeeze(0).to(hidden.dtype)  # (seq_len,)

        # token weights from word probs; special tokens (None) get 0
        w = torch.zeros(hidden.size(0), dtype=hidden.dtype, device=hidden.device)
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if 0 <= wid < len(probs):
                w[i] = float(probs[wid])

        w = w * attn
        denom = w.sum()
        # If a sequence is long and many high‑probability words are truncated, 
        # it’s possible all token weights become 0 and the denominator collapses. 
        # Add a safe fallback to unweighted mean when that happens
        if denom.item() < 1e-6:
            # Fallback: unweighted mean over non-padding tokens
            feat = (hidden * attn.unsqueeze(-1)).sum(dim=0) / attn.sum().clamp(min=1e-6)
            return feat
        else:
            feat = (hidden * w.unsqueeze(-1)).sum(dim=0) / denom
            return feat

    def extract_features(self, text, pooling='cls', word_conf_list=None, utter_conf=None):
        """
        Supported poolings: 'cls', 'mean', 'conf_weighted'
        """
        if not text or not str(text).strip():
            return np.zeros(self.model.config.hidden_size, dtype=np.float32)

        text = str(text).strip()

        if pooling == 'conf_weighted':
            # build words and probs from word_conf_list if available
            words, probs = [], []
            if isinstance(word_conf_list, list) and len(word_conf_list) > 0:
                for w, p in word_conf_list:
                    words.append(str(w))
                    probs.append(float(p))
            if len(words) > 0:
                feat = self._conf_weighted_from_words(words, probs)
                return feat.detach().cpu().numpy().astype(np.float32)

            # fallback: utterance-level weighting over mean
            enc = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
                hs = out.last_hidden_state.squeeze(0)
            attn = enc['attention_mask'].squeeze(0)
            mask = attn.unsqueeze(-1).expand_as(hs)
            s = float(utter_conf) if utter_conf is not None else 1.0
            feat = ((hs * mask) * s).sum(dim=0) / mask.sum(dim=0).clamp(min=1e-9)
            return feat.detach().cpu().numpy().astype(np.float32)

        # --- existing baselines ---
        enc = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
            hs = out.last_hidden_state.squeeze(0)

        if pooling == 'cls':
            feat = hs[0]
        elif pooling == 'mean':
            attn = enc['attention_mask'].squeeze(0)
            mask = attn.unsqueeze(-1).expand_as(hs)
            feat = (hs * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1e-9)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        return feat.detach().cpu().numpy().astype(np.float32)

    def process_dataframe(self, df, text_column='asr_transcript', pooling='cls'):
        feats, empty_idx = [], []
        print(f"[ConfWeighted] Extracting text features with pooling={pooling}")

        use_conf = (pooling == 'conf_weighted')

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Text"):
            txt = row.get(text_column, "")
            if not isinstance(txt, str) or not txt.strip():
                feats.append(np.zeros(self.model.config.hidden_size, dtype=np.float32))
                empty_idx.append(idx)
                continue

            if use_conf:
                wc_raw = self._parse_word_conf(row.get('word_confidences', ''))
                # convert to list[(word, prob)]
                wc = [(w, float(p)) for (w, p) in wc_raw] if isinstance(wc_raw, list) else []
                uc = float(row.get('utterance_confidence', 1.0))
                feat = self.extract_features(txt, pooling='conf_weighted', word_conf_list=wc, utter_conf=uc)
            else:
                feat = self.extract_features(txt, pooling=pooling)

            feats.append(feat)

        if empty_idx:
            print(f"[ConfWeighted] Empty/invalid text for {len(empty_idx)} rows.")
        return np.vstack(feats), empty_idx
