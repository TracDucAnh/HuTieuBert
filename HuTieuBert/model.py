import torch
from transformers.models.roberta.modeling_roberta import RobertaModel
from .embeddings import BoundaryAwareEmbeddings
from .bias_utils import create_bias_matrix

class MorphemeAwareRobertaModel(RobertaModel):
    """
    PhoBERT mở rộng với:
    - BoundaryAwareEmbeddings (BMES + gate)
    - BMES bias hook trên attention head, hỗ trợ batch
    """

    def __init__(self, config, target_heads=None, alpha=0.1, beta=-0.05, gamma=0.0, delta=0.0, **kwargs):
        super().__init__(config, **kwargs)

        # Embedding mới
        self.embeddings = BoundaryAwareEmbeddings(config, **kwargs)

        # Bias params
        self.target_heads = target_heads or {}
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.tokenizer = None
        self.patched_forwards = {}
        self.bias_matrix = None

    def set_tokenizer(self, tokenizer):
        assert tokenizer is not None
        self.tokenizer = tokenizer

    def set_bias_matrix(self, bmes_tags):
        """
        bmes_tags: tensor [B, seq_len] hoặc [seq_len]
        Trả về tensor [B, num_heads, seq_len, seq_len]
        """
        if isinstance(bmes_tags, torch.Tensor) and bmes_tags.dim() == 1:
            bmes_tags = bmes_tags.unsqueeze(0)

        batch_size, seq_len = bmes_tags.shape
        bias_np = create_bias_matrix(bmes_tags, alpha=self.alpha, beta=self.beta, gamma=self.gamma, delta=self.delta)
        bias_tensor = torch.tensor(bias_np, dtype=torch.float32, device=next(self.parameters()).device)
        num_heads = self.config.num_attention_heads
        bias_tensor = bias_tensor.unsqueeze(1).repeat(1, num_heads, 1, 1)
        self.bias_matrix = bias_tensor

    def _create_patched_forward(self, layer_idx, head_indices, original_forward, attn_module):
        """
        Tạo forward function mới có cộng bias vào attention scores trước softmax
        🔧 FIX: Tương thích với Transformers version mới
        """
        def patched_forward(
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            **kwargs  # 🔧 Catch-all cho các tham số khác
        ):
            batch_size, seq_length = hidden_states.shape[:2]
            
            # 🔧 Tính Q, K, V - sử dụng cách mới
            query_layer = attn_module.query(hidden_states)
            
            # Xử lý key và value
            is_cross_attention = encoder_hidden_states is not None
            
            if is_cross_attention:
                key_layer = attn_module.key(encoder_hidden_states)
                value_layer = attn_module.value(encoder_hidden_states)
            elif past_key_value is not None:
                key_layer = attn_module.key(hidden_states)
                value_layer = attn_module.value(hidden_states)
                # Concatenate với cached keys/values nếu có
                key_layer = torch.cat([past_key_value[0], key_layer], dim=1)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=1)
            else:
                key_layer = attn_module.key(hidden_states)
                value_layer = attn_module.value(hidden_states)
            
            # 🔧 Reshape để split heads - KHÔNG dùng transpose_for_scores
            # Thay vào đó, reshape trực tiếp
            def split_heads(tensor, num_heads, head_dim):
                """
                Splits hidden_size dim into num_heads and head_dim
                tensor shape: (batch_size, seq_length, hidden_size)
                return shape: (batch_size, num_heads, seq_length, head_dim)
                """
                new_shape = tensor.size()[:-1] + (num_heads, head_dim)
                tensor = tensor.view(new_shape)
                return tensor.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
            
            num_heads = attn_module.num_attention_heads
            head_dim = attn_module.attention_head_size
            
            query_layer = split_heads(query_layer, num_heads, head_dim)
            key_layer = split_heads(key_layer, num_heads, head_dim)
            value_layer = split_heads(value_layer, num_heads, head_dim)
            
            # Lưu past_key_value nếu cần (cho decoding)
            if hasattr(attn_module, 'is_decoder') and attn_module.is_decoder:
                past_key_value = (key_layer, value_layer)
            
            # 🔧 Tính attention scores (Q @ K^T)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / torch.sqrt(
                torch.tensor(head_dim, dtype=attention_scores.dtype, device=attention_scores.device)
            )
            
            # ✅ **CỘNG BIAS VÀO ĐÂY - TRƯỚC SOFTMAX**
            if self.bias_matrix is not None:
                B, H, L, _ = attention_scores.shape
                bias = self.bias_matrix
                
                # Crop bias nếu sequence length khác
                if bias.size(0) != B:
                    bias = bias[:B]
                if bias.size(-1) != L:
                    bias = bias[:, :, :L, :L]
                
                # Chỉ cộng vào các head được chỉ định
                for h in head_indices:
                    if h < H:
                        attention_scores[:, h, :, :] = attention_scores[:, h, :, :] + bias[:, h, :, :]
            
            # Cộng attention_mask nếu có
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            # Softmax để tạo attention probabilities
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            
            # Dropout
            attention_probs = attn_module.dropout(attention_probs)
            
            # Apply head_mask nếu có
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            
            # 🔧 Tính context layer
            context_layer = torch.matmul(attention_probs, value_layer)
            # Reshape lại: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (attn_module.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            
            # Prepare outputs
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            
            if hasattr(attn_module, 'is_decoder') and attn_module.is_decoder:
                outputs = outputs + (past_key_value,)
            
            return outputs
        
        return patched_forward

    def _patch_attention_layer(self, layer_idx, head_indices):
        """
        Monkey patch forward method của attention layer
        """
        attn_module = self.encoder.layer[layer_idx].attention.self
        
        if layer_idx not in self.patched_forwards:
            original_forward = attn_module.forward
            self.patched_forwards[layer_idx] = (attn_module, original_forward)
            
            patched_forward = self._create_patched_forward(
                layer_idx, head_indices, original_forward, attn_module
            )
            attn_module.forward = patched_forward

    def prepare_bias_patches(self):
        """
        Patch tất cả các layer có target heads
        """
        self.remove_bias_patches()
        for layer_idx, heads in self.target_heads.items():
            self._patch_attention_layer(layer_idx, heads)

    def remove_bias_patches(self):
        """
        Khôi phục lại original forward methods
        """
        for layer_idx, (attn_module, original_forward) in self.patched_forwards.items():
            attn_module.forward = original_forward
        self.patched_forwards = {}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        bmes_ids=None,
        bmes_tags=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if bmes_ids is None and bmes_tags is not None:
            bmes_ids = bmes_tags

        if bmes_ids is not None:
            self.set_bias_matrix(bmes_ids)

        if self.target_heads:
            self.prepare_bias_patches()

        output_attentions = True if output_attentions is None else output_attentions

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        self.remove_bias_patches()
        return outputs