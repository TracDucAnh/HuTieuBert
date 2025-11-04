import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from HuTieuBert.tokenizer import MorphemeAwareTokenizer
from HuTieuBert.embeddings import BoundaryAwareEmbeddings
from HuTieuBert.model import MorphemeAwareRobertaModel
from HuTieuBert.bias_utils import create_bias_matrix
from transformers import RobertaConfig
from matplotlib.gridspec import GridSpec

# =============================================================================
# Khởi tạo tokenizer
# =============================================================================
tokenizer = MorphemeAwareTokenizer(
    vncorenlp_dir=os.path.abspath("vncorenlp"),
    return_tensors="pt"
)

text = "Thành phố Hồ Chí Minh."
output = tokenizer(text, return_tensors="pt")

# =============================================================================
# Cấu hình bias cho nhiều layer
# =============================================================================
config = RobertaConfig.from_pretrained("vinai/phobert-base")

layers_to_visualize = [5, 6, 7]   # <-- tùy chỉnh danh sách layer ở đây
heads = list(range(12))        # dùng all heads

alpha_config = 0.5
beta_config = -0.25
gamma_config = 0.0
delta_config = 0.0

target_heads_with_bias = {layer: heads for layer in layers_to_visualize}

if len(heads) == config.num_attention_heads:
    heads_label = "all heads"
    heads_short = "all"
elif len(heads) == 1:
    heads_label = f"head {heads[0]}"
    heads_short = f"h{heads[0]}"
else:
    heads_label = f"heads {', '.join(map(str, heads))}"
    heads_short = f"h{'-'.join(map(str, heads))}"

print(f"\n🎯 Configuration: Layers {layers_to_visualize}, {heads_label}")

# =============================================================================
# Model WITH BIAS
# =============================================================================
model_bias = MorphemeAwareRobertaModel.from_pretrained(
    "vinai/phobert-base",
    config=config,
    target_heads=target_heads_with_bias,
    alpha=alpha_config,
    beta=beta_config,
    gamma=gamma_config,
    delta=delta_config,
    attn_implementation="eager"
)
model_bias.eval()

with torch.no_grad():
    outputs_bias = model_bias(
        input_ids=output['input_ids'],
        attention_mask=output['attention_mask'],
        bmes_tags=output['bmes_tags'],
        output_attentions=True
    )

# =============================================================================
# Model NO BIAS
# =============================================================================
model_no_bias = MorphemeAwareRobertaModel.from_pretrained(
    "vinai/phobert-base",
    config=config,
    target_heads=None,
    alpha=alpha_config,
    beta=beta_config,
    gamma=gamma_config,
    delta=delta_config,
    attn_implementation="eager"
)
model_no_bias.eval()

with torch.no_grad():
    outputs_no_bias = model_no_bias(
        input_ids=output['input_ids'],
        attention_mask=output['attention_mask'],
        bmes_tags=output['bmes_tags'],
        output_attentions=True
    )

# =============================================================================
# Chuẩn bị dữ liệu attention cho từng layer
# =============================================================================
attentions_bias = outputs_bias.attentions
attentions_no_bias = outputs_no_bias.attentions
tokens = tokenizer.convert_ids_to_tokens(output['input_ids'][0].tolist())

# =============================================================================
# Vẽ figure với nhiều hàng (mỗi hàng là 1 layer)
# =============================================================================
num_layers = len(layers_to_visualize)
fig_height = 6 * num_layers  # mỗi layer chiếm ~6 inch chiều cao

fig = plt.figure(figsize=(22, fig_height))
gs = GridSpec(num_layers, 3, figure=fig, wspace=0.35, hspace=0.45)

def draw_heatmap(ax, matrix, title, cmap, vmin=None, vmax=None, center=None, label='Weight'):
    sns.heatmap(
        matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap=cmap,
        annot=True,
        fmt='.3f',
        ax=ax,
        cbar_kws={'label': label, 'shrink': 0.85},
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=True,
        linewidths=0.3,
        linecolor='white'
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Key Tokens', fontsize=10, fontweight='bold')
    ax.set_ylabel('Query Tokens', fontsize=10, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

# Lặp qua từng layer và vẽ 3 cột
for row, layer in enumerate(layers_to_visualize):
    attention_layer_bias = attentions_bias[layer][0].mean(dim=0).cpu().numpy()
    attention_layer_no_bias = attentions_no_bias[layer][0].mean(dim=0).cpu().numpy()
    difference = attention_layer_bias - attention_layer_no_bias
    vmax_shared = max(attention_layer_bias.max(), attention_layer_no_bias.max())

    # Cột 1: WITH BIAS
    ax1 = fig.add_subplot(gs[row, 0])
    draw_heatmap(ax1, attention_layer_bias, f"Layer {layer} — WITH BIAS ({heads_label})", "YlOrRd", 0, vmax_shared)

    # Cột 2: NO BIAS
    ax2 = fig.add_subplot(gs[row, 1])
    draw_heatmap(ax2, attention_layer_no_bias, f"Layer {layer} — NO BIAS", "Blues", 0, vmax_shared)

    # Cột 3: DIFFERENCE
    ax3 = fig.add_subplot(gs[row, 2])
    draw_heatmap(ax3, difference, f"Layer {layer} — DIFFERENCE", "RdBu_r", center=0, label="Δ Weight")

# =============================================================================
# Lưu file ảnh tổng hợp
# =============================================================================
figs_dir = os.path.abspath(os.path.join("..", "figs"))
os.makedirs(figs_dir, exist_ok=True)

save_path = os.path.join(figs_dir, f"multi_layer_attention_{'-'.join(map(str, layers_to_visualize))}_{heads_short}.png")
fig.suptitle(f"Multi-Layer Attention Comparison ({heads_label})", fontsize=16, fontweight='bold', y=0.99)
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"\n✅ Saved combined figure: {save_path}")
print("="*70)
print("📊 SUMMARY STATS PER LAYER")
print("="*70)

# In thống kê cho từng layer
for layer in layers_to_visualize:
    bias_mat = attentions_bias[layer][0].mean(dim=0).cpu().numpy()
    nobias_mat = attentions_no_bias[layer][0].mean(dim=0).cpu().numpy()
    diff = bias_mat - nobias_mat
    print(f"Layer {layer:2d} | WITH BIAS mean={bias_mat.mean():.4f} | NO BIAS mean={nobias_mat.mean():.4f} | Δ mean={diff.mean():+.4f}")

print("="*70)
print(f"📁 Saved figure at: {save_path}")
print("="*70)


print("="*70)
print("📊 EXTENDED ATTENTION ANALYSIS PER LAYER")
print("="*70)

for layer in layers_to_visualize:
    bias_mat = attentions_bias[layer][0].mean(dim=0).cpu().numpy()
    nobias_mat = attentions_no_bias[layer][0].mean(dim=0).cpu().numpy()
    diff = bias_mat - nobias_mat

    # === Phân tích cơ bản, không dùng scipy ===
    mean_bias = bias_mat.mean()
    mean_nobias = nobias_mat.mean()
    mean_diff = diff.mean()
    abs_diff_mean = np.abs(diff).mean()
    std_diff = diff.std()

    # Tương quan thủ công: dùng công thức Pearson đơn giản
    flat_bias = bias_mat.flatten()
    flat_nobias = nobias_mat.flatten()
    corr = np.corrcoef(flat_bias, flat_nobias)[0, 1]

    # Tỷ lệ phần tử thay đổi dấu (attention đảo hướng)
    sign_change_ratio = np.mean(np.sign(flat_bias) != np.sign(flat_nobias))

    # In kết quả
    print(f"\n📍 Layer {layer}")
    print(f"  • Mean (Bias)     = {mean_bias:.6f}")
    print(f"  • Mean (No Bias)  = {mean_nobias:.6f}")
    print(f"  • Δ Mean          = {mean_diff:+.6f}")
    print(f"  • |Δ| Mean        = {abs_diff_mean:.6f}")
    print(f"  • Std(Δ)          = {std_diff:.6f}")
    print(f"  • Corr(Bias,NoBias)= {corr:.6f}")
    print(f"  • Sign change %   = {sign_change_ratio*100:.2f}%")

print("="*70)
print("✅ Basic analysis done.")
print("="*70)
