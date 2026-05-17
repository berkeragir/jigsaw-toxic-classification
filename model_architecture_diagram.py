"""
Multi-Task DistilBERT Architecture Visualization
Creates a detailed diagram of the model architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Colors
color_input = '#E3F2FD'
color_embedding = '#BBDEFB'
color_transformer = '#90CAF9'
color_pooling = '#64B5F6'
color_heads = '#42A5F5'
color_output = '#1E88E5'

# Helper function to draw boxes
def draw_box(ax, x, y, width, height, text, color, fontsize=10, fontweight='normal'):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='black',
                         facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color='black',
                           linewidth=2,
                           mutation_scale=20)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Title
ax.text(5, 19.5, 'Multi-Task DistilBERT Architecture',
        ha='center', fontsize=16, fontweight='bold')
ax.text(5, 19, 'for Toxic Comment Classification',
        ha='center', fontsize=12, style='italic')

# ============================================================
# INPUT LAYER
# ============================================================
y_pos = 17.5
draw_box(ax, 1, y_pos, 8, 0.8,
         'INPUT: "You are an idiot!"',
         color_input, fontsize=11, fontweight='bold')

# Tokenization
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.7, '[CLS] you are an idiot ! [SEP]')

# ============================================================
# TOKENIZATION
# ============================================================
y_pos = 16.2
draw_box(ax, 1, y_pos, 8, 0.8,
         'Tokenized IDs: [101, 2017, 2024, 2019, 13159, 999, 102]',
         color_input, fontsize=9)

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.7)

# ============================================================
# EMBEDDING LAYER
# ============================================================
y_pos = 14.8
draw_box(ax, 1.5, y_pos, 7, 1.2,
         'Embedding Layer (DistilBERT)\n' +
         '• Token Embeddings (30,522 vocab → 768 dim)\n' +
         '• Position Embeddings (0-511 positions)\n' +
         'Output: [batch, seq_len, 768]',
         color_embedding, fontsize=9)

ax.text(0.3, y_pos + 0.6, '23M\nparams', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.9)

# ============================================================
# TRANSFORMER LAYERS
# ============================================================
y_pos = 13.3
ax.text(5, y_pos + 0.5, 'DistilBERT Transformer Blocks (6 layers)',
        ha='center', fontsize=11, fontweight='bold')

# Draw 6 transformer blocks
for i in range(6):
    block_y = y_pos - (i * 1.4)

    # Main transformer block
    draw_box(ax, 1.5, block_y - 1, 7, 1.2,
             f'Transformer Layer {i+1}\n' +
             '• Multi-Head Attention (12 heads)\n' +
             '• Feed-Forward Network (768 → 3072 → 768)\n' +
             '• LayerNorm + Residual Connections',
             color_transformer, fontsize=8)

    # Parameter count
    ax.text(0.3, block_y - 0.4, '7M\nparams', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Arrow between blocks
    if i < 5:
        draw_arrow(ax, 5, block_y - 1, 5, block_y - 1.4)

# Final output shape
y_pos = y_pos - 8.4
ax.text(5, y_pos - 0.3, 'Shape: [batch, seq_len, 768]',
        ha='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

draw_arrow(ax, 5, y_pos - 0.5, 5, y_pos - 1.2)

# ============================================================
# POOLING / CLS TOKEN EXTRACTION
# ============================================================
y_pos = y_pos - 1.5
draw_box(ax, 2, y_pos, 6, 0.8,
         '[CLS] Token Extraction\n' +
         'Take first token: outputs[:, 0, :]',
         color_pooling, fontsize=9)

ax.text(5, y_pos - 0.5, 'Shape: [batch, 768]',
        ha='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ============================================================
# CLASSIFICATION HEADS (Multi-task)
# ============================================================
y_pos = y_pos - 1.5
ax.text(5, y_pos, 'Multi-Task Classification Heads (6 separate heads)',
        ha='center', fontsize=11, fontweight='bold')

# Draw arrows to each head
head_positions = []
for i in range(6):
    head_x = 1 + i * 1.4
    head_y = y_pos - 1.5
    head_positions.append((head_x, head_y))
    draw_arrow(ax, 5, y_pos - 0.3, head_x + 0.6, head_y + 1.2, style='->')

# Draw each classification head
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for i, (label, (hx, hy)) in enumerate(zip(labels, head_positions)):
    # Head architecture
    draw_box(ax, hx, hy, 1.2, 0.8,
             f'{label}\nHead',
             color_heads, fontsize=7, fontweight='bold')

    # Show internal layers
    ax.text(hx + 0.6, hy - 0.4,
            'Dropout(0.1)\n↓\nLinear(768→768)\n↓\nReLU\n↓\nDropout(0.1)\n↓\nLinear(768→1)',
            ha='center', fontsize=6,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Parameter count
    ax.text(hx + 0.6, hy - 1.8, '590K\nparams', fontsize=6,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Arrow to output
    draw_arrow(ax, hx + 0.6, hy - 2, hx + 0.6, hy - 2.8)

# ============================================================
# OUTPUTS
# ============================================================
y_pos = y_pos - 5
ax.text(5, y_pos + 0.3, 'Outputs (Logits)',
        ha='center', fontsize=11, fontweight='bold')

# Output boxes
for i, label in enumerate(labels):
    out_x = 1 + i * 1.4
    draw_box(ax, out_x, y_pos - 0.5, 1.2, 0.4,
             f'{label}\nlogit',
             color_output, fontsize=7)

# Final shape
ax.text(5, y_pos - 1.2, 'Shape: [batch, 6]',
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ============================================================
# LOSS FUNCTION
# ============================================================
y_pos = y_pos - 2
draw_box(ax, 2.5, y_pos, 5, 0.8,
         'Focal Loss with Class Weights\n' +
         'Sigmoid → Binary Cross-Entropy + Focal Factor',
         '#FFE0B2', fontsize=9)

draw_arrow(ax, 5, y_pos - 0.8, 5, y_pos - 1.5, 'Backpropagation')

# ============================================================
# SUMMARY STATISTICS
# ============================================================
y_pos = 0.5
summary_text = """
MODEL STATISTICS:
├─ Total Parameters: 69,538,310 (~70M)
│  ├─ DistilBERT: 66,362,880 (66M)
│  │  ├─ Embeddings: 23,835,648 (23M)
│  │  └─ Transformers: 42,527,232 (42M) [6 layers × 7M each]
│  └─ Classification Heads: 3,175,430 (3M) [6 heads × 590K each]
│
├─ Model Size (FP32): ~265 MB
├─ Model Size (FP16): ~133 MB
│
├─ Input: Text sequences (max 512 tokens)
├─ Output: 6 toxicity probabilities [0-1]
│
└─ Training: Multi-task learning with Focal Loss
"""

ax.text(0.2, y_pos, summary_text, fontsize=8, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9,
                  edgecolor='black', linewidth=2),
        verticalalignment='bottom')

plt.tight_layout()
plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved as 'model_architecture.png'")
plt.show()
