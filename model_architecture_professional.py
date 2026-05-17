"""
Professional Multi-Task DistilBERT Architecture Visualization
Publication-quality diagram with proper spacing and alignment
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set professional style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.5

# Create figure with proper size
fig = plt.figure(figsize=(16, 20), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(0, 24)
ax.axis('off')

# Professional color palette
colors = {
    'input': '#E8EAF6',      # Light indigo
    'embedding': '#C5CAE9',   # Indigo 200
    'transformer': '#9FA8DA', # Indigo 300
    'attention': '#7986CB',   # Indigo 400
    'pooling': '#5C6BC0',     # Indigo 500
    'heads': '#3F51B5',       # Indigo 600
    'output': '#3949AB',      # Indigo 700
    'loss': '#FFE0B2',        # Orange 100
    'params': '#FFF9C4',      # Yellow 100
    'shape': '#E1F5FE',       # Light blue 50
}

def draw_box(x, y, width, height, text, facecolor, edgecolor='black',
             linewidth=2, alpha=1.0, fontsize=9, fontweight='normal',
             text_color='black', padding=0.05):
    """Draw a professional rounded box with text"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad={padding}",
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=2
    )
    ax.add_patch(box)

    # Add text with proper wrapping
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            color=text_color, zorder=3)
    return box

def draw_arrow(x1, y1, x2, y2, label='', color='black', linewidth=2, style='-|>'):
    """Draw professional arrow with optional label"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        mutation_scale=25,
        zorder=1
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='white',
                         edgecolor='gray',
                         linewidth=1),
                ha='center', va='center', zorder=4)

def draw_param_badge(x, y, text, size=0.5):
    """Draw parameter count badge"""
    circle = plt.Circle((x, y), size/2,
                        color=colors['params'],
                        ec='orange',
                        linewidth=1.5,
                        zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=7, fontweight='bold', zorder=6)

def draw_shape_label(x, y, text):
    """Draw shape information label"""
    ax.text(x, y, text, fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.4',
                     facecolor=colors['shape'],
                     edgecolor='steelblue',
                     linewidth=1.5),
            ha='center', va='center', zorder=4)

# ============================================================
# TITLE
# ============================================================
ax.text(8, 23.2, 'Multi-Task DistilBERT Architecture',
        ha='center', fontsize=18, fontweight='bold')
ax.text(8, 22.7, 'Toxic Comment Classification with 6 Independent Classification Heads',
        ha='center', fontsize=11, style='italic', color='gray')

# Divider line
ax.plot([1, 15], [22.4, 22.4], 'k-', linewidth=2)

# ============================================================
# INPUT LAYER
# ============================================================
y = 21.5
draw_box(3, y, 10, 0.7,
         'INPUT: "You are an idiot!"',
         colors['input'], linewidth=2.5, fontsize=11, fontweight='bold')

draw_arrow(8, y, 8, y - 0.8)

# ============================================================
# TOKENIZATION
# ============================================================
y = 20.2
draw_box(2.5, y, 11, 0.8,
         'Tokenization: [CLS] you are an idiot ! [SEP]\n' +
         'Token IDs: [101, 2017, 2024, 2019, 13159, 999, 102]',
         colors['input'], fontsize=9)

draw_arrow(8, y, 8, y - 0.9)

# ============================================================
# EMBEDDING LAYER
# ============================================================
y = 18.8
draw_box(2, y, 12, 1.3,
         'Embedding Layer\n\n' +
         '• Token Embeddings: 30,522 vocab × 768 dimensions\n' +
         '• Position Embeddings: 512 positions × 768 dimensions',
         colors['embedding'], fontsize=9)

draw_param_badge(1.2, y + 0.65, '23M')
draw_shape_label(8, y - 0.5, 'Output: [batch_size, seq_len=512, hidden=768]')

draw_arrow(8, y - 0.7, 8, y - 1.4)

# ============================================================
# TRANSFORMER BLOCKS
# ============================================================
y = 16.8
ax.text(8, y + 0.5, '6 Transformer Encoder Layers',
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray',
                 edgecolor='black', linewidth=2))

# Draw detailed transformer layers
transformer_spacing = 2.2
for i in range(6):
    ty = y - (i * transformer_spacing) - 0.8

    # Main container
    main_box = Rectangle((1.5, ty - 1.8), 13, 1.8,
                         linewidth=2.5,
                         edgecolor='black',
                         facecolor='none',
                         linestyle='--',
                         alpha=0.3)
    ax.add_patch(main_box)

    # Layer number
    ax.text(1.2, ty - 0.9, f'L{i+1}',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='circle,pad=0.3',
                     facecolor='white',
                     edgecolor='black',
                     linewidth=2))

    # Multi-Head Attention
    draw_box(2.5, ty - 0.5, 5, 0.9,
             'Multi-Head Self-Attention\n\n' +
             '12 heads × 64 dims = 768\n' +
             'Q, K, V projections + Output',
             colors['attention'], fontsize=8)

    draw_param_badge(2.2, ty - 0.05, '2.4M')

    # Add & Norm
    draw_box(8, ty - 0.5, 2, 0.4,
             'Add &\nLayerNorm',
             colors['transformer'], fontsize=7, fontweight='bold')

    # Feed Forward
    draw_box(2.5, ty - 1.6, 5, 0.9,
             'Feed-Forward Network\n\n' +
             '768 → 3072 (expand)\n' +
             'GELU activation\n' +
             '3072 → 768 (contract)',
             colors['attention'], fontsize=8)

    draw_param_badge(2.2, ty - 1.15, '4.7M')

    # Add & Norm
    draw_box(8, ty - 1.6, 2, 0.4,
             'Add &\nLayerNorm',
             colors['transformer'], fontsize=7, fontweight='bold')

    # Residual connections (curved arrows)
    # Attention residual
    from matplotlib.patches import Arc
    arc1 = Arc((10.5, ty - 0.3), 1.5, 0.8, angle=0,
              theta1=90, theta2=270,
              color='gray', linewidth=1.5, linestyle='--')
    ax.add_patch(arc1)

    # FFN residual
    arc2 = Arc((10.5, ty - 1.4), 1.5, 0.8, angle=0,
              theta1=90, theta2=270,
              color='gray', linewidth=1.5, linestyle='--')
    ax.add_patch(arc2)

    # Total params for layer
    ax.text(13.5, ty - 0.9, f'7M\nparams',
            ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3',
                     facecolor=colors['params'],
                     edgecolor='orange',
                     linewidth=1.5))

    # Arrow to next layer
    if i < 5:
        draw_arrow(8, ty - 1.85, 8, ty - 2.1, style='-|>')

# Final transformer output
y = y - (5 * transformer_spacing) - 1.2
draw_shape_label(8, y - 1.3, 'Output: [batch_size, seq_len=512, hidden=768]')

# ============================================================
# POOLING
# ============================================================
y = y - 2.2
draw_box(3, y, 10, 0.9,
         'CLS Token Pooling\n\n' +
         'Extract [CLS] representation: output[:, 0, :]',
         colors['pooling'], fontsize=10, text_color='white', fontweight='bold')

draw_shape_label(8, y - 0.6, 'Output: [batch_size, 768]')

draw_arrow(8, y - 0.8, 8, y - 1.3)

# ============================================================
# MULTI-TASK HEADS
# ============================================================
y = y - 1.8
ax.text(8, y, 'Multi-Task Classification Heads (6 independent heads)',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5',
                 facecolor='lightgray',
                 edgecolor='black',
                 linewidth=2))

# Connection arrows fanning out
head_y = y - 1
for i in range(6):
    hx = 1.8 + i * 2.3
    draw_arrow(8, y - 0.3, hx + 1, head_y + 0.1, color='gray', linewidth=1.5)

# Draw each head
labels = ['toxic', 'severe\ntoxic', 'obscene', 'threat', 'insult', 'identity\nhate']
head_y = y - 1
for i, label in enumerate(labels):
    hx = 1.8 + i * 2.3

    # Head box
    draw_box(hx, head_y, 2, 3.5,
             '', colors['heads'], text_color='white')

    # Label at top
    ax.text(hx + 1, head_y + 3.1, label,
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color='white')

    # Architecture details
    components = [
        ('Dropout', 0.1, 2.5),
        ('Linear', '768→768', 2.1),
        ('ReLU', '', 1.7),
        ('Dropout', 0.1, 1.3),
        ('Linear', '768→1', 0.9),
    ]

    for comp_name, comp_detail, comp_y in components:
        detail_text = f'{comp_name}\n{comp_detail}' if comp_detail else comp_name
        ax.text(hx + 1, head_y + comp_y, detail_text,
                ha='center', va='center',
                fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.25',
                         facecolor=(0, 0, 0, 0.2),
                         edgecolor='white',
                         linewidth=1))

    # Arrows between components
    arrow_positions = [2.3, 1.9, 1.5, 1.1]
    for ay in arrow_positions:
        ax.annotate('', xy=(hx + 1, head_y + ay - 0.15),
                   xytext=(hx + 1, head_y + ay + 0.15),
                   arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    # Param count
    draw_param_badge(hx + 1, head_y + 0.3, '590K')

# ============================================================
# OUTPUT LOGITS
# ============================================================
y = head_y - 0.8
ax.text(8, y, 'Output Logits (raw scores)',
        ha='center', fontsize=10, fontweight='bold')

# Output boxes
output_y = y - 0.6
for i, label in enumerate(['toxic', 'severe_t', 'obscene', 'threat', 'insult', 'identity_h']):
    ox = 1.8 + i * 2.3
    draw_box(ox, output_y, 2, 0.5,
             f'{label}',
             colors['output'], fontsize=8, text_color='white', fontweight='bold')

    # Arrow from head to output
    draw_arrow(ox + 1, head_y, ox + 1, output_y + 0.5,
               color=colors['heads'], linewidth=2)

draw_shape_label(8, output_y - 0.5, 'Shape: [batch_size, 6]')

# ============================================================
# LOSS FUNCTION
# ============================================================
y = output_y - 1.2
draw_box(3.5, y, 9, 1,
         'Focal Loss with Class Weights\n\n' +
         'Sigmoid(logits) → probabilities\n' +
         'BCE Loss × (1 - pt)^γ × pos_weight',
         colors['loss'], fontsize=9)

# Arrows from outputs to loss
for i in range(6):
    ox = 1.8 + i * 2.3 + 1
    draw_arrow(ox, output_y, 8, y + 1, color='gray', linewidth=1)

# Backprop arrow
draw_arrow(8, y, 8, y - 0.8, label='Backpropagation',
           color='red', linewidth=2.5, style='<-')

# ============================================================
# SUMMARY PANEL
# ============================================================
y = y - 1.5

# Summary box
summary_box = Rectangle((0.5, 0.3), 15, y - 0.5,
                        linewidth=3,
                        edgecolor='black',
                        facecolor='#FAFAFA',
                        alpha=0.9)
ax.add_patch(summary_box)

# Title
ax.text(8, y - 0.3, 'Model Summary',
        ha='center', fontsize=13, fontweight='bold')

# Stats in columns
col1_x, col2_x, col3_x = 2, 8, 12
stat_y = y - 0.9

# Column 1: Architecture
ax.text(col1_x, stat_y, 'Architecture', fontsize=10, fontweight='bold')
ax.text(col1_x, stat_y - 0.4, '• Base: DistilBERT', fontsize=8)
ax.text(col1_x, stat_y - 0.7, '• Transformer layers: 6', fontsize=8)
ax.text(col1_x, stat_y - 1.0, '• Attention heads: 12', fontsize=8)
ax.text(col1_x, stat_y - 1.3, '• Hidden size: 768', fontsize=8)
ax.text(col1_x, stat_y - 1.6, '• Max sequence: 512', fontsize=8)

# Column 2: Parameters
ax.text(col2_x, stat_y, 'Parameters', fontsize=10, fontweight='bold')
ax.text(col2_x, stat_y - 0.4, '• Total: 69.5M', fontsize=8, fontweight='bold')
ax.text(col2_x, stat_y - 0.7, '  - Embeddings: 23.8M', fontsize=8)
ax.text(col2_x, stat_y - 1.0, '  - Transformers: 42.5M', fontsize=8)
ax.text(col2_x, stat_y - 1.3, '  - Heads: 3.2M', fontsize=8)
ax.text(col2_x, stat_y - 1.6, '• Size (FP16): ~133 MB', fontsize=8)

# Column 3: Training
ax.text(col3_x, stat_y, 'Training', fontsize=10, fontweight='bold')
ax.text(col3_x, stat_y - 0.4, '• Loss: Focal + BCE', fontsize=8)
ax.text(col3_x, stat_y - 0.7, '• Class weights: Yes', fontsize=8)
ax.text(col3_x, stat_y - 1.0, '• Optimizer: AdamW', fontsize=8)
ax.text(col3_x, stat_y - 1.3, '• LR schedule: Warmup', fontsize=8)
ax.text(col3_x, stat_y - 1.6, '• Mixed precision: FP16', fontsize=8)

plt.tight_layout()
plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Professional architecture diagram saved as 'model_architecture.png'")
plt.show()
