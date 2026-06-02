# docs/generate_ml_pipeline.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# For ML pipelines without cloud icons, use pure matplotlib
fig, ax = plt.subplots(1, 1, figsize=(16, 6), facecolor='#1C1C1E')
ax.set_facecolor('#1C1C1E')
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

steps = [
    ("Raw Dataset\n(Slope/Deflection CSV)", 0.8, "#0A84FF"),
    ("Encoding\n& Scaling", 2.2, "#30D158"),
    ("Split\nTrain/Val/Test", 3.6, "#FFD60A"),
    ("Standard MLP\nBeamNet", 5.0, "#FF6B6B"),
    ("Physics-Informed\nBeamPINN (Tanh)", 6.4, "#00E676"),
    ("Evaluation\n& Comparison", 7.8, "#BF5AF2"),
    ("Production Deploy\n(PM2 & HF Spaces)", 9.2, "#64D2FF"),
]

for i, (label, x, color) in enumerate(steps):
    box = mpatches.FancyBboxPatch(
        (x - 0.62, 1.3), 1.24, 1.4,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9
    )
    ax.add_patch(box)
    
    # Use dark text for bright yellow for contrast
    text_color = 'black' if color == "#FFD60A" else 'white'
    
    ax.text(x, 2.0, label, ha='center', va='center',
            fontsize=8, fontweight='bold', color=text_color,
            fontfamily='DejaVu Sans')
    if i < len(steps) - 1:
        ax.annotate('', xy=(steps[i+1][1] - 0.63, 2.0),
                    xytext=(x + 0.63, 2.0),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))

ax.set_title('BeamML Training & Deployment Pipeline', color='white', fontsize=18,
             fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('docs/images/ml-pipeline.png', dpi=200,
            bbox_inches='tight', facecolor='#1C1C1E')
plt.close()
print("✅ docs/images/ml-pipeline.png generated")
