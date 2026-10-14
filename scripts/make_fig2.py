import matplotlib.pyplot as plt
labels = ['Source-only\n(clear val)', 'Source-only\n(foggy val)', 'Oracle\n(foggy val)']
vals   = [54.3, 29.4, 49.2]
fig, ax = plt.subplots(figsize=(4.6, 3.3))
bars = ax.bar(labels, vals, color=['#9e9e9e', '#c0392b', '#2e7d32'], width=0.6)
ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=10)
# gap marker in the space between the red (x=1) and green (x=2) bars
ax.plot([1.3, 1.5], [29.4, 29.4], ls='--', color='gray', lw=0.8)
ax.plot([1.5, 1.7], [49.2, 49.2], ls='--', color='gray', lw=0.8)
ax.annotate('', xy=(1.5, 49.2), xytext=(1.5, 29.4),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2, shrinkA=0, shrinkB=0))
ax.text(1.5, 39.3, '19.8', ha='center', va='center', fontsize=9,
        bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
ax.set_ylabel('mAP@0.5 (%)'); ax.set_ylim(0, 62)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig('fig2_gap.pdf'); plt.savefig('fig2_gap.png', dpi=300)
print('saved')
