import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Örnek: segment + predicted_label içeren DataFrame
# df_predictions = pd.read_csv("your_predictions.csv")  # Eğer dışardan yüklüyorsan

# Adımlar: Gruplama → Yüzde hesaplama → Bar chart çizimi
plot_df = df_predictions.groupby(['segment', 'predicted_label']).size().reset_index(name='count')
plot_df = plot_df.pivot(index='segment', columns='predicted_label', values='count').fillna(0)

# Yüzdelere çevir
plot_df_pct = plot_df.div(plot_df.sum(axis=1), axis=0) * 100

# Renk sırası: senin model çıktına göre sıralanmalı
promo_order = ['NO_PROMOTION', 'GROWTH_TARGET', 'INTERVENTION_NEEDED']
plot_df_pct = plot_df_pct[promo_order]  # Kolon sıralaması

# Plot
sns.set(style="whitegrid")
plot_df_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')

plt.title('Figure 5.7 – Predicted Promotion Distribution by Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Percentage of Customers (%)')
plt.legend(title='Promotion Category', loc='upper right')
plt.tight_layout()

# Kaydet
plt.savefig("figures/segment_promo_distribution.png", dpi=300)
plt.show()
