"""
Temporal Evolution Visualization Suite
Bath University Thesis - Data Visualization Component
Generate publication-ready charts for academic presentation

Author: Muhammed Yavuzhan CANLI
Institution: University of Bath
Course: MSc Business Analytics
Academic Standard: A-Grade Compliance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TemporalVisualizationSuite:
    """
    Comprehensive visualization suite for temporal promotion evolution analysis.
    Generates thesis-ready charts and academic documentation.
    """
    
    def __init__(self, csv_file=None):
        """
        Initialize with temporal evolution data.
        
        Args:
            csv_file: Path to unified_temporal_evolution_*.csv file
        """
        if csv_file:
            self.evolution_data = pd.read_csv(csv_file)
        else:
            # Load the most recent file if available
            try:
                import glob
                csv_files = glob.glob('unified_temporal_evolution_*.csv')
                if csv_files:
                    latest_file = max(csv_files)
                    self.evolution_data = pd.read_csv(latest_file)
                    print(f"Loaded data from: {latest_file}")
                else:
                    raise FileNotFoundError("No temporal evolution CSV files found")
            except Exception as e:
                print(f"Error loading data: {e}")
                self.evolution_data = None
        
        # Set visualization style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_confidence_evolution_chart(self, save_path=None):
        """Generate confidence evolution line chart across periods."""
        if self.evolution_data is None:
            print("No data available for visualization")
            return
        
        print("Creating confidence evolution chart...")
        
        # Calculate period-wise confidence statistics
        confidence_stats = self.evolution_data.groupby('period')['confidence'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(4)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Main confidence evolution line plot
        periods = confidence_stats.index
        means = confidence_stats['mean']
        stds = confidence_stats['std']
        
        ax1.plot(periods, means, marker='o', linewidth=3, markersize=8, 
                label='Mean Confidence', color='#2E86AB')
        ax1.fill_between(periods, means - stds, means + stds, 
                        alpha=0.3, color='#2E86AB', label='±1 Standard Deviation')
        
        ax1.set_title('Random Forest Confidence Evolution\nAcross Temporal Periods', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Analysis Period', fontsize=12)
        ax1.set_ylabel('Prediction Confidence', fontsize=12)
        ax1.set_ylim(0.3, 0.8)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add value annotations
        for i, (period, mean) in enumerate(zip(periods, means)):
            ax1.annotate(f'{mean:.3f}', (i, mean), 
                        textcoords="offset points", xytext=(0,10), ha='center',
                        fontweight='bold', fontsize=10)
        
        # Confidence distribution box plot
        period_data = []
        period_labels = []
        
        for period in periods:
            period_conf = self.evolution_data[self.evolution_data['period'] == period]['confidence']
            period_data.append(period_conf)
            period_labels.append(f"{period}\n(n={len(period_conf):,})")
        
        bp = ax2.boxplot(period_data, labels=period_labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        
        ax2.set_title('Confidence Distribution\nby Period', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Prediction Confidence', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence evolution chart saved: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\nConfidence Evolution Summary:")
        print(confidence_stats)
        
        return confidence_stats
    
    def create_priority_heatmap(self, save_path=None):
        """Generate promotion priority heatmap visualization."""
        if self.evolution_data is None:
            print("No data available for visualization")
            return
        
        print("Creating promotion priority heatmap...")
        
        # Create crosstab for heatmap
        priority_matrix = pd.crosstab(
            self.evolution_data['predicted_promotion'],
            self.evolution_data['business_priority'],
            normalize='index'
        ) * 100  # Convert to percentages
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Generate heatmap
        sns.heatmap(priority_matrix, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Percentage of Customers (%)'},
                   linewidths=0.5,
                   annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        
        plt.title('Business Priority Distribution by Promotion Type\n' +
                 'Random Forest Customer Segmentation Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Business Priority Level', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Promotion Category', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Priority heatmap saved: {save_path}")
        
        plt.show()
        
        # Print raw counts for academic documentation
        raw_counts = pd.crosstab(
            self.evolution_data['predicted_promotion'],
            self.evolution_data['business_priority']
        )
        
        print("\nPromotion-Priority Cross-tabulation (Raw Counts):")
        print(raw_counts)
        print(f"\nTotal customers analyzed: {raw_counts.sum().sum():,}")
        
        return priority_matrix, raw_counts
    
    def create_temporal_distribution_analysis(self, save_path=None):
        """Generate comprehensive temporal distribution analysis."""
        if self.evolution_data is None:
            print("No data available for visualization")
            return
        
        print("Creating temporal distribution analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Customer volume by period
        period_counts = self.evolution_data['period'].value_counts().sort_index()
        
        bars1 = ax1.bar(period_counts.index, period_counts.values, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Customer Volume Growth\nAcross Analysis Periods', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Customers')
        ax1.set_xlabel('Analysis Period')
        
        # Add value labels on bars
        for bar, value in zip(bars1, period_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Promotion distribution by period
        promo_period = pd.crosstab(self.evolution_data['period'], 
                                  self.evolution_data['predicted_promotion'])
        promo_period_pct = promo_period.div(promo_period.sum(axis=1), axis=0) * 100
        
        promo_period_pct.plot(kind='bar', stacked=True, ax=ax2, 
                             colormap='Set3', alpha=0.8)
        ax2.set_title('Promotion Distribution by Period\n(Percentage)', 
                     fontsize=12, fontweight='bold')
        ax2.set_ylabel('Percentage of Customers (%)')
        ax2.set_xlabel('Analysis Period')
        ax2.legend(title='Promotion Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Business priority evolution
        priority_period = pd.crosstab(self.evolution_data['period'], 
                                     self.evolution_data['business_priority'])
        
        priority_period.plot(kind='area', ax=ax3, alpha=0.7, colormap='viridis')
        ax3.set_title('Business Priority Evolution\nOver Time', 
                     fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Customers')
        ax3.set_xlabel('Analysis Period')
        ax3.legend(title='Priority Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Confidence score distribution
        self.evolution_data.boxplot(column='confidence', by='period', ax=ax4)
        ax4.set_title('Confidence Score Distribution\nby Analysis Period', 
                     fontsize=12, fontweight='bold')
        ax4.set_ylabel('Prediction Confidence')
        ax4.set_xlabel('Analysis Period')
        plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Temporal distribution analysis saved: {save_path}")
        
        plt.show()
        
        return period_counts, promo_period_pct, priority_period
    
    def create_academic_summary_dashboard(self, save_path=None):
        """Generate comprehensive academic dashboard for thesis presentation."""
        if self.evolution_data is None:
            print("No data available for visualization")
            return
        
        print("Creating academic summary dashboard...")
        
        # Create main dashboard figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Random Forest Temporal Promotion Evolution System\n' +
                    'Bath University MSc Business Analytics Thesis', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. System Overview (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Key metrics
        total_customers = len(self.evolution_data)
        unique_customers = self.evolution_data['customer_id'].nunique()
        avg_confidence = self.evolution_data['confidence'].mean()
        periods_analyzed = self.evolution_data['period'].nunique()
        
        metrics_text = f"""
SYSTEM PERFORMANCE METRICS

Total Customer Records: {total_customers:,}
Unique Customers: {unique_customers:,}
Analysis Periods: {periods_analyzed}
Average Confidence: {avg_confidence:.3f}

Growth Rate: {(total_customers/2326-1)*100:.0f}% from baseline
High Priority Customers: {len(self.evolution_data[self.evolution_data['business_priority']=='HIGH']):,}
Risk Interventions: {len(self.evolution_data[self.evolution_data['business_priority']=='URGENT']):,}
        """
        
        ax1.text(0.1, 0.5, metrics_text, transform=ax1.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('System Overview', fontsize=14, fontweight='bold')
        
        # 2. Confidence Evolution (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        period_conf = self.evolution_data.groupby('period')['confidence'].mean()
        ax2.plot(period_conf.index, period_conf.values, marker='o', 
                linewidth=3, markersize=8, color='#2E86AB')
        ax2.set_title('Confidence Evolution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Mean Confidence')
        ax2.grid(True, alpha=0.3)
        
        # 3. Priority Heatmap (middle)
        ax3 = fig.add_subplot(gs[1, :])
        
        priority_matrix = pd.crosstab(
            self.evolution_data['predicted_promotion'],
            self.evolution_data['business_priority'],
            normalize='index'
        ) * 100
        
        sns.heatmap(priority_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r',
                   ax=ax3, cbar_kws={'label': 'Percentage (%)'})
        ax3.set_title('Business Priority Distribution Matrix', 
                     fontsize=14, fontweight='bold')
        
        # 4. Volume Growth (bottom-left)
        ax4 = fig.add_subplot(gs[2, :2])
        
        period_counts = self.evolution_data['period'].value_counts().sort_index()
        bars = ax4.bar(period_counts.index, period_counts.values, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax4.set_title('Customer Volume Growth', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Customers')
        
        for bar, value in zip(bars, period_counts.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Promotion Distribution (bottom-right)
        ax5 = fig.add_subplot(gs[2, 2:])
        
        promo_counts = self.evolution_data['predicted_promotion'].value_counts()
        wedges, texts, autotexts = ax5.pie(promo_counts.values, 
                                          labels=promo_counts.index,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          colors=sns.color_palette("Set3"))
        ax5.set_title('Overall Promotion Distribution', fontsize=14, fontweight='bold')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Academic dashboard saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def generate_all_visualizations(self, output_prefix="temporal_viz"):
        """Generate all visualizations with timestamp-based filenames."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        print("Generating complete visualization suite...")
        
        # Create all visualizations
        confidence_stats = self.create_confidence_evolution_chart(
            f"{output_prefix}_confidence_{timestamp}.png"
        )
        
        priority_matrix, raw_counts = self.create_priority_heatmap(
            f"{output_prefix}_priority_heatmap_{timestamp}.png"
        )
        
        period_data = self.create_temporal_distribution_analysis(
            f"{output_prefix}_temporal_analysis_{timestamp}.png"
        )
        
        dashboard = self.create_academic_summary_dashboard(
            f"{output_prefix}_academic_dashboard_{timestamp}.png"
        )
        
        print(f"\nAll visualizations generated with prefix: {output_prefix}_{timestamp}")
        
        return {
            'confidence_stats': confidence_stats,
            'priority_matrix': priority_matrix,
            'raw_counts': raw_counts,
            'period_data': period_data,
            'dashboard': dashboard
        }
    
    def export_visualization_summary(self):
        """Export summary statistics for academic documentation."""
        if self.evolution_data is None:
            print("No data available for summary")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        summary_file = f'visualization_summary_{timestamp}.txt'
        
        with open(summary_file, 'w') as f:
            f.write("TEMPORAL EVOLUTION VISUALIZATION SUMMARY\n")
            f.write("Bath University Thesis - Statistical Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("DATASET OVERVIEW:\n")
            f.write(f"Total records: {len(self.evolution_data):,}\n")
            f.write(f"Unique customers: {self.evolution_data['customer_id'].nunique():,}\n")
            f.write(f"Analysis periods: {self.evolution_data['period'].nunique()}\n")
            f.write(f"Mean confidence: {self.evolution_data['confidence'].mean():.4f}\n")
            f.write(f"Confidence std: {self.evolution_data['confidence'].std():.4f}\n\n")
            
            # Period-wise breakdown
            f.write("PERIOD-WISE ANALYSIS:\n")
            period_stats = self.evolution_data.groupby('period').agg({
                'customer_id': 'count',
                'confidence': ['mean', 'std']
            }).round(4)
            f.write(str(period_stats))
            f.write("\n\n")
            
            # Promotion distribution
            f.write("PROMOTION DISTRIBUTION:\n")
            promo_dist = self.evolution_data['predicted_promotion'].value_counts()
            for promo, count in promo_dist.items():
                pct = (count / len(self.evolution_data)) * 100
                f.write(f"{promo}: {count:,} ({pct:.1f}%)\n")
            f.write("\n")
            
            # Priority distribution
            f.write("BUSINESS PRIORITY DISTRIBUTION:\n")
            priority_dist = self.evolution_data['business_priority'].value_counts()
            for priority, count in priority_dist.items():
                pct = (count / len(self.evolution_data)) * 100
                f.write(f"{priority}: {count:,} ({pct:.1f}%)\n")
        
        print(f"Visualization summary exported: {summary_file}")
        return summary_file

def main():
    """Execute visualization suite."""
    print("TEMPORAL EVOLUTION VISUALIZATION SUITE")
    print("Bath University Thesis - Data Visualization")
    print("=" * 50)
    
    try:
        # Initialize visualization suite
        viz_suite = TemporalVisualizationSuite()
        
        if viz_suite.evolution_data is not None:
            # Generate all visualizations
            results = viz_suite.generate_all_visualizations()
            
            # Export summary
            summary_file = viz_suite.export_visualization_summary()
            
            print(f"\nVisualization suite completed successfully!")
            print(f"Summary documentation: {summary_file}")
            print(f"All charts generated and ready for thesis presentation")
        else:
            print("No data available for visualization")
            
    except Exception as e:
        print(f"Visualization suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()