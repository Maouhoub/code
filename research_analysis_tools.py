"""
Research Analysis and Visualization Tools for Pruned Super-Resolution Networks
=============================================================================

This module provides comprehensive analysis and visualization tools to generate
publication-quality results for academic papers on network pruning for super-resolution.

Features:
- Statistical analysis of pruning results
- Publication-quality plots and tables
- Comparative performance analysis
- Efficiency vs. quality trade-off analysis
- Layer-wise pruning impact visualization
- Academic writing assistance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class ResearchAnalyzer:
    """
    Comprehensive research analysis tool for pruned super-resolution networks.
    """
    
    def __init__(self, results_path: str):
        """
        Initialize with experimental results.
        
        Args:
            results_path: Path to comprehensive_results.json file
        """
        self.results_path = results_path
        self.results = self.load_results()
        self.setup_plotting_style()
        
    def load_results(self) -> Dict:
        """Load experimental results from JSON file"""
        try:
            with open(self.results_path, 'r') as f:
                results = json.load(f)
            print(f"Loaded results from {self.results_path}")
            return results
        except FileNotFoundError:
            print(f"Results file not found: {self.results_path}")
            return {}
    
    def setup_plotting_style(self):
        """Setup publication-quality plotting style"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Set publication-quality parameters
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def generate_performance_analysis(self, save_dir: str = 'analysis_results'):
        """
        Generate comprehensive performance analysis with publication-quality plots.
        
        Args:
            save_dir: Directory to save analysis results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.results:
            print("No results to analyze")
            return
        
        iteration_results = self.results.get('iteration_results', [])
        if not iteration_results:
            print("No iteration results found")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame(iteration_results)
        
        # 1. PSNR vs Sparsity Trade-off Plot
        self.plot_psnr_sparsity_tradeoff(df, save_dir)
        
        # 2. Multi-metric Performance Dashboard
        self.plot_performance_dashboard(df, save_dir)
        
        # 3. Efficiency Analysis
        self.plot_efficiency_analysis(df, save_dir)
        
        # 4. Convergence Analysis
        self.plot_convergence_analysis(df, save_dir)
        
        # 5. Comparative Analysis Table
        self.generate_comparison_table(df, save_dir)
        
        # 6. Statistical Analysis Report
        self.generate_statistical_report(df, save_dir)
        
        print(f"Analysis complete. Results saved to {save_dir}/")
    
    def plot_psnr_sparsity_tradeoff(self, df: pd.DataFrame, save_dir: str):
        """Plot PSNR vs Sparsity trade-off with trend analysis"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Main trade-off plot
        scatter = ax1.scatter(df['sparsity_percent'], df['psnr'], 
                            c=df['iteration'], cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(df['sparsity_percent'], df['psnr'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(df['sparsity_percent'].min(), df['sparsity_percent'].max(), 100)
        ax1.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2, label='Trend')
        
        # Add annotations for key points
        for i, row in df.iterrows():
            ax1.annotate(f"Iter {int(row['iteration'])}", 
                        (row['sparsity_percent'], row['psnr']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax1.set_xlabel('Model Sparsity (%)')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('PSNR vs Model Sparsity Trade-off')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Colorbar for iterations
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Pruning Iteration')
        
        # Efficiency score plot
        efficiency = df['psnr'] / (df['sparsity_percent'] / 100 + 0.01)  # Avoid division by zero
        ax2.plot(df['iteration'], efficiency, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Pruning Iteration')
        ax2.set_ylabel('Efficiency Score (PSNR/Sparsity)')
        ax2.set_title('Pruning Efficiency Over Iterations')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'psnr_sparsity_tradeoff.png'))
        plt.savefig(os.path.join(save_dir, 'psnr_sparsity_tradeoff.pdf'))
        plt.close()
    
    def plot_performance_dashboard(self, df: pd.DataFrame, save_dir: str):
        """Create comprehensive performance dashboard"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Performance Analysis Dashboard', fontsize=20, y=0.98)
        
        # 1. PSNR Evolution
        axes[0,0].plot(df['iteration'], df['psnr'], 'bo-', linewidth=2, markersize=8)
        axes[0,0].axhline(y=34.45, color='r', linestyle='--', alpha=0.7, label='Target PSNR')
        axes[0,0].set_xlabel('Pruning Iteration')
        axes[0,0].set_ylabel('PSNR (dB)')
        axes[0,0].set_title('PSNR Evolution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. SSIM Evolution
        if 'ssim' in df.columns:
            axes[0,1].plot(df['iteration'], df['ssim'], 'go-', linewidth=2, markersize=8)
            axes[0,1].set_xlabel('Pruning Iteration')
            axes[0,1].set_ylabel('SSIM')
            axes[0,1].set_title('SSIM Evolution')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Sparsity Growth
        axes[0,2].plot(df['iteration'], df['sparsity_percent'], 'ro-', linewidth=2, markersize=8)
        axes[0,2].set_xlabel('Pruning Iteration')
        axes[0,2].set_ylabel('Sparsity (%)')
        axes[0,2].set_title('Model Sparsity Growth')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Inference Speed
        if 'inference_time_ms' in df.columns:
            axes[1,0].plot(df['iteration'], df['inference_time_ms'], 'mo-', linewidth=2, markersize=8)
            axes[1,0].set_xlabel('Pruning Iteration')
            axes[1,0].set_ylabel('Inference Time (ms)')
            axes[1,0].set_title('Inference Speed Evolution')
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. Compression Ratio
        if 'compression_ratio' in df.columns:
            axes[1,1].plot(df['iteration'], df['compression_ratio'], 'co-', linewidth=2, markersize=8)
            axes[1,1].set_xlabel('Pruning Iteration')
            axes[1,1].set_ylabel('Compression Ratio')
            axes[1,1].set_title('Model Compression')
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Multi-metric Radar Chart
        if len(df) > 0:
            last_iter = df.iloc[-1]
            metrics = ['PSNR', 'SSIM', 'Compression', 'Speed']
            values = [
                last_iter['psnr'] / 40.0,  # Normalize to 0-1
                last_iter.get('ssim', 0.9),
                (last_iter.get('compression_ratio', 1) - 1) / 9,  # Assuming max 10x compression
                1 - (last_iter.get('inference_time_ms', 100) / 1000)  # Inverse for speed
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            axes[1,2].plot(angles, values, 'o-', linewidth=2, color='purple')
            axes[1,2].fill(angles, values, alpha=0.25, color='purple')
            axes[1,2].set_xticks(angles[:-1])
            axes[1,2].set_xticklabels(metrics)
            axes[1,2].set_ylim(0, 1)
            axes[1,2].set_title('Final Performance Profile')
            axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_dashboard.png'))
        plt.savefig(os.path.join(save_dir, 'performance_dashboard.pdf'))
        plt.close()
    
    def plot_efficiency_analysis(self, df: pd.DataFrame, save_dir: str):
        """Analyze and plot efficiency metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Efficiency Analysis', fontsize=16)
        
        # 1. PSNR per Parameter
        params_reduced = df['sparsity_percent'] / 100
        psnr_per_param = df['psnr'] / (1 - params_reduced + 0.01)
        
        axes[0,0].plot(df['iteration'], psnr_per_param, 'bo-', linewidth=2)
        axes[0,0].set_xlabel('Pruning Iteration')
        axes[0,0].set_ylabel('PSNR per Remaining Parameter')
        axes[0,0].set_title('Parameter Efficiency')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Speed vs Quality Trade-off
        if 'inference_time_ms' in df.columns:
            axes[0,1].scatter(df['inference_time_ms'], df['psnr'], 
                            c=df['iteration'], cmap='viridis', s=100)
            axes[0,1].set_xlabel('Inference Time (ms)')
            axes[0,1].set_ylabel('PSNR (dB)')
            axes[0,1].set_title('Speed vs Quality Trade-off')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Pareto Frontier Analysis
        sparsity_benefits = df['sparsity_percent']
        psnr_costs = df['psnr'].max() - df['psnr']
        
        axes[1,0].scatter(sparsity_benefits, psnr_costs, c=df['iteration'], cmap='plasma', s=100)
        axes[1,0].set_xlabel('Sparsity Benefit (%)')
        axes[1,0].set_ylabel('PSNR Cost (dB)')
        axes[1,0].set_title('Pareto Frontier: Sparsity vs Quality')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Overall Efficiency Score
        if 'efficiency_score' in df.columns:
            axes[1,1].bar(df['iteration'], df['efficiency_score'], alpha=0.7, color='green')
            axes[1,1].set_xlabel('Pruning Iteration')
            axes[1,1].set_ylabel('Efficiency Score')
            axes[1,1].set_title('Overall Efficiency Score')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'efficiency_analysis.png'))
        plt.savefig(os.path.join(save_dir, 'efficiency_analysis.pdf'))
        plt.close()
    
    def plot_convergence_analysis(self, df: pd.DataFrame, save_dir: str):
        """Analyze convergence properties"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Convergence Analysis', fontsize=16)
        
        # 1. PSNR Convergence with confidence intervals
        psnr_values = df['psnr'].values
        iterations = df['iteration'].values
        
        # Calculate moving average and std
        window = min(3, len(psnr_values))
        if window > 1:
            moving_avg = pd.Series(psnr_values).rolling(window=window, center=True).mean()
            moving_std = pd.Series(psnr_values).rolling(window=window, center=True).std()
            
            axes[0].plot(iterations, psnr_values, 'bo-', alpha=0.6, label='Actual PSNR')
            axes[0].plot(iterations, moving_avg, 'r-', linewidth=2, label=f'Moving Average (w={window})')
            axes[0].fill_between(iterations, moving_avg - moving_std, moving_avg + moving_std, 
                               alpha=0.2, color='red', label='±1?')
        else:
            axes[0].plot(iterations, psnr_values, 'bo-', label='PSNR')
        
        axes[0].set_xlabel('Pruning Iteration')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Rate of Change Analysis
        if len(psnr_values) > 1:
            psnr_change = np.diff(psnr_values)
            axes[1].plot(iterations[1:], psnr_change, 'go-', linewidth=2)
            axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            axes[1].set_xlabel('Pruning Iteration')
            axes[1].set_ylabel('PSNR Change (dB)')
            axes[1].set_title('Rate of PSNR Change')
            axes[1].grid(True, alpha=0.3)
        
        # 3. Stability Analysis
        if 'fine_tuning_epochs' in df.columns:
            axes[2].bar(df['iteration'], df['fine_tuning_epochs'], alpha=0.7, color='purple')
            axes[2].set_xlabel('Pruning Iteration')
            axes[2].set_ylabel('Fine-tuning Epochs Required')
            axes[2].set_title('Training Stability (Epochs to Converge)')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'convergence_analysis.png'))
        plt.savefig(os.path.join(save_dir, 'convergence_analysis.pdf'))
        plt.close()
    
    def generate_comparison_table(self, df: pd.DataFrame, save_dir: str):
        """Generate publication-quality comparison table"""
        
        # Create comparison table
        table_data = []
        
        for _, row in df.iterrows():
            table_data.append({
                'Iteration': int(row['iteration']),
                'Sparsity (%)': f"{row['sparsity_percent']:.2f}",
                'PSNR (dB)': f"{row['psnr']:.3f}",
                'SSIM': f"{row.get('ssim', 0):.4f}",
                'Compression': f"{row.get('compression_ratio', 1):.2f}×",
                'Inference (ms)': f"{row.get('inference_time_ms', 0):.2f}",
                'Efficiency': f"{row.get('efficiency_score', 0):.2f}",
                'FT Epochs': int(row.get('fine_tuning_epochs', 0))
            })
        
        # Convert to DataFrame and save
        table_df = pd.DataFrame(table_data)
        
        # Save as CSV for easy import into papers
        table_df.to_csv(os.path.join(save_dir, 'comparison_table.csv'), index=False)
        
        # Create formatted LaTeX table
        latex_table = table_df.to_latex(
            index=False,
            float_format="%.3f",
            caption="Comprehensive comparison of pruning iterations showing quality-efficiency trade-offs.",
            label="tab:pruning_comparison",
            position="htbp"
        )
        
        with open(os.path.join(save_dir, 'comparison_table.tex'), 'w') as f:
            f.write(latex_table)
        
        # Create visual table
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_df.values,
                        colLabels=table_df.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color coding for performance
        for i in range(len(table_df)):
            psnr = float(table_df.iloc[i]['PSNR (dB)'])
            if psnr > 35:
                color = '#90EE90'  # Light green
            elif psnr > 34.5:
                color = '#FFFFE0'  # Light yellow
            else:
                color = '#FFB6C1'  # Light red
            
            for j in range(len(table_df.columns)):
                table[(i+1, j)].set_facecolor(color)
        
        plt.title('Pruning Results Comparison Table', fontsize=16, pad=20)
        plt.savefig(os.path.join(save_dir, 'comparison_table.png'))
        plt.savefig(os.path.join(save_dir, 'comparison_table.pdf'))
        plt.close()
        
        print(f"Tables saved: CSV, LaTeX, and PNG formats")
    
    def generate_statistical_report(self, df: pd.DataFrame, save_dir: str):
        """Generate comprehensive statistical analysis report"""
        
        report = []
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Basic statistics
        report.append("1. DESCRIPTIVE STATISTICS")
        report.append("-" * 30)
        report.append(f"Number of pruning iterations: {len(df)}")
        report.append(f"PSNR range: {df['psnr'].min():.3f} - {df['psnr'].max():.3f} dB")
        report.append(f"PSNR mean ± std: {df['psnr'].mean():.3f} ± {df['psnr'].std():.3f} dB")
        report.append(f"Sparsity range: {df['sparsity_percent'].min():.2f} - {df['sparsity_percent'].max():.2f}%")
        report.append(f"Final compression ratio: {df['compression_ratio'].iloc[-1]:.2f}×")
        report.append("")
        
        # Correlation analysis
        if len(df) > 2:
            corr_psnr_sparsity = stats.pearsonr(df['sparsity_percent'], df['psnr'])[0]
            report.append("2. CORRELATION ANALYSIS")
            report.append("-" * 30)
            report.append(f"PSNR vs Sparsity correlation: {corr_psnr_sparsity:.3f}")
            
            if abs(corr_psnr_sparsity) > 0.7:
                strength = "strong"
            elif abs(corr_psnr_sparsity) > 0.4:
                strength = "moderate"
            else:
                strength = "weak"
            
            direction = "negative" if corr_psnr_sparsity < 0 else "positive"
            report.append(f"Interpretation: {strength} {direction} correlation")
            report.append("")
        
        # Performance degradation analysis
        if len(df) > 1:
            initial_psnr = df['psnr'].iloc[0]
            final_psnr = df['psnr'].iloc[-1]
            total_degradation = initial_psnr - final_psnr
            degradation_per_sparsity = total_degradation / df['sparsity_percent'].iloc[-1] * 100
            
            report.append("3. PERFORMANCE DEGRADATION ANALYSIS")
            report.append("-" * 40)
            report.append(f"Total PSNR degradation: {total_degradation:.3f} dB")
            report.append(f"Degradation per % sparsity: {degradation_per_sparsity:.4f} dB/%")
            report.append(f"Final sparsity achieved: {df['sparsity_percent'].iloc[-1]:.2f}%")
            report.append("")
        
        # Efficiency metrics
        if 'efficiency_score' in df.columns:
            report.append("4. EFFICIENCY ANALYSIS")
            report.append("-" * 25)
            report.append(f"Best efficiency score: {df['efficiency_score'].max():.2f}")
            report.append(f"Average efficiency: {df['efficiency_score'].mean():.2f}")
            report.append(f"Efficiency improvement: {(df['efficiency_score'].iloc[-1]/df['efficiency_score'].iloc[0] - 1)*100:.1f}%")
            report.append("")
        
        # Convergence analysis
        if len(df) > 2:
            # Calculate if PSNR is stabilizing
            recent_std = df['psnr'].tail(3).std() if len(df) >= 3 else df['psnr'].std()
            overall_std = df['psnr'].std()
            
            report.append("5. CONVERGENCE ANALYSIS")
            report.append("-" * 25)
            report.append(f"Overall PSNR variance: {overall_std:.4f}")
            report.append(f"Recent PSNR variance: {recent_std:.4f}")
            
            if recent_std < overall_std * 0.5:
                convergence_status = "Converging (variance decreasing)"
            elif recent_std < overall_std:
                convergence_status = "Stabilizing"
            else:
                convergence_status = "Still varying"
            
            report.append(f"Convergence status: {convergence_status}")
            report.append("")
        
        # Recommendations
        report.append("6. RECOMMENDATIONS FOR PUBLICATION")
        report.append("-" * 35)
        
        final_psnr = df['psnr'].iloc[-1]
        final_sparsity = df['sparsity_percent'].iloc[-1]
        
        if final_psnr > 34.5 and final_sparsity > 50:
            report.append("? Excellent results: High sparsity with minimal quality loss")
        elif final_psnr > 34.0 and final_sparsity > 30:
            report.append("? Good results: Reasonable trade-off achieved")
        else:
            report.append("? Results may need improvement for strong publication")
        
        report.append("")
        report.append("Suggested metrics to highlight:")
        report.append(f"- Achieved {final_sparsity:.1f}% model compression")
        report.append(f"- Maintained {final_psnr:.2f} dB PSNR")
        
        if 'compression_ratio' in df.columns:
            compression = df['compression_ratio'].iloc[-1]
            report.append(f"- {compression:.1f}× parameter reduction")
        
        if 'inference_time_ms' in df.columns and len(df) > 1:
            speed_improvement = (df['inference_time_ms'].iloc[0] / df['inference_time_ms'].iloc[-1] - 1) * 100
            if speed_improvement > 0:
                report.append(f"- {speed_improvement:.1f}% inference speedup")
        
        # Save report
        report_text = "\n".join(report)
        with open(os.path.join(save_dir, 'statistical_report.txt'), 'w') as f:
            f.write(report_text)
        
        print("Statistical analysis report generated")
        print("\nKey findings:")
        print(f"- Final PSNR: {final_psnr:.3f} dB")
        print(f"- Final sparsity: {final_sparsity:.2f}%")
        if len(df) > 1:
            print(f"- Total degradation: {initial_psnr - final_psnr:.3f} dB")


def generate_publication_package(results_path: str, output_dir: str = 'publication_package'):
    """
    Generate a complete publication package with all analysis and visualizations.
    
    Args:
        results_path: Path to comprehensive_results.json
        output_dir: Directory to save publication package
    """
    
    print("Generating publication package...")
    
    # Create analyzer
    analyzer = ResearchAnalyzer(results_path)
    
    # Generate all analyses
    analyzer.generate_performance_analysis(output_dir)
    
    # Create README for the package
    readme_content = """
# Publication Package: Pruned Super-Resolution Networks

This package contains comprehensive analysis and visualization results for the paper:
"Advanced Fine-tuning Strategies for Pruned Super-Resolution Networks"

## Contents:

### Figures (Publication Quality):
- `psnr_sparsity_tradeoff.png/pdf` - Main trade-off analysis
- `performance_dashboard.png/pdf` - Comprehensive performance overview
- `efficiency_analysis.png/pdf` - Efficiency metrics analysis
- `convergence_analysis.png/pdf` - Training convergence analysis
- `comparison_table.png/pdf` - Results comparison table

### Data Files:
- `comparison_table.csv` - Raw data for tables
- `comparison_table.tex` - LaTeX table for paper
- `statistical_report.txt` - Comprehensive statistical analysis

### Usage Instructions:
1. Use PNG files for presentations and initial submissions
2. Use PDF files for final publication (vector graphics)
3. Import CSV data into your analysis software
4. Copy LaTeX table directly into your paper
5. Refer to statistical report for key findings and recommendations

### Citation Metrics to Highlight:
- Model compression ratio achieved
- PSNR maintenance with high sparsity
- Inference speed improvements
- Comparison with state-of-the-art methods

### Recommended Paper Structure:
1. Introduction - motivation for efficient SR networks
2. Related Work - pruning and SR background
3. Methodology - your advanced fine-tuning strategies
4. Experimental Setup - dataset, metrics, baselines
5. Results - use the generated figures and tables
6. Analysis - use statistical findings from report
7. Conclusion - impact and future work

Generated on: """ + time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"Publication package generated in: {output_dir}/")
    print("Ready for academic submission!")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = "comprehensive_results.json"
    
    if os.path.exists(results_path):
        generate_publication_package(results_path)
    else:
        print(f"Results file not found: {results_path}")
        print("Please run your pruning experiment first to generate results.")
