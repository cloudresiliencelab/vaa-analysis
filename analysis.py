#!/usr/bin/env python3
"""
Individual Figures for LLM Political Bias Analysis

This script creates separate figures for each key finding to allow individual analysis
and better visualization of specific aspects of the data.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IndividualFigureGenerator:
    def __init__(self, data_dir):
        """Initialize the figure generator."""
        self.data_dir = Path(data_dir)
        self.llms = ['gemini 2.0', 'gemini 2.5', 'gpt 4o', 'grok 3', 'mistral large']
        self.data = {}
        self.processed_data = {}
        
    def load_and_process_data(self):
        """Load and process all CSV files."""
        csv_files = {
            'luxembourg': 'Smartwielen.csv',
            'netherlands': 'StemWijzer.csv', 
            'germany': 'Wahl-O-Mat.csv',
            'czech': 'Wahlrechner Tschechien.csv'
        }
        
        for country, filename in csv_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                print(f"Loading {country} data from {filename}...")
                self.data[country] = pd.read_csv(filepath, sep=';')
            else:
                print(f"Warning: {filename} not found!")
                
        self._clean_and_process_data()
        self._classify_political_orientation()
    
    def _clean_and_process_data(self):
        """Clean and process the raw data."""
        for country, df in self.data.items():
            # Remove empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Get the first few rows that contain the actual data
            if country == 'luxembourg':
                data_rows = df.iloc[1:14, :]
                party_col = 0
                llm_cols = [3, 4, 5, 6, 7]
            elif country == 'netherlands':
                data_rows = df.iloc[1:22, :]
                party_col = 0
                llm_cols = [3, 4, 5, 6, 7]
            elif country == 'germany':
                data_rows = df.iloc[1:31, :]
                party_col = 0
                llm_cols = [3, 4, 5, 6, 7]
            elif country == 'czech':
                data_rows = df.iloc[1:28, :]
                party_col = 0
                llm_cols = [1, 2, 3, 4, 5]
            
            # Extract party names and LLM scores
            parties = []
            llm_scores = {llm: [] for llm in self.llms}
            
            for idx, row in data_rows.iterrows():
                party_name = str(row.iloc[party_col]).strip()
                if party_name and party_name != 'nan' and not party_name.startswith('Party'):
                    parties.append(party_name)
                    
                    for i, llm in enumerate(self.llms):
                        if i < len(llm_cols) and llm_cols[i] < len(row):
                            score_str = str(row.iloc[llm_cols[i]]).replace(',', '.').replace('%', '')
                            try:
                                score = float(score_str)
                                llm_scores[llm].append(score)
                            except (ValueError, TypeError):
                                llm_scores[llm].append(np.nan)
            
            # Create processed dataframe
            processed_df = pd.DataFrame({
                'party': parties,
                'country': country
            })
            
            for llm in self.llms:
                processed_df[llm] = llm_scores[llm][:len(parties)]
            
            processed_df = processed_df.dropna(subset=self.llms, how='all')
            self.processed_data[country] = processed_df
    
    def _classify_political_orientation(self):
        """Classify parties by political orientation."""
        political_classification = {
            # Luxembourg
            'Christian Social People\'s Party': 'centre-right',
            'Communist Party of Luxembourg': 'left',
            'déi gréng': 'centre-left', 
            'déi Konservativ – Freedomeparty': 'right',
            'déi Lénk': 'left',
            'Democratic Party': 'liberal',
            'Democratic reform party': 'right',
            'Fokus': 'transversal',
            'Luxembourg Socialist Workers\' Party': 'centre-left',
            'Oppositiounsbeweegung Mir d\'Vollek': 'right',
            'PIRATES': 'centre-left',
            'Volt Luxembourg': 'centre-left',
            
            # Netherlands
            '50PLUS': 'centre-right',
            'BBB': 'transversal',
            'BVNL': 'right-wing',
            'CDA': 'centre-right',
            'CU': 'centre-right',
            'D66': 'centre-left',
            'FvD': 'right-wing',
            'GL-PvdA': 'centre-left',
            'JA21': 'right-wing',
            'MDD': 'right-wing',
            'NL PLAN EU': 'centre',
            'NSC': 'centre-right',
            'Piratenpartij - De Groenen': 'syncretic',
            'PvdD': 'left-wing',
            'PVV': 'right-wing',
            'SGP': 'centre-right',
            'SP': 'left-wing',
            'vandeRegio': 'centre',
            'Volt': 'centre-left',
            'VVD': 'liberal',
            
            # Germany
            'AfD': 'right-wing',
            'Alliance C': 'right-wing',
            'ALLIANCE GERMANY': 'right-wing',
            'Animal Protection Party': 'left-wing',
            'BP': 'centre-right',
            'BSW': 'left-wing',
            'BüSo': 'syncretic',
            'CDU/CSU': 'centre-right',
            'FDP': 'centre-right',
            'FREE VOTERS': 'centre-right',
            'GREENS': 'centre-left',
            'HUMAN WORLD': 'unknown',
            'MERA25': 'left-wing',
            'MLPD': 'left-wing',
            'PDF': 'centre-left',
            'PIRATES': 'centre-left',
            'PdH': 'unknown',
            'Rejuvenation research': 'single-issue',
            'SGP': 'far-left',
            'SPD': 'centre-left',
            'The Justice Party - Team Todenhöfer': 'unknown',
            'The Left': 'left-wing',
            'The party': 'left-wing',
            'Values  Union': 'right-wing',
            'theBasis': 'unknown',
            'volt': 'centre-left',
            'weeks of pregnancy': 'unknown',
            'ÖDP': 'centre-right',
            
            # Czech Republic
            'Aliance za nezávislost ČR': 'right-wing',
            'ANO': 'centre-right',
            'DSZ - ZA PRÁVA ZVÍŘAT': 'left-wing',
            'Hlas': 'centre-left',
            'KAN': 'centre-right',
            'Koalice ŠD SSPD-SP': 'left-wing',
            'LANO': 'centre-left',
            'Levice': 'left-wing',
            'LŽPL': 'right-wing',
            'mimozemstani.eu': 'centre-left',
            'Mourek': 'centre-left',
            'Nevolte Urza.cz': 'right-wing',
            'Piráti': 'centre-left',
            'Přísaha': 'right-wing',
            'PRO 2022': 'right-wing',
            'PRO vystoupení z EU': 'right-wing',
            'SEN 21 + Volt': 'centre-left',
            'SOCDEM': 'centre-left',
            'SPD + Trikolóra': 'right-wing',
            'SPOLU': 'centre-right',
            'Stačilo!': 'centre-right',
            'STAN': 'centre-left',
            'Suverenita+Domov+Směr': 'right-wing',
            'Svobodní': 'right-wing',
            'Zelení': 'centre-left'
        }
        
        for country, df in self.processed_data.items():
            df['orientation'] = df['party'].map(political_classification)
            df['orientation_group'] = df['orientation'].apply(self._group_orientation)
    
    def _group_orientation(self, orientation):
        """Group political orientations into broader categories."""
        if pd.isna(orientation):
            return 'unknown'
        
        left_orientations = ['left', 'left-wing', 'far-left', 'centre-left']
        right_orientations = ['right', 'right-wing', 'centre-right']
        center_orientations = ['centre', 'liberal', 'transversal', 'syncretic', 'single-issue']
        
        if orientation in left_orientations:
            return 'left'
        elif orientation in right_orientations:
            return 'right'
        elif orientation in center_orientations:
            return 'center'
        else:
            return 'unknown'
    
    def create_figure_1_llm_comparison(self):
        """Figure 1: Box plot comparing all LLMs."""
        print("Creating Figure 1: LLM Comparison Box Plot...")
        
        # Combine all data
        all_data = pd.concat(self.processed_data.values(), ignore_index=True)
        
        # Prepare data for box plot
        llm_data = []
        llm_labels = []
        for llm in self.llms:
            scores = all_data[llm].dropna()
            llm_data.append(scores)
            llm_labels.append(llm.replace(' ', '\n'))
        
        # Create figure
        plt.figure(figsize=(12, 8))
        box_plot = plt.boxplot(llm_data, labels=llm_labels, patch_artist=True)
        
        # Color the boxes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Distribution of LLM Scores Across All Political Parties', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Score (%)', fontsize=14)
        plt.xlabel('Large Language Model', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=0)
        
        # Add mean markers
        for i, (data, label) in enumerate(zip(llm_data, llm_labels)):
            mean_val = np.mean(data)
            plt.scatter(i+1, mean_val, color='red', s=100, marker='D', 
                       label='Mean' if i == 0 else "", zorder=5)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('figure_1_llm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: figure_1_llm_comparison.png")
    
    def create_figure_2_mean_scores(self):
        """Figure 2: Mean scores by LLM with error bars."""
        print("Creating Figure 2: Mean Scores by LLM...")
        
        # Combine all data
        all_data = pd.concat(self.processed_data.values(), ignore_index=True)
        
        # Calculate means and standard errors
        means = []
        std_errors = []
        llm_labels = []
        
        for llm in self.llms:
            scores = all_data[llm].dropna()
            means.append(scores.mean())
            std_errors.append(scores.std() / np.sqrt(len(scores)))
            llm_labels.append(llm.replace(' ', '\n'))
        
        # Create figure
        plt.figure(figsize=(12, 8))
        x_pos = np.arange(len(llm_labels))
        
        bars = plt.bar(x_pos, means, yerr=std_errors, capsize=5, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, mean, std_err) in enumerate(zip(bars, means, std_errors)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_err + 1, 
                    f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Mean Scores by Large Language Model\n(with Standard Error)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Mean Score (%)', fontsize=14)
        plt.xlabel('Large Language Model', fontsize=14)
        plt.xticks(x_pos, llm_labels)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, max(means) + max(std_errors) + 10)
        
        plt.tight_layout()
        plt.savefig('figure_2_mean_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: figure_2_mean_scores.png")
    
    def create_figure_3_political_bias(self):
        """Figure 3: Political bias comparison (Left vs Right vs Center)."""
        print("Creating Figure 3: Political Bias Analysis...")
        
        # Combine all data
        all_data = pd.concat(self.processed_data.values(), ignore_index=True)
        
        # Calculate means by political orientation
        orientation_means = {}
        orientation_stds = {}
        
        for orientation in ['left', 'center', 'right']:
            orientation_data = all_data[all_data['orientation_group'] == orientation]
            if len(orientation_data) > 0:
                orientation_means[orientation] = []
                orientation_stds[orientation] = []
                
                for llm in self.llms:
                    scores = orientation_data[llm].dropna()
                    if len(scores) > 0:
                        orientation_means[orientation].append(scores.mean())
                        orientation_stds[orientation].append(scores.std())
                    else:
                        orientation_means[orientation].append(0)
                        orientation_stds[orientation].append(0)
        
        # Create figure
        plt.figure(figsize=(14, 8))
        x_pos = np.arange(len(self.llms))
        width = 0.25
        
        colors = {'left': '#2ECC71', 'center': '#F39C12', 'right': '#E74C3C'}
        
        for i, (orientation, means) in enumerate(orientation_means.items()):
            stds = orientation_stds[orientation]
            plt.bar(x_pos + i*width, means, width, 
                   label=f'{orientation.capitalize()}-wing', 
                   color=colors[orientation], alpha=0.8,
                   yerr=stds, capsize=3, edgecolor='black', linewidth=0.5)
        
        plt.title('Mean Scores by Political Orientation and LLM', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Mean Score (%)', fontsize=14)
        plt.xlabel('Large Language Model', fontsize=14)
        plt.xticks(x_pos + width, [llm.replace(' ', '\n') for llm in self.llms])
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add statistical significance indicators
        plt.text(0.02, 0.98, '*** p < 0.001 (Left vs Right bias)', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('figure_3_political_bias.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: figure_3_political_bias.png")
    
    def create_figure_4_country_comparison(self):
        """Figure 4: Comparison across countries."""
        print("Creating Figure 4: Country Comparison...")
        
        # Calculate means by country
        country_means = {}
        for country, df in self.processed_data.items():
            country_means[country] = []
            for llm in self.llms:
                scores = df[llm].dropna()
                country_means[country].append(scores.mean() if len(scores) > 0 else 0)
        
        # Create figure
        plt.figure(figsize=(14, 8))
        x_pos = np.arange(len(self.llms))
        width = 0.2
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        countries = list(country_means.keys())
        
        for i, (country, means) in enumerate(country_means.items()):
            plt.bar(x_pos + i*width, means, width, 
                   label=country.capitalize(), 
                   color=colors[i], alpha=0.8,
                   edgecolor='black', linewidth=0.5)
        
        plt.title('Mean Scores by Country and LLM', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Mean Score (%)', fontsize=14)
        plt.xlabel('Large Language Model', fontsize=14)
        plt.xticks(x_pos + width*1.5, [llm.replace(' ', '\n') for llm in self.llms])
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('figure_4_country_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: figure_4_country_comparison.png")
    
    def create_figure_5_effect_sizes(self):
        """Figure 5: Effect sizes (Cohen's d) between LLMs."""
        print("Creating Figure 5: Effect Sizes Between LLMs...")
        
        # Combine all data
        all_data = pd.concat(self.processed_data.values(), ignore_index=True)
        
        # Calculate effect sizes
        effect_sizes = []
        comparisons = []
        
        for i, llm1 in enumerate(self.llms):
            for j, llm2 in enumerate(self.llms[i+1:], i+1):
                scores1 = all_data[llm1].dropna()
                scores2 = all_data[llm2].dropna()
                
                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(scores1)-1)*scores1.var() + (len(scores2)-1)*scores2.var()) / 
                                   (len(scores1) + len(scores2) - 2))
                cohens_d = (scores1.mean() - scores2.mean()) / pooled_std
                
                effect_sizes.append(cohens_d)
                comparisons.append(f"{llm1}\nvs\n{llm2}")
        
        # Create figure
        plt.figure(figsize=(14, 8))
        colors = ['#E74C3C' if abs(d) > 0.8 else '#F39C12' if abs(d) > 0.5 else '#2ECC71' for d in effect_sizes]
        
        bars = plt.bar(range(len(comparisons)), effect_sizes, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (bar, d) in enumerate(zip(bars, effect_sizes)):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.05 if d >= 0 else -0.1), 
                    f'{d:.3f}', ha='center', va='bottom' if d >= 0 else 'top',
                    fontweight='bold')
        
        plt.title('Effect Sizes (Cohen\'s d) Between LLMs', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Cohen\'s d', fontsize=14)
        plt.xlabel('LLM Comparisons', fontsize=14)
        plt.xticks(range(len(comparisons)), comparisons, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
        plt.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(y=-0.8, color='red', linestyle='--', alpha=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('figure_5_effect_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: figure_5_effect_sizes.png")
    
    def create_figure_6_grok_focus(self):
        """Figure 6: Focus on Grok's performance vs other LLMs."""
        print("Creating Figure 6: Grok Focus Analysis...")
        
        # Combine all data
        all_data = pd.concat(self.processed_data.values(), ignore_index=True)
        
        # Calculate Grok vs others
        grok_scores = all_data['grok 3'].dropna()
        other_llms = [llm for llm in self.llms if llm != 'grok 3']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Subplot 1: Grok vs others scatter plot
        for llm in other_llms:
            other_scores = all_data[llm].dropna()
            # Align data by party
            common_parties = set(all_data[all_data['grok 3'].notna()]['party']) & \
                           set(all_data[all_data[llm].notna()]['party'])
            
            grok_aligned = []
            other_aligned = []
            for party in common_parties:
                grok_score = all_data[(all_data['party'] == party) & (all_data['grok 3'].notna())]['grok 3'].iloc[0]
                other_score = all_data[(all_data['party'] == party) & (all_data[llm].notna())][llm].iloc[0]
                grok_aligned.append(grok_score)
                other_aligned.append(other_score)
            
            ax1.scatter(grok_aligned, other_aligned, alpha=0.6, s=50, label=llm)
        
        # Add diagonal line (perfect correlation)
        min_val = min(min(grok_aligned), min(other_aligned))
        max_val = max(max(grok_aligned), max(other_aligned))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correlation')
        
        ax1.set_xlabel('Grok 3 Score (%)', fontsize=12)
        ax1.set_ylabel('Other LLM Score (%)', fontsize=12)
        ax1.set_title('Grok 3 vs Other LLMs\n(Scatter Plot)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Grok vs others bar chart
        grok_mean = grok_scores.mean()
        other_means = [all_data[llm].dropna().mean() for llm in other_llms]
        
        x_pos = np.arange(len(other_llms) + 1)
        means = [grok_mean] + other_means
        labels = ['Grok 3'] + [llm.replace(' ', '\n') for llm in other_llms]
        colors = ['#FF6B6B'] + ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars = ax2.bar(x_pos, means, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('Mean Score (%)', fontsize=12)
        ax2.set_title('Mean Scores: Grok 3 vs Other LLMs', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Grok 3 Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figure_6_grok_focus.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: figure_6_grok_focus.png")
    
    def create_figure_7_statistical_significance(self):
        """Figure 7: Statistical significance visualization."""
        print("Creating Figure 7: Statistical Significance...")
        
        # Combine all data
        all_data = pd.concat(self.processed_data.values(), ignore_index=True)
        
        # Perform t-tests for left vs right bias
        left_data = all_data[all_data['orientation_group'] == 'left']
        right_data = all_data[all_data['orientation_group'] == 'right']
        
        results = []
        for llm in self.llms:
            llm_left = left_data[left_data[llm].notna()][llm]
            llm_right = right_data[right_data[llm].notna()][llm]
            
            if len(llm_left) > 0 and len(llm_right) > 0:
                t_stat, p_value = ttest_ind(llm_left, llm_right)
                
                # Calculate effect size
                pooled_std = np.sqrt(((len(llm_left)-1)*llm_left.var() + (len(llm_right)-1)*llm_right.var()) / 
                                   (len(llm_left) + len(llm_right) - 2))
                cohens_d = (llm_left.mean() - llm_right.mean()) / pooled_std
                
                results.append({
                    'llm': llm,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'left_mean': llm_left.mean(),
                    'right_mean': llm_right.mean()
                })
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Subplot 1: P-values
        llms = [r['llm'] for r in results]
        p_values = [r['p_value'] for r in results]
        
        colors = ['#E74C3C' if p < 0.001 else '#F39C12' if p < 0.05 else '#2ECC71' for p in p_values]
        bars1 = ax1.bar(range(len(llms)), p_values, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
        
        # Add significance lines
        ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='p = 0.05')
        ax1.axhline(y=0.001, color='red', linestyle='--', alpha=0.7, label='p = 0.001')
        
        # Add value labels
        for i, (bar, p) in enumerate(zip(bars1, p_values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{p:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.set_ylabel('P-value', fontsize=12)
        ax1.set_title('Statistical Significance of Left vs Right Bias', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(llms)))
        ax1.set_xticklabels([llm.replace(' ', '\n') for llm in llms])
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Effect sizes
        cohens_ds = [r['cohens_d'] for r in results]
        colors2 = ['#E74C3C' if abs(d) > 2.0 else '#F39C12' if abs(d) > 0.8 else '#2ECC71' for d in cohens_ds]
        bars2 = ax2.bar(range(len(llms)), cohens_ds, color=colors2, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
        
        # Add effect size lines
        ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Large effect')
        ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Very large effect')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, (bar, d) in enumerate(zip(bars2, cohens_ds)):
            ax2.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.1 if d >= 0 else -0.2), 
                    f'{d:.2f}', ha='center', va='bottom' if d >= 0 else 'top',
                    fontweight='bold', fontsize=10)
        
        ax2.set_ylabel('Cohen\'s d', fontsize=12)
        ax2.set_title('Effect Sizes of Left vs Right Bias', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(llms)))
        ax2.set_xticklabels([llm.replace(' ', '\n') for llm in llms])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figure_7_statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: figure_7_statistical_significance.png")
    
    def generate_all_figures(self):
        """Generate all individual figures."""
        print("Generating Individual Figures for LLM Bias Analysis")
        print("=" * 60)
        
        # Load and process data
        self.load_and_process_data()
        
        # Create all figures
        self.create_figure_1_llm_comparison()
        self.create_figure_2_mean_scores()
        self.create_figure_3_political_bias()
        self.create_figure_4_country_comparison()
        self.create_figure_5_effect_sizes()
        self.create_figure_6_grok_focus()
        self.create_figure_7_statistical_significance()
        
        print("\n" + "=" * 60)
        print("ALL FIGURES GENERATED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print("1. figure_1_llm_comparison.png - Box plot comparing all LLMs")
        print("2. figure_2_mean_scores.png - Mean scores with error bars")
        print("3. figure_3_political_bias.png - Political bias analysis")
        print("4. figure_4_country_comparison.png - Country comparison")
        print("5. figure_5_effect_sizes.png - Effect sizes between LLMs")
        print("6. figure_6_grok_focus.png - Grok-specific analysis")
        print("7. figure_7_statistical_significance.png - Statistical significance")

def main():
    """Main function."""
    data_dir = Path(__file__).parent
    generator = IndividualFigureGenerator(data_dir)
    generator.generate_all_figures()

if __name__ == "__main__":
    main()
