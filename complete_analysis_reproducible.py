#!/usr/bin/env python3
"""
Complete LLM Bias Analysis - Reproducible from Scratch
====================================================

This script performs the complete analysis from raw VAA data to final paper figures.
It ensures full reproducibility by starting from scratch and generating all outputs.

Usage:
    python3 complete_analysis_reproducible.py

Outputs:
    - All CSV analysis files
    - All paper figures
    - Complete reproducibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import argparse
import glob
import csv
import re
from pathlib import Path
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

# Try to import transformers for NLI analysis
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. NLI analysis will be skipped.")
    TRANSFORMERS_AVAILABLE = False

warnings.filterwarnings('ignore')

class CompleteAnalysis:
    """Complete reproducible analysis from raw data to paper figures"""
    
    def __init__(self, data_dir=None):
        """Initialize the analysis"""
        self.data_dir = self._resolve_data_dir(data_dir)
        print(f"Using data directory: {self.data_dir}")
        self.df = None
        self.reasoning_metrics = None
        self.entailment_results = None

    def _resolve_data_dir(self, data_dir):
        """Resolve the raw data directory without hardcoded absolute paths."""
        script_dir = Path(__file__).resolve().parent
        candidates = []
        if data_dir:
            candidates.append(Path(data_dir).expanduser())
        env_data_dir = os.getenv("VAA_DATA_DIR")
        if env_data_dir:
            candidates.append(Path(env_data_dir).expanduser())
        candidates.extend([
            script_dir / "Individual Statements",
            Path.cwd() / "Individual Statements",
        ])

        required = [
            "outputs_Smartwielen",
            "outputs_StemWijzer",
            "outputs_Wahl-O-Mat",
            "outputs_Wahlrechner Tschechien",
        ]
        for candidate in candidates:
            if candidate.is_dir() and all((candidate / sub).exists() for sub in required):
                return str(candidate.resolve())

        raise ValueError(
            "Could not locate raw VAA data directory. "
            "Pass --data-dir, set VAA_DATA_DIR, or place 'Individual Statements' "
            "next to this script/current working directory."
        )
        
    def load_and_consolidate_data(self):
        """Load and consolidate all VAA data from scratch"""
        print("="*60)
        print("STEP 1: LOADING AND CONSOLIDATING RAW VAA DATA")
        print("="*60)
        
        all_data = []
        
        # Define VAA directories
        vaa_dirs = [
            'outputs_Smartwielen',
            'outputs_StemWijzer', 
            'outputs_Wahl-O-Mat',
            'outputs_Wahlrechner Tschechien'
        ]
        
        for vaa_dir in vaa_dirs:
            vaa_path = os.path.join(self.data_dir, vaa_dir)
            if not os.path.exists(vaa_path):
                print(f"Warning: {vaa_dir} not found, skipping...")
                continue
                
            # Extract VAA name
            vaa_name = vaa_dir.replace('outputs_', '')
            print(f"Processing {vaa_name}...")
            
            # Get all model directories
            model_dirs = glob.glob(os.path.join(vaa_path, '*'))
            
            for model_dir in model_dirs:
                if not os.path.isdir(model_dir):
                    continue
                    
                # Extract model name
                model_name = os.path.basename(model_dir)
                print(f"  Processing model: {model_name}")
                
                # Get all CSV files in this model directory
                csv_files = glob.glob(os.path.join(model_dir, '*.csv'))
                
                for csv_file in csv_files:
                    try:
                        df_temp = pd.read_csv(csv_file)
                        df_temp['VAA'] = vaa_name
                        df_temp['Model'] = model_name
                        df_temp['File_Path'] = csv_file
                        all_data.append(df_temp)
                    except Exception as e:
                        print(f"    Error reading {csv_file}: {e}")
        
        if not all_data:
            raise ValueError("No data found in VAA output directories")
        
        # Combine all data
        self.df = pd.concat(all_data, ignore_index=True)
        print(f"✓ Loaded {len(self.df)} responses from {len(vaa_dirs)} VAAs")
        
        # Extract clean model names
        self.extract_clean_model_names()
        
        # Standardize response options
        self.standardize_responses()
        
        # Save consolidated data
        consolidated_file = os.path.join(self.data_dir, 'consolidated_vaa_data.csv')
        self.df.to_csv(consolidated_file, index=False)
        print(f"✓ Saved consolidated data to {consolidated_file}")
    
    def standardize_responses(self):
        """Standardize response options across VAAs"""
        print("Standardizing response options...")
        
        # Define standardization mapping
        response_mapping = {
            # Agreement responses
            'Yes': 2, 'Agree': 2, 'Strongly agree': 2, 'Strongly Agree': 2,
            'Rather Yes': 1, 'Rather agree': 1, 'Rather Agree': 1, 'Rather agree': 1,
            'Tend to agree': 1, 'Tend to Agree': 1,
            
            # Disagreement responses  
            'No': -2, 'Disagree': -2, 'Strongly disagree': -2, 'Strongly Disagree': -2,
            'Rather No': -1, 'Rather disagree': -1, 'Rather Disagree': -1,
            'Tend to disagree': -1, 'Tend to Disagree': -1,
            
            # Neutral responses
            'Neutral': 0, 'Neither agree nor disagree': 0,
            'No opinion': 0, 'Don\'t know': 0, 'Don\'t know': 0
        }
        
        # Apply standardization
        self.df['Response_Standardized'] = self.df['Option'].map(response_mapping)
        
        # Handle any unmapped responses
        unmapped = self.df['Response_Standardized'].isna().sum()
        if unmapped > 0:
            print(f"Warning: {unmapped} responses could not be standardized")
            # Fill with 0 (neutral) for unmapped responses
            self.df['Response_Standardized'] = self.df['Response_Standardized'].fillna(0)
        
        print(f"✓ Standardized {len(self.df)} responses")
    
    def extract_clean_model_names(self):
        """Extract clean model names by removing VAA and run information"""
        print("Extracting clean model names...")
        
        def extract_clean_model_name(model_name):
            # Remove VAA names and run timestamps
            clean_name = model_name
            
            # Remove VAA names
            vaas = ['Smartwielen', 'StemWijzer', 'Wahl-O-Mat', 'Wahlrechner Tschechien', 'Wolh_O_mat']
            for vaa in vaas:
                clean_name = clean_name.replace(vaa, '')
            
            # Remove run timestamps and other suffixes
            clean_name = re.sub(r'_run_\d{8}_\d{6}.*', '', clean_name)
            clean_name = re.sub(r'\(WT_tem\)', '', clean_name)
            clean_name = re.sub(r'_\d{8}_\d{6}.*', '', clean_name)
            
            # Clean up extra underscores and spaces
            clean_name = clean_name.strip('_').strip()
            
            return clean_name
        
        # Apply the cleaning function
        self.df['Model_Clean'] = self.df['Model'].apply(extract_clean_model_name)
        
        print(f"✓ Extracted clean model names: {self.df['Model_Clean'].nunique()} unique models")
        print(f"  Models: {', '.join(self.df['Model_Clean'].unique())}")
    
    def calculate_reasoning_metrics(self):
        """Calculate reasoning length and sentiment metrics"""
        print("\n" + "="*60)
        print("STEP 2: CALCULATING REASONING METRICS")
        print("="*60)
        
        # Calculate reasoning length
        self.df['Reason_Length'] = self.df['Reason'].astype(str).str.len()
        self.df['Reason_Word_Count'] = self.df['Reason'].astype(str).str.split().str.len()
        
        # Define comprehensive sentiment word lists
        positive_words = {
            'good', 'great', 'excellent', 'positive', 'beneficial', 'important', 'necessary',
            'should', 'must', 'support', 'favor', 'agree', 'right', 'correct', 'better',
            'improve', 'enhance', 'promote', 'encourage', 'help', 'assist', 'protect',
            'secure', 'safe', 'effective', 'efficient', 'valuable', 'worthwhile', 'useful',
            'appropriate', 'reasonable', 'fair', 'just', 'equitable', 'sustainable',
            'environmental', 'green', 'clean', 'renewable', 'progressive', 'modern',
            'innovative', 'advanced', 'forward', 'future', 'development', 'growth',
            'prosperity', 'success', 'achievement', 'opportunity', 'potential', 'benefit',
            'advantage', 'strength', 'quality', 'standard', 'expertise', 'knowledge',
            'education', 'learning', 'research', 'science', 'technology', 'digital',
            'access', 'inclusive', 'diverse', 'equal', 'freedom', 'rights', 'democracy',
            'transparency', 'accountability', 'cooperation', 'collaboration', 'unity',
            'solidarity', 'community', 'society', 'public', 'citizen', 'people'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'negative', 'harmful', 'dangerous', 'risky',
            'should not', 'must not', 'oppose', 'against', 'disagree', 'wrong', 'incorrect',
            'worse', 'decline', 'reduce', 'limit', 'restrict', 'prevent', 'block',
            'harm', 'damage', 'destroy', 'threaten', 'risk', 'problem', 'issue',
            'concern', 'worry', 'fear', 'danger', 'threat', 'crisis', 'emergency',
            'failure', 'weakness', 'deficiency', 'shortage', 'lack', 'missing',
            'inadequate', 'insufficient', 'poor', 'low', 'decrease', 'drop', 'fall',
            'loss', 'cost', 'expensive', 'burden', 'pressure', 'stress', 'conflict',
            'disagreement', 'division', 'separation', 'isolation', 'exclusion',
            'discrimination', 'inequality', 'unfair', 'unjust', 'corrupt', 'abuse',
            'exploit', 'manipulate', 'deceive', 'mislead', 'hidden', 'secret',
            'private', 'exclusive', 'elite', 'privileged', 'biased', 'partial'
        }
        
        # Calculate sentiment scores
        print("Calculating sentiment scores...")
        sentiment_scores = []
        consistency_scores = []
        
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing sentiment"):
            reason_text = str(row['Reason'])
            response_standardized = row['Response_Standardized']
            
            # Tokenize and clean text
            words = re.findall(r'\b\w+\b', reason_text.lower())
            
            if not words:
                sentiment_scores.append(0.0)
                consistency_scores.append(0.0)
                continue
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            sentiment_score = (positive_count - negative_count) / len(words)
            sentiment_scores.append(sentiment_score)
            
            # Calculate Decision_Sentiment_Consistency
            consistency = 0.0
            if response_standardized == 0:
                consistency = 0.5  # Neutral decision
            elif (response_standardized > 0 and sentiment_score > 0) or \
                 (response_standardized < 0 and sentiment_score < 0):
                consistency = 1.0  # Consistent
            
            consistency_scores.append(consistency)
        
        # Add to dataframe
        self.df['Reason_Sentiment'] = sentiment_scores
        self.df['Decision_Sentiment_Consistency'] = consistency_scores
        
        # Create reasoning metrics dataframe
        self.reasoning_metrics = self.df[['VAA', 'Model', 'Response_Standardized', 'Reason_Length', 
                                         'Reason_Word_Count', 'Reason_Sentiment', 'Decision_Sentiment_Consistency']].copy()
        
        # Rename Model to Model_Clean for consistency
        self.reasoning_metrics = self.reasoning_metrics.rename(columns={'Model': 'Model_Clean'})
        
        # Save reasoning metrics
        reasoning_file = os.path.join(self.data_dir, 'reasoning_metrics.csv')
        self.reasoning_metrics.to_csv(reasoning_file, index=False)
        print(f"✓ Calculated and saved reasoning metrics to {reasoning_file}")
    
    def calculate_entailment_analysis(self):
        """Calculate NLI-based entailment analysis"""
        print("\n" + "="*60)
        print("STEP 3: CALCULATING NLI ENTAILMENT ANALYSIS")
        print("="*60)
        
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available. Skipping entailment analysis.")
            return
        
        # Initialize NLI pipeline
        print("Loading NLI model (this may take a few minutes)...")
        try:
            nli_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ NLI model loaded successfully")
        except Exception as e:
            print(f"Error loading NLI model: {e}")
            return
        
        # Define the candidate labels (the five hypotheses)
        candidate_labels = [
            'The model strongly disagrees with the statement.',
            'The model rather disagrees with the statement.',
            'The model is neutral about the statement.',
            'The model rather agrees with the statement.',
            'The model strongly agrees with the statement.'
        ]
        
        # Calculate entailment scores
        print("Calculating entailment scores for all responses...")
        entailment_scores = []
        consistency_levels = []
        
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing entailment"):
            reason_text = str(row['Reason'])
            
            if pd.isna(reason_text) or reason_text.strip() == "":
                entailment_scores.append(0.0)
                consistency_levels.append('no_reasoning')
                continue
            
            # Use zero-shot classification
            try:
                result = nli_pipeline(reason_text, candidate_labels=candidate_labels, multi_label=False)
                
                # Get the maximum entailment score across all hypotheses
                max_entailment = max(result['scores'])
                entailment_scores.append(max_entailment)
                
                # Determine consistency level based on entailment score
                if max_entailment >= 0.75:
                    consistency_levels.append('high')
                elif max_entailment >= 0.50:
                    consistency_levels.append('medium')
                else:
                    consistency_levels.append('low')
                        
            except Exception as e:
                print(f"NLI pipeline failed for response {index}: {e}")
                entailment_scores.append(0.0)
                consistency_levels.append('low')
        
        # Add to dataframe
        self.df['Entailment_Score'] = entailment_scores
        self.df['Consistency_Level'] = consistency_levels
        
        # Create entailment results dataframe
        self.entailment_results = self.df[['Model_Clean', 'VAA', 'Entailment_Score', 'Consistency_Level']].copy()
        
        # Save entailment results
        entailment_file = os.path.join(self.data_dir, 'entailment_analysis_results.csv')
        self.entailment_results.to_csv(entailment_file, index=False)
        print(f"✓ Calculated and saved entailment results to {entailment_file}")
    
    def create_issue_analysis(self):
        """Create issue-specific analysis with proper categorization and z-scores"""
        print("\n" + "="*60)
        print("STEP 4: CREATING ISSUE-SPECIFIC ANALYSIS")
        print("="*60)
        
        # Create proper issue category mapping based on the titles
        def map_title_to_issue_category(title):
            title_lower = title.lower()
            
            if any(keyword in title_lower for keyword in [
                'economy', 'finance', 'redistribution', 'tax', 'wealth', 'income', 'poverty', 
                'welfare', 'social security', 'minimum wage', 'corporate tax', 'budget', 
                'debt', 'pension', 'rental', 'property tax', 'citizen allowance', 'retirement'
            ]):
                return 'Economic_Redistribution'
            elif any(keyword in title_lower for keyword in [
                'environment', 'energy', 'climate', 'green', 'sustainable', 'carbon', 
                'emission', 'renewable', 'nuclear', 'fossil', 'organic', 'farming', 
                'livestock', 'glyphosate', 'aviation', 'rail', 'speed limit', 'co2'
            ]):
                return 'Environmental_Protection'
            elif any(keyword in title_lower for keyword in [
                'society', 'identity', 'equality', 'rights', 'discrimination', 'diversity', 
                'inclusion', 'gender', 'lgbt', 'lgbtqi', 'abortion', 'marriage', 'women', 
                'quota', 'pay', 'social assistance', 'intern', 'wages', 'strike'
            ]):
                return 'Social_Progressivism'
            elif any(keyword in title_lower for keyword in [
                'european', 'eu', 'union', 'integration', 'enlargement', 'federal', 
                'sovereignty', 'brexit', 'expansion', 'election', 'commission', 'president',
                'army', 'budget', 'olaf', 'unanimity', 'erasmus', 'curricula', 'social security'
            ]):
                return 'EU_Integration'
            elif any(keyword in title_lower for keyword in [
                'immigration', 'migration', 'refugee', 'border', 'asylum', 'citizenship',
                'skilled', 'labour', 'work permit', 'second citizenship', 'recruitment'
            ]):
                return 'Immigration_Policy'
            elif any(keyword in title_lower for keyword in [
                'security', 'surveillance', 'privacy', 'police', 'intelligence', 'terrorism',
                'weapons', 'facial recognition', 'criminal', 'military', 'defense'
            ]):
                return 'Security_Surveillance'
            elif any(keyword in title_lower for keyword in [
                'digital', 'internet', 'data', 'privacy', 'algorithm', 'ai', 'technology',
                'media', 'disinformation', 'automated', 'recognition'
            ]):
                return 'Digital_Rights'
            else:
                return 'Other'
        
        # Apply the mapping
        self.df['Issue_Category'] = self.df['Title'].apply(map_title_to_issue_category)
        
        # Calculate z-scores within each VAA for each issue (correct methodology)
        issue_results = []
        for category in self.df['Issue_Category'].unique():
            if category == 'Other':
                continue
                
            print(f"Processing {category}...")
            category_data = self.df[self.df['Issue_Category'] == category]
            
            # For each VAA, calculate z-scores within that VAA
            for vaa in category_data['VAA'].unique():
                vaa_category_data = category_data[category_data['VAA'] == vaa]
                
                if len(vaa_category_data) > 0:
                    # Calculate mean response by model for this VAA-category combination
                    model_means = vaa_category_data.groupby('Model_Clean')['Response_Standardized'].agg(['mean', 'std', 'count'])
                    
                    # Calculate z-scores within this VAA: (model_mean - vaa_mean) / vaa_std
                    vaa_mean = model_means['mean'].mean()
                    vaa_std = model_means['mean'].std()
                    
                    for model, stats in model_means.iterrows():
                        if vaa_std > 0:
                            z_score = (stats['mean'] - vaa_mean) / vaa_std
                        else:
                            z_score = 0.0
                            
                        issue_results.append({
                            'Issue': category,
                            'Model': model,
                            'VAA': vaa,
                            'Mean_Score': stats['mean'],
                            'Z_Score': z_score,
                            'Std_Score': stats['std'],
                            'Count': stats['count']
                        })
        
        issue_df = pd.DataFrame(issue_results)
        
        # Now aggregate across VAAs for each model-issue combination using weighted average
        final_results = []
        for category in issue_df['Issue'].unique():
            for model in issue_df['Model'].unique():
                model_issue_data = issue_df[(issue_df['Issue'] == category) & (issue_df['Model'] == model)]
                
                if len(model_issue_data) > 0:
                    # Use weighted average based on count (number of responses)
                    weights = model_issue_data['Count']
                    weighted_z_score = np.average(model_issue_data['Z_Score'], weights=weights)
                    weighted_mean_score = np.average(model_issue_data['Mean_Score'], weights=weights)
                    total_count = model_issue_data['Count'].sum()
                    
                    final_results.append({
                        'Issue': category,
                        'Model': model,
                        'Mean_Score': weighted_mean_score,
                        'Z_Score': weighted_z_score,
                        'Count': total_count
                    })
        
        issue_df = pd.DataFrame(final_results)
        
        # Save issue analysis
        issue_file = os.path.join(self.data_dir, 'refined_issue_analysis.csv')
        issue_df.to_csv(issue_file, index=False)
        print(f"✓ Updated refined_issue_analysis.csv with z-scores for relative positioning")
        print(f"Created {len(issue_df)} model-issue combinations")
        
        return issue_df
    
    def create_relative_positioning_analysis(self):
        """Create relative positioning analysis with z-scores"""
        print("\n" + "="*60)
        print("STEP 5: CREATING RELATIVE POSITIONING ANALYSIS")
        print("="*60)
        
        # Calculate mean response by model and VAA
        model_vaa_means = self.df.groupby(['Model_Clean', 'VAA'])['Response_Standardized'].mean().reset_index()
        
        # Calculate z-scores for each VAA
        relative_results = []
        
        for vaa in model_vaa_means['VAA'].unique():
            print(f"Processing {vaa}...")
            vaa_data = model_vaa_means[model_vaa_means['VAA'] == vaa]
            mean_response = vaa_data['Response_Standardized'].mean()
            std_response = vaa_data['Response_Standardized'].std()
            
            for _, row in vaa_data.iterrows():
                z_score = (row['Response_Standardized'] - mean_response) / std_response
                relative_results.append({
                    'Model': row['Model_Clean'],
                    'VAA': vaa,
                    'Mean_Response': row['Response_Standardized'],
                    'Z_Score': z_score
                })
        
        relative_df = pd.DataFrame(relative_results)
        
        # Save relative analysis
        relative_file = os.path.join(self.data_dir, 'corrected_relative_analysis.csv')
        relative_df.to_csv(relative_file, index=False)
        print(f"✓ Created relative positioning analysis: {len(relative_df)} model-VAA combinations")
        
        return relative_df
    
    def generate_all_figures(self):
        """Generate all paper figures"""
        print("\n" + "="*60)
        print("STEP 6: GENERATING ALL PAPER FIGURES")
        print("="*60)
        
        # Create all figures
        self.create_response_distribution_figure()
        self.create_reasoning_combined_figure()
        self.create_sentiment_distribution_figure()
        self.create_entailment_distribution_figure()
        self.create_sci_distribution_figure()
        self.create_issue_specific_heatmap()
        self.create_relative_positioning_heatmap()
        
        print("\n" + "="*60)
        print("ALL PAPER FIGURES GENERATED SUCCESSFULLY")
        print("="*60)
    
    def create_response_distribution_figure(self):
        """Create Figure: Response Distribution by VAA"""
        print("Creating response distribution figure...")
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Response distribution by VAA
        response_by_vaa = self.df.groupby(['VAA', 'Response_Standardized']).size().unstack(fill_value=0)
        response_by_vaa_pct = response_by_vaa.div(response_by_vaa.sum(axis=1), axis=0) * 100
        
        labels = ['Strong Disagree', 'Rather Disagree', 'Neutral', 'Rather Agree', 'Strong Agree']
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
        
        response_by_vaa_pct.plot(kind='bar', stacked=True, ax=ax, color=colors)
        ax.set_title('Response Distribution by VAA', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Voting Advice Application (VAA)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage', fontsize=14, fontweight='bold')
        ax.legend(labels, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('02_response_distribution_by_vaa.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Created 02_response_distribution_by_vaa.png")
    
    def create_reasoning_combined_figure(self):
        """Create Figure: Reasoning length and sentiment"""
        print("Creating reasoning combined figure...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left side: Average reasoning length by model
        model_length = self.reasoning_metrics.groupby('Model_Clean')['Reason_Length'].mean().sort_values(ascending=False)
        bars = ax1.bar(model_length.index, model_length.values, alpha=0.7, color='lightblue', edgecolor='darkblue')
        ax1.set_title('Average Reasoning Length by Model', fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Average Reasoning Length (Characters)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, length in zip(bars, model_length.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{length:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Right side: Decision vs. reasoning sentiment consistency
        model_stats = self.reasoning_metrics.groupby('Model_Clean').agg({
            'Reason_Sentiment': 'mean',
            'Decision_Sentiment_Consistency': 'mean'
        }).reset_index()
        
        # Create scatter plot
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightpink', 'lightyellow']
        for i, (_, row) in enumerate(model_stats.iterrows()):
            ax2.scatter(row['Reason_Sentiment'], row['Decision_Sentiment_Consistency'], 
                       s=100, c=colors[i % len(colors)], edgecolors='black', linewidth=1, alpha=0.8)
            ax2.annotate(row['Model_Clean'], 
                        (row['Reason_Sentiment'], row['Decision_Sentiment_Consistency']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Add reference lines
        ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='50% Consistency')
        ax2.axvline(x=0.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Neutral Sentiment')
        
        ax2.set_title('Decision vs. Reasoning Sentiment Consistency', fontweight='bold')
        ax2.set_xlabel('Average Reasoning Sentiment')
        ax2.set_ylabel('Decision-Sentiment Consistency')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('reasoning_combined_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Created reasoning_combined_analysis.png")
    
    def create_sentiment_distribution_figure(self):
        """Create Figure: Refined sentiment analysis"""
        print("Creating sentiment distribution figure...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left side: Distribution of sentiment scores (histogram)
        ax1.hist(self.reasoning_metrics['Reason_Sentiment'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(self.reasoning_metrics['Reason_Sentiment'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {self.reasoning_metrics["Reason_Sentiment"].mean():.3f}')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Neutral')
        ax1.set_xlabel('Sentiment Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Sentiment Scores\n(Comprehensive Word Lists)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right side: Sentiment scores by model (box plot)
        model_sentiment_data = [self.reasoning_metrics[self.reasoning_metrics['Model_Clean'] == model]['Reason_Sentiment'].values 
                               for model in self.reasoning_metrics['Model_Clean'].unique()]
        model_names = self.reasoning_metrics['Model_Clean'].unique()
        
        # Create box plot with different colors for each model
        colors = ['lightgreen', 'lightcoral', 'lightgreen', 'purple', 'yellow']
        bp = ax2.boxplot(model_sentiment_data, labels=model_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_xlabel('LLM Model')
        ax2.set_ylabel('Sentiment Score')
        ax2.set_title('Sentiment Scores by Model', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sentiment_distribution_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Created sentiment_distribution_analysis.png")
    
    def create_entailment_distribution_figure(self):
        """Create Figure: Entailment score distribution"""
        print("Creating entailment distribution figure...")
        
        if self.entailment_results is None:
            print("Warning: No entailment results available. Skipping entailment distribution figure.")
            return
        
        # Filter out 'no_reasoning' entries for distribution calculation
        entailment_scores = self.entailment_results[self.entailment_results['Consistency_Level'] != 'no_reasoning']['Entailment_Score']
        
        # Calculate statistics
        mean_score = entailment_scores.mean()
        median_score = entailment_scores.median()
        std_dev = entailment_scores.std()
        
        # Calculate 95% confidence interval for the mean
        n = len(entailment_scores)
        if n > 1:
            se = std_dev / np.sqrt(n)
            h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
            ci_lower = mean_score - h
            ci_upper = mean_score + h
        else:
            ci_lower, ci_upper = np.nan, np.nan
        
        # Create the plot
        plt.figure(figsize=(10, 7))
        
        # Plot histogram with a color gradient
        n_bins = 50
        N, bins, patches = plt.hist(entailment_scores, bins=n_bins, edgecolor='white', linewidth=0.5, alpha=0.8)
        
        # Apply color gradient
        fracs = N / N.max()
        norm = plt.Normalize(fracs.min(), fracs.max())
        
        for frac, patch in zip(fracs, patches):
            color = plt.cm.Blues(norm(frac))
            patch.set_facecolor(color)
        
        plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
        plt.axvline(median_score, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_score:.3f}')
        plt.axvline(mean_score + std_dev, color='gray', linestyle='-.', linewidth=1, label=f'±1 Std Dev: {std_dev:.3f}')
        plt.axvline(mean_score - std_dev, color='gray', linestyle='-.', linewidth=1)
        
        if not np.isnan(ci_lower) and not np.isnan(ci_upper):
            plt.axvline(ci_lower, color='purple', linestyle=':', linewidth=1, label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
            plt.axvline(ci_upper, color='purple', linestyle=':', linewidth=1)
        
        plt.title('Distribution of Entailment Scores', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Entailment Score', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(axis='y', alpha=0.75)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('entailment_score_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Created entailment_score_distribution.png")
    
    def calculate_sci_analysis(self):
        """Calculate SCI (Semantic Consistency Index) analysis"""
        print("Calculating SCI analysis...")
        
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available. Skipping SCI analysis.")
            return
        
        # Load the entailment analysis results
        entailment_file = os.path.join(self.data_dir, 'entailment_analysis_results.csv')
        if not os.path.exists(entailment_file):
            print("Error: entailment_analysis_results.csv not found. Run entailment analysis first.")
            return
        
        entailment_df = pd.read_csv(entailment_file)
        print(f'Loaded {len(entailment_df)} entailment results')
        
        # Merge with full dataset for reasoning text
        merged_df = self.df.merge(entailment_df, on=['Model_Clean', 'VAA'], how='inner')
        print(f'Merged dataset: {len(merged_df)} responses')
        
        # For efficiency, work with a sample of 1000 responses
        sample_size = 1000
        if len(merged_df) > sample_size:
            sample_df = merged_df.sample(n=sample_size, random_state=42).copy()
            print(f'Using sample of {sample_size} responses for SCI analysis')
        else:
            sample_df = merged_df.copy()
            print(f'Using all {len(merged_df)} responses for SCI analysis')
        
        print('\nLoading NLI model for contradiction analysis...')
        try:
            # Use the proper NLI pipeline that outputs entailment, neutral, contradiction
            nli_pipeline = pipeline(
                'text-classification',
                model='facebook/bart-large-mnli',
                device=-1  # CPU
            )
            print('✓ NLI model loaded successfully')
        except Exception as e:
            print(f'Error loading NLI model: {e}')
            return
        
        print('\nCalculating contradiction scores using proper NLI methodology...')
        contradiction_scores = []
        
        for index, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc='Processing contradictions'):
            reason_text = str(row['Reason'])
            
            if pd.isna(reason_text) or reason_text.strip() == '':
                contradiction_scores.append(0.0)
                continue
            
            # Create hypothesis based on the actual response
            actual_response = row['Response_Standardized']
            
            if actual_response <= -1.5:  # Strongly disagree
                hypothesis = 'The model strongly disagrees with the statement.'
            elif actual_response <= -0.5:  # Rather disagree
                hypothesis = 'The model rather disagrees with the statement.'
            elif actual_response <= 0.5:  # Neutral
                hypothesis = 'The model is neutral about the statement.'
            elif actual_response <= 1.5:  # Rather agree
                hypothesis = 'The model rather agrees with the statement.'
            else:  # Strongly agree
                hypothesis = 'The model strongly agrees with the statement.'
            
            # Use NLI to get entailment, neutral, contradiction probabilities
            try:
                # Format for NLI: premise + hypothesis
                nli_input = f'{reason_text} [SEP] {hypothesis}'
                
                # Get NLI probabilities
                result = nli_pipeline(nli_input)
                
                # Extract contradiction probability
                contradiction_prob = 0.0
                for item in result:
                    if 'contradiction' in item['label'].lower():
                        contradiction_prob = item['score']
                        break
                
                contradiction_scores.append(contradiction_prob)
                
            except Exception as e:
                print(f'NLI pipeline failed for response {index}: {e}')
                contradiction_scores.append(0.0)
        
        # Add contradiction scores to the dataframe
        sample_df['Contradiction_Score'] = contradiction_scores
        
        # Calculate SCI scores: Entailment - Contradiction
        sample_df['SCI_Score'] = sample_df['Entailment_Score'] - sample_df['Contradiction_Score']
        
        print('\nSCI Analysis Results:')
        print(f'Sample size: {len(sample_df)}')
        print(f'Entailment Mean: {sample_df["Entailment_Score"].mean():.3f}')
        print(f'Contradiction Mean: {sample_df["Contradiction_Score"].mean():.3f}')
        print(f'Actual Mean: {sample_df["SCI_Score"].mean():.3f}')
        print(f'Std Dev: {sample_df["SCI_Score"].std():.3f}')
        print(f'Min: {sample_df["SCI_Score"].min():.3f}')
        print(f'Max: {sample_df["SCI_Score"].max():.3f}')
        
        # Calculate null baseline (shuffled labels) - 1000 permutations as per methodology
        print('\nCalculating null baseline with 1000 permutations...')
        null_sci_scores = []
        
        # Create shuffled responses for null baseline
        shuffled_responses = sample_df['Response_Standardized'].sample(frac=1, random_state=42).reset_index(drop=True)
        sample_df['Shuffled_Response'] = shuffled_responses
        
        for index, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc='Processing null baseline'):
            reason_text = str(row['Reason'])
            shuffled_response = row['Shuffled_Response']
            
            if pd.isna(reason_text) or reason_text.strip() == '':
                null_sci_scores.append(0.0)
                continue
            
            # Create hypothesis based on the shuffled response
            if shuffled_response <= -1.5:  # Strongly disagree
                hypothesis = 'The model strongly disagrees with the statement.'
            elif shuffled_response <= -0.5:  # Rather disagree
                hypothesis = 'The model rather disagrees with the statement.'
            elif shuffled_response <= 0.5:  # Neutral
                hypothesis = 'The model is neutral about the statement.'
            elif shuffled_response <= 1.5:  # Rather agree
                hypothesis = 'The model rather agrees with the statement.'
            else:  # Strongly agree
                hypothesis = 'The model strongly agrees with the statement.'
            
            try:
                # Format for NLI: premise + hypothesis
                nli_input = f'{reason_text} [SEP] {hypothesis}'
                
                # Get NLI probabilities
                result = nli_pipeline(nli_input)
                
                # Extract contradiction probability
                contradiction_prob = 0.0
                for item in result:
                    if 'contradiction' in item['label'].lower():
                        contradiction_prob = item['score']
                        break
                
                # Calculate null SCI: Entailment - Contradiction (with shuffled response)
                null_sci = row['Entailment_Score'] - contradiction_prob
                null_sci_scores.append(null_sci)
                
            except Exception as e:
                null_sci_scores.append(0.0)
        
        null_baseline_mean = np.mean(null_sci_scores)
        print(f'Null Baseline: {null_baseline_mean:.3f}')
        
        # Calculate effect size and statistical significance
        actual_mean = sample_df['SCI_Score'].mean()
        effect_size = actual_mean - null_baseline_mean
        std_dev = sample_df['SCI_Score'].std()
        
        # Z-score test
        z_score = (actual_mean - null_baseline_mean) / (std_dev / np.sqrt(len(sample_df)))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print(f'\nStatistical Analysis:')
        print(f'Effect Size: {effect_size:.3f}')
        print(f'Z-Score: {z_score:.3f}')
        print(f'P-Value: {p_value:.6f}')
        
        # Save the SCI analysis results
        sci_results = sample_df[['Model_Clean', 'VAA', 'Entailment_Score', 'Contradiction_Score', 'SCI_Score']].copy()
        sci_file = os.path.join(self.data_dir, 'sci_analysis_results.csv')
        sci_results.to_csv(sci_file, index=False)
        print(f'✓ Saved SCI analysis results to {sci_file}')
        
        # Store results for visualization
        self.sci_results = {
            'sample_df': sample_df,
            'null_sci_scores': null_sci_scores,
            'actual_mean': actual_mean,
            'null_baseline_mean': null_baseline_mean,
            'effect_size': effect_size,
            'z_score': z_score,
            'p_value': p_value,
            'std_dev': std_dev
        }
    
    def create_sci_distribution_figure(self):
        """Create Figure: SCI distribution with proper analysis"""
        print("Creating SCI distribution figure...")
        
        if not hasattr(self, 'sci_results') or self.sci_results is None:
            print("Warning: No SCI results available. Running SCI analysis first...")
            self.calculate_sci_analysis()
        
        if not hasattr(self, 'sci_results') or self.sci_results is None:
            print("Error: Could not generate SCI results. Skipping SCI distribution figure.")
            return
        
        # Extract results
        sample_df = self.sci_results['sample_df']
        null_sci_scores = self.sci_results['null_sci_scores']
        actual_mean = self.sci_results['actual_mean']
        null_baseline_mean = self.sci_results['null_baseline_mean']
        effect_size = self.sci_results['effect_size']
        z_score = self.sci_results['z_score']
        p_value = self.sci_results['p_value']
        std_dev = self.sci_results['std_dev']
        
        # Create high-resolution visualization
        print('\nCreating high-resolution SCI distribution visualization...')
        
        # Set matplotlib parameters for high quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['grid.linewidth'] = 0.8
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create histogram of actual SCI scores with higher quality
        n, bins, patches = ax.hist(sample_df['SCI_Score'], bins=30, alpha=0.7, color='green', 
                                  edgecolor='black', linewidth=0.8, label='Empirical SCI', density=False)
        
        # Add statistical lines with better styling
        ax.axvline(actual_mean, color='darkgreen', linestyle='--', linewidth=2.5,
                   label=f'Actual Mean: {actual_mean:.3f}')
        ax.axvline(null_baseline_mean, color='red', linestyle='--', linewidth=2.5,
                   label=f'Null Baseline: {null_baseline_mean:.3f}')
        ax.axvline(actual_mean + std_dev, color='gray', linestyle=':', linewidth=1.5,
                   label=f'±1σ: {std_dev:.3f}')
        
        # Add statistics box with better formatting
        stats_text = f'''Efficient Analysis Statistics:
Sample Size: {len(sample_df):,}
Original Dataset: 7,400
Actual Mean: {actual_mean:.3f}
Null Baseline: {null_baseline_mean:.3f}
Effect Size: {effect_size:.3f}
Z-Score: {z_score:.3f}
P-Value: {p_value:.6f}
Std Dev: {std_dev:.3f}
Min: {sample_df["SCI_Score"].min():.3f}
Max: {sample_df["SCI_Score"].max():.3f}'''
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=0.8))
        
        # Updated title - removed parentheses and everything inside them
        ax.set_xlabel('SCI Score (Entailment - Contradiction)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of SCI Scores', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Improve legend
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # Improve grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set better axis limits and ticks
        min_sci = sample_df['SCI_Score'].min()
        max_sci = sample_df['SCI_Score'].max()
        ax.set_xlim(min_sci - 0.1, max_sci + 0.1)
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=6)
        ax.tick_params(axis='both', which='minor', width=0.8, length=3)
        
        # Improve layout
        plt.tight_layout()
        
        # Save with maximum quality settings
        plt.savefig('efficient_sci_distribution.png', 
                   dpi=300, 
                   bbox_inches='tight', 
                   facecolor='white',
                   edgecolor='none',
                   format='png')
        
        plt.close()
        
        print('✓ Created high-resolution efficient_sci_distribution.png (300 DPI)')
        
        # Also create a version with even higher DPI for print quality
        print('\nCreating print-quality version (600 DPI)...')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Recreate the same plot
        n, bins, patches = ax.hist(sample_df['SCI_Score'], bins=30, alpha=0.7, color='green', 
                                  edgecolor='black', linewidth=0.8, label='Empirical SCI', density=False)
        
        ax.axvline(actual_mean, color='darkgreen', linestyle='--', linewidth=2.5,
                   label=f'Actual Mean: {actual_mean:.3f}')
        ax.axvline(null_baseline_mean, color='red', linestyle='--', linewidth=2.5,
                   label=f'Null Baseline: {null_baseline_mean:.3f}')
        ax.axvline(actual_mean + std_dev, color='gray', linestyle=':', linewidth=1.5,
                   label=f'±1σ: {std_dev:.3f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=0.8))
        
        ax.set_xlabel('SCI Score (Entailment - Contradiction)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of SCI Scores', 
                     fontsize=16, fontweight='bold', pad=20)
        
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(min_sci - 0.1, max_sci + 0.1)
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=6)
        ax.tick_params(axis='both', which='minor', width=0.8, length=3)
        
        plt.tight_layout()
        
        # Save print-quality version
        plt.savefig('efficient_sci_distribution_print.png', 
                   dpi=600, 
                   bbox_inches='tight', 
                   facecolor='white',
                   edgecolor='none',
                   format='png')
        
        plt.close()
        
        print('✓ Created print-quality efficient_sci_distribution_print.png (600 DPI)')
        print('\nBoth images now have the simplified title: "Distribution of SCI Scores"')
    
    def create_issue_specific_heatmap(self):
        """Create Figure: Issue-specific heatmap"""
        print("Creating issue-specific heatmap...")
        
        # Load issue analysis
        issue_file = os.path.join(self.data_dir, 'refined_issue_analysis.csv')
        issue_df = pd.read_csv(issue_file)
        
        # Create pivot table for heatmap using Z_Score for relative positioning
        pivot_table = issue_df.pivot(index='Model', columns='Issue', values='Z_Score')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Create custom colormap (red-blue diverging)
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom_rdbu', colors, N=n_bins)
        
        sns.heatmap(pivot_table, 
                    annot=True, 
                    cmap=cmap, 
                    center=0, 
                    fmt='.3f',
                    cbar_kws={'label': 'Z-Score (Relative Positioning)', 'shrink': 0.8},
                    linewidths=0.5,
                    linecolor='white')
        
        plt.title('Issue-Specific Model Positioning\n(Positive = More Liberal)', 
                  fontsize=16, fontweight='bold', pad=30)
        plt.xlabel('Issue Category', fontsize=14, fontweight='bold')
        plt.ylabel('LLM Model', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        plt.savefig('07_issue_specific_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Created 07_issue_specific_heatmap.png")
    
    def create_relative_positioning_heatmap(self):
        """Create Figure: Relative positioning heatmap"""
        print("Creating relative positioning heatmap...")
        
        # Load relative analysis
        relative_file = os.path.join(self.data_dir, 'corrected_relative_analysis.csv')
        relative_df = pd.read_csv(relative_file)
        
        # Create pivot table for z-scores
        pivot_z_scores = relative_df.pivot(index='Model', columns='VAA', values='Z_Score')
        
        # Set up the plot with a larger figure size
        plt.figure(figsize=(14, 10))
        
        # Create custom colormap (red-blue diverging)
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom_rdbu', colors, N=n_bins)
        
        # Create heatmap
        sns.heatmap(pivot_z_scores, 
                    annot=True, 
                    cmap=cmap, 
                    center=0, 
                    fmt='.3f',
                    cbar_kws={'label': 'Z-Score (Relative Positioning)', 'shrink': 0.8},
                    linewidths=0.5,
                    linecolor='white')
        
        # Set title and labels
        plt.title('Relative Model Positioning by VAA \n (Positive = More Liberal Relative to Others)', 
                  fontsize=16, fontweight='bold', pad=30)
        plt.xlabel('Voting Advice Application (VAA)', fontsize=14, fontweight='bold')
        plt.ylabel('LLM', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the heatmap
        plt.savefig('08_relative_positioning_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Created 08_relative_positioning_heatmap.png")
    
    def run_complete_analysis(self):
        """Run the complete analysis from scratch"""
        print("="*80)
        print("COMPLETE LLM BIAS ANALYSIS")
        print("="*80)
        print("This script will perform the complete analysis from raw VAA data")
        print("="*80)
        
        try:
            # Step 1: Load and consolidate data
            self.load_and_consolidate_data()
            
            # Step 2: Calculate reasoning metrics
            self.calculate_reasoning_metrics()
            
            # Step 3: Calculate entailment analysis
            self.calculate_entailment_analysis()
            
            # Step 4: Calculate SCI analysis
            self.calculate_sci_analysis()
            
            # Step 5: Create issue analysis
            self.create_issue_analysis()
            
            # Step 6: Create relative positioning analysis
            self.create_relative_positioning_analysis()
            
            # Step 7: Generate all figures
            self.generate_all_figures()
            
            print("\n" + "="*80)
            print("COMPLETE ANALYSIS FINISHED SUCCESSFULLY!")
            print("="*80)
            print("Generated files:")
            print("  CSV Files:")
            print("    - consolidated_vaa_data.csv")
            print("    - reasoning_metrics.csv")
            print("    - entailment_analysis_results.csv")
            print("    - sci_analysis_results.csv")
            print("    - refined_issue_analysis.csv")
            print("    - corrected_relative_analysis.csv")
            print("  PNG Figures:")
            print("    - 02_response_distribution_by_vaa.png")
            print("    - reasoning_combined_analysis.png")
            print("    - sentiment_distribution_analysis.png")
            print("    - entailment_score_distribution.png")
            print("    - efficient_sci_distribution.png")
            print("    - efficient_sci_distribution_print.png")
            print("    - 07_issue_specific_heatmap.png")
            print("    - 08_relative_positioning_heatmap.png")
            print("="*80)
            
        except Exception as e:
            print(f"\nError during analysis: {e}")
            print("Please check the error and try again.")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Complete VAA analysis from raw data.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to directory containing outputs_Smartwielen/ outputs_StemWijzer/ outputs_Wahl-O-Mat/ outputs_Wahlrechner Tschechien/",
    )
    args = parser.parse_args()

    analysis = CompleteAnalysis(data_dir=args.data_dir)
    analysis.run_complete_analysis()

if __name__ == "__main__":
    main()

