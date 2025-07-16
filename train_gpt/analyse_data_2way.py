import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from scipy import stats
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data with better error handling"""
    downloads = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    for resource_path, resource_name in downloads:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name, quiet=True)

download_nltk_data()

class ChildSpeechAnalyzer:
    def __init__(self, real_data_path, synthetic_data_path):
        """
        Initialize analyzer with paths to JSON files containing utterances
        Expected format: [{"text": "utterance1"}, {"text": "utterance2"}, ...]
        """
        self.real_data = self.load_data(real_data_path)
        self.synthetic_data = self.load_data(synthetic_data_path)
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, file_path):
        """Load utterances from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [item['text'] for item in data if 'text' in item and item['text'].strip()]
    
    def calculate_lexical_diversity(self, texts):
        """Calculate various lexical diversity metrics"""
        all_words = []
        all_sentences = []
        
        for text in texts:
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha()]  # Only alphabetic tokens
            all_words.extend(words)
            all_sentences.extend(text)
        
        # Type-Token Ratio (TTR)
        unique_words = set(all_words)
        ttr = len(unique_words) / len(all_words) if all_words else 0
        
        # Moving Average Type-Token Ratio (MATTR) - simplified version
        # Calculate TTR for chunks of 100 words
        chunk_size = 200
        ttrs = []
        for i in range(0, len(all_words) - chunk_size + 1, chunk_size):
            chunk = all_words[i:i + chunk_size]
            chunk_unique = set(chunk)
            ttrs.append(len(chunk_unique) / len(chunk))
        mattr = np.mean(ttrs) if ttrs else 0
        
        # Vocabulary size
        vocab_size = len(unique_words)
        
        # Average word frequency
        word_freq = Counter(all_words)
        avg_word_freq = np.mean(list(word_freq.values())) if word_freq else 0
        
        return {
            'ttr': ttr,
            'mattr': mattr,
            'vocab_size': vocab_size,
            'total_words': len(all_words),
            'avg_word_freq': avg_word_freq,
            'total_utterances': len(texts)
        }
    
    def calculate_syntactic_complexity(self, texts):
        """Calculate syntactic complexity measures"""
        utterance_lengths = []
        pos_tags_all = []
        flesch_score = []
        flesch_kincaid = []
        
        for text in texts:
            # Utterance-level metrics
            words = word_tokenize(text)
            utterance_lengths.append(len(words))
            
            # POS tags
            #pos_tags = pos_tag(words_alpha)
            #pos_tags_all.extend([tag for word, tag in pos_tags])
        
            flesch_score.append(textstat.flesch_reading_ease(text))
            flesch_kincaid.append(textstat.flesch_kincaid_grade(text))
        
        return {
            'mean_utterance_length': np.mean(utterance_lengths) if utterance_lengths else 0,
            'std_utterance_length': np.std(utterance_lengths) if utterance_lengths else 0,
            'flesch_reading_ease': np.mean(flesch_score),
            'flesch_reading_ease_std': np.std(flesch_score),
            'flesch_kincaid_grade': np.mean(flesch_kincaid),
            'flesch_kincaid_grade_std': np.std(flesch_kincaid),
            #'pos_tag_diversity': len(set(pos_tags_all)),
            #'total_pos_tags': len(pos_tags_all)
        }
    
    def analyze_vocabulary_overlap(self):
        """Analyze vocabulary overlap between real and synthetic data"""
        # Get vocabulary from both datasets
        real_words = set()
        synthetic_words = set()
        
        for text in self.real_data:
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            real_words.update(words)
        
        for text in self.synthetic_data:
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            synthetic_words.update(words)
        
        # Calculate overlap metrics
        intersection = real_words.intersection(synthetic_words)
        union = real_words.union(synthetic_words)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        overlap_real = len(intersection) / len(real_words) if real_words else 0
        overlap_synthetic = len(intersection) / len(synthetic_words) if synthetic_words else 0
        
        return {
            'real_vocab_size': len(real_words),
            'synthetic_vocab_size': len(synthetic_words),
            'shared_vocab_size': len(intersection),
            'jaccard_similarity': jaccard_similarity,
            'overlap_in_real': overlap_real,
            'overlap_in_synthetic': overlap_synthetic,
            'unique_to_real': len(real_words - synthetic_words),
            'unique_to_synthetic': len(synthetic_words - real_words)
        }
    
    def analyze_content_diversity(self):
        """Analyze content diversity using TF-IDF and topic distribution"""
        # Combine all data for TF-IDF analysis
        all_texts = self.real_data + self.synthetic_data
        labels = ['real'] * len(self.real_data) + ['synthetic'] * len(self.synthetic_data)
        
        # TF-IDF analysis
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Split back into real and synthetic
        real_tfidf = tfidf_matrix[:len(self.real_data)]
        synthetic_tfidf = tfidf_matrix[len(self.real_data):]
        
        # Calculate average cosine similarity within each group
        real_similarities = []
        synthetic_similarities = []
        
        # Sample to avoid memory issues with large datasets
        sample_size = min(100, min(len(self.real_data), len(self.synthetic_data)))
        
        if real_tfidf.shape[0] > 1:
            real_sample = real_tfidf[:sample_size]
            real_sim_matrix = cosine_similarity(real_sample)
            real_similarities = real_sim_matrix[np.triu_indices_from(real_sim_matrix, k=1)]
        
        if synthetic_tfidf.shape[0] > 1:
            synthetic_sample = synthetic_tfidf[:sample_size]
            synthetic_sim_matrix = cosine_similarity(synthetic_sample)
            synthetic_similarities = synthetic_sim_matrix[np.triu_indices_from(synthetic_sim_matrix, k=1)]
        
        # Cross-group similarity (how similar are real and synthetic)
        cross_similarities = []
        if real_tfidf.shape[0] > 0 and synthetic_tfidf.shape[0] > 0:
            cross_sim_matrix = cosine_similarity(real_tfidf[:sample_size], synthetic_tfidf[:sample_size])
            cross_similarities = cross_sim_matrix.flatten()
        
        return {
            'real_internal_similarity_mean': np.mean(real_similarities) if len(real_similarities) > 0 else 0,
            'real_internal_similarity_std': np.std(real_similarities) if len(real_similarities) > 0 else 0,
            'synthetic_internal_similarity_mean': np.mean(synthetic_similarities) if len(synthetic_similarities) > 0 else 0,
            'synthetic_internal_similarity_std': np.std(synthetic_similarities) if len(synthetic_similarities) > 0 else 0,
            'cross_similarity_mean': np.mean(cross_similarities) if len(cross_similarities) > 0 else 0,
            'cross_similarity_std': np.std(cross_similarities) if len(cross_similarities) > 0 else 0
        }
    
    """
    def compare_distributions(self):
        # Utterance lengths
        real_lengths = [len(word_tokenize(text)) for text in self.real_data]
        synthetic_lengths = [len(word_tokenize(text)) for text in self.synthetic_data]
        
        # Statistical tests
        length_ks_stat, length_ks_p = stats.ks_2samp(real_lengths, synthetic_lengths)
        length_mannwhitney_stat, length_mannwhitney_p = stats.mannwhitneyu(real_lengths, synthetic_lengths, alternative='two-sided')
        
        return {
            'real_length_mean': np.mean(real_lengths),
            'real_length_std': np.std(real_lengths),
            'synthetic_length_mean': np.mean(synthetic_lengths),
            'synthetic_length_std': np.std(synthetic_lengths),
            'length_ks_statistic': length_ks_stat,
            'length_ks_pvalue': length_ks_p,
            'length_mannwhitney_statistic': length_mannwhitney_stat,
            'length_mannwhitney_pvalue': length_mannwhitney_p
        }
    """
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        print("=" * 60)
        print("CHILD SPEECH DATA COMPARISON REPORT")
        print("=" * 60)
        
        # Basic statistics
        print(f"\nBASIC STATISTICS:")
        print(f"Real data utterances: {len(self.real_data)}")
        print(f"Synthetic data utterances: {len(self.synthetic_data)}")
        
        # Lexical diversity
        print(f"\nLEXICAL DIVERSITY:")
        real_lexical = self.calculate_lexical_diversity(self.real_data)
        synthetic_lexical = self.calculate_lexical_diversity(self.synthetic_data)
        
        print(f"Real data TTR: {real_lexical['ttr']:.4f}")
        print(f"Synthetic data TTR: {synthetic_lexical['ttr']:.4f}")
        print(f"Real data vocabulary size: {real_lexical['vocab_size']}")
        print(f"Synthetic data vocabulary size: {synthetic_lexical['vocab_size']}")
        print(f"Real data MATTR: {real_lexical['mattr']:.4f}")
        print(f"Synthetic data MATTR: {synthetic_lexical['mattr']:.4f}")
        
        # Syntactic complexity
        print(f"\nSYNTACTIC COMPLEXITY:")
        real_syntax = self.calculate_syntactic_complexity(self.real_data)
        synthetic_syntax = self.calculate_syntactic_complexity(self.synthetic_data)
        
        print(f"Real data mean utterance length: {real_syntax['mean_utterance_length']:.2f} ± {real_syntax['std_utterance_length']:.2f}")
        print(f"Synthetic data mean utterance length: {synthetic_syntax['mean_utterance_length']:.2f} ± {synthetic_syntax['std_utterance_length']:.2f}")
        print(f"Real data flesch reading ease: {real_syntax['flesch_reading_ease']:.2f} ± {real_syntax['flesch_reading_ease_std']:.2f}")
        print(f"Synthetic data flesch reading ease: {synthetic_syntax['flesch_reading_ease']:.2f} ± {synthetic_syntax['flesch_reading_ease_std']:.2f}")
        print(f"Real data kincaid grade: {real_syntax['flesch_kincaid_grade']:.2f} ± {real_syntax['flesch_kincaid_grade_std']:.2f}")
        print(f"Synthetic data kincaid grade: {synthetic_syntax['flesch_kincaid_grade']:.2f} ± {synthetic_syntax['flesch_kincaid_grade_std']:.2f}")

        # Vocabulary overlap
        print(f"\nVOCABULARY OVERLAP:")
        vocab_overlap = self.analyze_vocabulary_overlap()
        print(f"Jaccard similarity: {vocab_overlap['jaccard_similarity']:.4f}")
        print(f"Shared vocabulary: {vocab_overlap['shared_vocab_size']} words")
        print(f"Overlap in real data: {vocab_overlap['overlap_in_real']:.4f}")
        print(f"Overlap in synthetic data: {vocab_overlap['overlap_in_synthetic']:.4f}")
        print(f"Unique to real: {vocab_overlap['unique_to_real']} words")
        print(f"Unique to synthetic: {vocab_overlap['unique_to_synthetic']} words")
        
        # Content diversity
        print(f"\nCONTENT DIVERSITY:")
        content_div = self.analyze_content_diversity()
        print(f"Real data internal similarity: {content_div['real_internal_similarity_mean']:.4f} ± {content_div['real_internal_similarity_std']:.4f}")
        print(f"Synthetic data internal similarity: {content_div['synthetic_internal_similarity_mean']:.4f} ± {content_div['synthetic_internal_similarity_std']:.4f}")
        print(f"Cross-group similarity: {content_div['cross_similarity_mean']:.4f} ± {content_div['cross_similarity_std']:.4f}")
        
        """
        # Statistical tests
        print(f"\nSTATISTICAL TESTS:")
        distributions = self.compare_distributions()
        print(f"Utterance length KS test p-value: {distributions['length_ks_pvalue']:.6f}")
        print(f"Utterance length Mann-Whitney U test p-value: {distributions['length_mannwhitney_pvalue']:.6f}")
        
        # Interpretation
        print(f"\nINTERPRETATION:")
        if vocab_overlap['jaccard_similarity'] > 0.7:
            print("✓ High vocabulary overlap - synthetic data covers similar vocabulary")
        elif vocab_overlap['jaccard_similarity'] > 0.5:
            print("~ Moderate vocabulary overlap - some vocabulary differences")
        else:
            print("✗ Low vocabulary overlap - significant vocabulary differences")
            
        if abs(real_syntax['mean_utterance_length'] - synthetic_syntax['mean_utterance_length']) < 2:
            print("✓ Similar utterance lengths between datasets")
        else:
            print("✗ Different utterance length patterns")
            
        if distributions['length_ks_pvalue'] > 0.05:
            print("✓ Similar length distributions (KS test p > 0.05)")
        else:
            print("✗ Different length distributions (KS test p ≤ 0.05)")
            
        return {
            'lexical_diversity': {'real': real_lexical, 'synthetic': synthetic_lexical},
            'syntactic_complexity': {'real': real_syntax, 'synthetic': synthetic_syntax},
            'vocabulary_overlap': vocab_overlap,
            'content_diversity': content_div,
            'statistical_tests': distributions
        }
        """
    
    def create_visualizations(self):
        """Create visualizations comparing the datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Utterance length distribution
        real_lengths = [len(word_tokenize(text)) for text in self.real_data]
        synthetic_lengths = [len(word_tokenize(text)) for text in self.synthetic_data]
        
        axes[0, 0].hist(real_lengths, bins=30, alpha=0.7, label='Real', density=True)
        axes[0, 0].hist(synthetic_lengths, bins=30, alpha=0.7, label='Synthetic', density=True)
        axes[0, 0].set_xlabel('Utterance Length (words)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Utterance Length Distribution')
        axes[0, 0].legend()
        
        # 2. Word length distribution
        real_word_lengths = []
        synthetic_word_lengths = []
        
        for text in self.real_data[:1000]:  # Sample to avoid memory issues
            words = word_tokenize(text)
            real_word_lengths.extend([len(w) for w in words if w.isalpha()])
            
        for text in self.synthetic_data[:1000]:
            words = word_tokenize(text)
            synthetic_word_lengths.extend([len(w) for w in words if w.isalpha()])
        
        axes[0, 1].hist(real_word_lengths, bins=15, alpha=0.7, label='Real', density=True)
        axes[0, 1].hist(synthetic_word_lengths, bins=15, alpha=0.7, label='Synthetic', density=True)
        axes[0, 1].set_xlabel('Word Length (characters)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Word Length Distribution')
        axes[0, 1].legend()
        
        # 3. Vocabulary overlap visualization
        vocab_overlap = self.analyze_vocabulary_overlap()
        categories = ['Real Only', 'Shared', 'Synthetic Only']
        values = [vocab_overlap['unique_to_real'], vocab_overlap['shared_vocab_size'], vocab_overlap['unique_to_synthetic']]
        
        axes[1, 0].pie(values, labels=categories, autopct='%1.1f%%')
        axes[1, 0].set_title('Vocabulary Overlap')
        
        # 4. Complexity comparison
        real_syntax = self.calculate_syntactic_complexity(self.real_data)
        synthetic_syntax = self.calculate_syntactic_complexity(self.synthetic_data)
        
        metrics = ['Mean Utterance\nLength', 'Flesch Reading\nEase', 'POS Tag\nDiversity']
        real_values = [real_syntax['mean_utterance_length'], 
                      real_syntax['flesch_reading_ease'], 
                      real_syntax['pos_tag_diversity']]
        synthetic_values = [synthetic_syntax['mean_utterance_length'], 
                           synthetic_syntax['flesch_reading_ease'], 
                           synthetic_syntax['pos_tag_diversity']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, real_values, width, label='Real', alpha=0.7)
        axes[1, 1].bar(x + width/2, synthetic_values, width, label='Synthetic', alpha=0.7)
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Values')
        axes[1, 1].set_title('Complexity Metrics Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics, rotation=45, ha='right')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('child_speech_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with your JSON file paths
    analyzer = ChildSpeechAnalyzer('combined_filtered.json', 'tedlium.json')
    
    # Generate comprehensive report
    results = analyzer.generate_report()
    
    # Create visualizations
    # analyzer.create_visualizations()
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print("="*60)
