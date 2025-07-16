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
from itertools import combinations

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

class ThreeWaySpeechAnalyzer:
    def __init__(self, real_child_path, synthetic_child_path, adult_path):
        """
        Initialize analyzer with paths to JSON files containing utterances
        Expected format: [{"text": "utterance1"}, {"text": "utterance2"}, ...]
        """
        self.real_child_data = self.load_data(real_child_path)
        self.synthetic_child_data = self.load_data(synthetic_child_path)
        self.adult_data = self.load_data(adult_path)
        self.stop_words = set(stopwords.words('english'))
        
        # Labels for datasets
        self.dataset_labels = ['Real Child', 'Synthetic Child', 'Adult']
        self.datasets = [self.real_child_data, self.synthetic_child_data, self.adult_data]

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
            all_sentences.extend(sent_tokenize(text))

        # Type-Token Ratio (TTR)
        unique_words = set(all_words)
        ttr = len(unique_words) / len(all_words) if all_words else 0

        # Moving Average Type-Token Ratio (MATTR) - simplified version
        # Calculate TTR for chunks of 100 words
        chunk_size = 100
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
        sentence_lengths = []
        word_lengths = []
        pos_tags_all = []

        for text in texts:
            # Utterance-level metrics
            words = word_tokenize(text)
            words_alpha = [w for w in words if w.isalpha()]
            utterance_lengths.append(len(words_alpha))

            # Word length
            word_lengths.extend([len(w) for w in words_alpha])

            # Sentence-level metrics
            sentences = sent_tokenize(text)
            for sent in sentences:
                sent_words = word_tokenize(sent)
                sent_words_alpha = [w for w in sent_words if w.isalpha()]
                sentence_lengths.append(len(sent_words_alpha))

            # POS tags
            pos_tags = pos_tag(words_alpha)
            pos_tags_all.extend([tag for word, tag in pos_tags])

        # Calculate readability scores
        all_text = ' '.join(texts)
        try:
            flesch_score = textstat.flesch_reading_ease(all_text)
            flesch_kincaid = textstat.flesch_kincaid_grade(all_text)
        except:
            flesch_score = flesch_kincaid = 0

        return {
            'mean_utterance_length': np.mean(utterance_lengths) if utterance_lengths else 0,
            'std_utterance_length': np.std(utterance_lengths) if utterance_lengths else 0,
            'mean_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'std_sentence_length': np.std(sentence_lengths) if sentence_lengths else 0,
            'mean_word_length': np.mean(word_lengths) if word_lengths else 0,
            'flesch_reading_ease': flesch_score,
            'flesch_kincaid_grade': flesch_kincaid,
            'pos_tag_diversity': len(set(pos_tags_all)),
            'total_pos_tags': len(pos_tags_all)
        }

    def analyze_pairwise_vocabulary_overlap(self, data1, data2):
        """Analyze vocabulary overlap between two datasets"""
        # Get vocabulary from both datasets
        words1 = set()
        words2 = set()

        for text in data1:
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            words1.update(words)

        for text in data2:
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            words2.update(words)

        # Calculate overlap metrics
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        jaccard_similarity = len(intersection) / len(union) if union else 0
        overlap_data1 = len(intersection) / len(words1) if words1 else 0
        overlap_data2 = len(intersection) / len(words2) if words2 else 0

        return {
            'data1_vocab_size': len(words1),
            'data2_vocab_size': len(words2),
            'shared_vocab_size': len(intersection),
            'jaccard_similarity': jaccard_similarity,
            'overlap_in_data1': overlap_data1,
            'overlap_in_data2': overlap_data2,
            'unique_to_data1': len(words1 - words2),
            'unique_to_data2': len(words2 - words1)
        }

    def analyze_all_vocabulary_overlaps(self):
        """Analyze vocabulary overlaps between all pairs of datasets"""
        overlaps = {}
        pairs = [
            ('Real Child', 'Synthetic Child', self.real_child_data, self.synthetic_child_data),
            ('Real Child', 'Adult', self.real_child_data, self.adult_data),
            ('Synthetic Child', 'Adult', self.synthetic_child_data, self.adult_data)
        ]
        
        for label1, label2, data1, data2 in pairs:
            overlap = self.analyze_pairwise_vocabulary_overlap(data1, data2)
            overlaps[f"{label1} vs {label2}"] = overlap
            
        return overlaps

    def analyze_three_way_content_diversity(self):
        """Analyze content diversity across all three datasets using TF-IDF"""
        # Combine all data for TF-IDF analysis
        all_texts = self.real_child_data + self.synthetic_child_data + self.adult_data
        labels = (['real_child'] * len(self.real_child_data) + 
                 ['synthetic_child'] * len(self.synthetic_child_data) + 
                 ['adult'] * len(self.adult_data))

        # TF-IDF analysis
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Split into three groups
        real_child_end = len(self.real_child_data)
        synthetic_child_end = real_child_end + len(self.synthetic_child_data)
        
        real_child_tfidf = tfidf_matrix[:real_child_end]
        synthetic_child_tfidf = tfidf_matrix[real_child_end:synthetic_child_end]
        adult_tfidf = tfidf_matrix[synthetic_child_end:]

        # Sample to avoid memory issues
        sample_size = min(100, min(len(self.real_child_data), len(self.synthetic_child_data), len(self.adult_data)))
        
        datasets_tfidf = [
            real_child_tfidf[:sample_size],
            synthetic_child_tfidf[:sample_size], 
            adult_tfidf[:sample_size]
        ]

        # Calculate internal similarities for each group
        internal_similarities = {}
        for i, (label, tfidf_data) in enumerate(zip(self.dataset_labels, datasets_tfidf)):
            if tfidf_data.shape[0] > 1:
                sim_matrix = cosine_similarity(tfidf_data)
                similarities = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                internal_similarities[label] = {
                    'mean': np.mean(similarities),
                    'std': np.std(similarities)
                }
            else:
                internal_similarities[label] = {'mean': 0, 'std': 0}

        # Calculate cross-group similarities
        cross_similarities = {}
        pairs = list(combinations(range(3), 2))
        pair_labels = [
            'Real Child vs Synthetic Child',
            'Real Child vs Adult', 
            'Synthetic Child vs Adult'
        ]
        
        for (i, j), pair_label in zip(pairs, pair_labels):
            if datasets_tfidf[i].shape[0] > 0 and datasets_tfidf[j].shape[0] > 0:
                cross_sim_matrix = cosine_similarity(datasets_tfidf[i], datasets_tfidf[j])
                cross_similarities[pair_label] = {
                    'mean': np.mean(cross_sim_matrix),
                    'std': np.std(cross_sim_matrix)
                }
            else:
                cross_similarities[pair_label] = {'mean': 0, 'std': 0}

        return {
            'internal_similarities': internal_similarities,
            'cross_similarities': cross_similarities
        }

    def compare_all_distributions(self):
        """Compare statistical distributions across all three datasets"""
        # Get utterance lengths for all datasets
        lengths = []
        for dataset in self.datasets:
            dataset_lengths = [len(word_tokenize(text)) for text in dataset]
            lengths.append(dataset_lengths)

        # Statistical tests between all pairs
        statistical_tests = {}
        pairs = list(combinations(range(3), 2))
        pair_labels = [
            'Real Child vs Synthetic Child',
            'Real Child vs Adult',
            'Synthetic Child vs Adult'
        ]

        for (i, j), pair_label in zip(pairs, pair_labels):
            ks_stat, ks_p = stats.ks_2samp(lengths[i], lengths[j])
            mw_stat, mw_p = stats.mannwhitneyu(lengths[i], lengths[j], alternative='two-sided')
            
            statistical_tests[pair_label] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_p,
                'mannwhitney_statistic': mw_stat,
                'mannwhitney_pvalue': mw_p,
                'mean_diff': np.mean(lengths[i]) - np.mean(lengths[j])
            }

        # Basic statistics for each dataset
        dataset_stats = {}
        for i, label in enumerate(self.dataset_labels):
            dataset_stats[label] = {
                'length_mean': np.mean(lengths[i]),
                'length_std': np.std(lengths[i]),
                'length_median': np.median(lengths[i]),
                'utterance_count': len(lengths[i])
            }

        return {
            'dataset_statistics': dataset_stats,
            'pairwise_tests': statistical_tests
        }

    def generate_comprehensive_report(self):
        """Generate comprehensive three-way comparison report"""
        print("=" * 80)
        print("THREE-WAY SPEECH DATA COMPARISON REPORT")
        print("Real Child vs Synthetic Child vs Adult Speech")
        print("=" * 80)

        # Basic statistics
        print(f"\nBASIC STATISTICS:")
        for i, label in enumerate(self.dataset_labels):
            print(f"{label} utterances: {len(self.datasets[i])}")

        # Lexical diversity for all datasets
        print(f"\nLEXICAL DIVERSITY:")
        lexical_results = {}
        for i, label in enumerate(self.dataset_labels):
            lexical_results[label] = self.calculate_lexical_diversity(self.datasets[i])
            print(f"\n{label}:")
            print(f"  TTR: {lexical_results[label]['ttr']:.4f}")
            print(f"  MATTR: {lexical_results[label]['mattr']:.4f}")
            print(f"  Vocabulary size: {lexical_results[label]['vocab_size']}")
            print(f"  Total words: {lexical_results[label]['total_words']}")

        # Syntactic complexity for all datasets
        print(f"\nSYNTACTIC COMPLEXITY:")
        syntax_results = {}
        for i, label in enumerate(self.dataset_labels):
            syntax_results[label] = self.calculate_syntactic_complexity(self.datasets[i])
            print(f"\n{label}:")
            print(f"  Mean utterance length: {syntax_results[label]['mean_utterance_length']:.2f} ± {syntax_results[label]['std_utterance_length']:.2f}")
            print(f"  Mean word length: {syntax_results[label]['mean_word_length']:.2f}")
            print(f"  Flesch Reading Ease: {syntax_results[label]['flesch_reading_ease']:.2f}")
            print(f"  POS tag diversity: {syntax_results[label]['pos_tag_diversity']}")

        # Vocabulary overlaps
        print(f"\nVOCABULARY OVERLAPS:")
        vocab_overlaps = self.analyze_all_vocabulary_overlaps()
        for pair_name, overlap in vocab_overlaps.items():
            print(f"\n{pair_name}:")
            print(f"  Jaccard similarity: {overlap['jaccard_similarity']:.4f}")
            print(f"  Shared vocabulary: {overlap['shared_vocab_size']} words")

        # Content diversity
        print(f"\nCONTENT DIVERSITY:")
        content_div = self.analyze_three_way_content_diversity()
        
        print("\nInternal similarities (within dataset):")
        for dataset, similarity in content_div['internal_similarities'].items():
            print(f"  {dataset}: {similarity['mean']:.4f} ± {similarity['std']:.4f}")
            
        print("\nCross-dataset similarities:")
        for pair, similarity in content_div['cross_similarities'].items():
            print(f"  {pair}: {similarity['mean']:.4f} ± {similarity['std']:.4f}")

        # Statistical tests
        print(f"\nSTATISTICAL TESTS:")
        distributions = self.compare_all_distributions()
        
        print("\nDataset statistics:")
        for dataset, stats_data in distributions['dataset_statistics'].items():
            print(f"  {dataset}: Mean length = {stats_data['length_mean']:.2f}, "
                  f"Std = {stats_data['length_std']:.2f}, "
                  f"Median = {stats_data['length_median']:.2f}")
        
        print("\nPairwise statistical tests:")
        for pair, test_results in distributions['pairwise_tests'].items():
            print(f"  {pair}:")
            print(f"    KS test p-value: {test_results['ks_pvalue']:.6f}")
            print(f"    Mann-Whitney U test p-value: {test_results['mannwhitney_pvalue']:.6f}")
            print(f"    Mean difference: {test_results['mean_diff']:.2f}")

        # Enhanced interpretation
        print(f"\nINTERPRETATION:")
        
        # Vocabulary overlap interpretation
        rc_sc_overlap = vocab_overlaps['Real Child vs Synthetic Child']['jaccard_similarity']
        rc_adult_overlap = vocab_overlaps['Real Child vs Adult']['jaccard_similarity']
        sc_adult_overlap = vocab_overlaps['Synthetic Child vs Adult']['jaccard_similarity']
        
        print("\nVocabulary Analysis:")
        if rc_sc_overlap > 0.7:
            print("✓ High vocabulary overlap between real and synthetic child speech")
        elif rc_sc_overlap > 0.5:
            print("~ Moderate vocabulary overlap between real and synthetic child speech")
        else:
            print("✗ Low vocabulary overlap between real and synthetic child speech")
            
        if rc_adult_overlap < sc_adult_overlap:
            print("✓ Synthetic child speech is more similar to adult speech than real child speech")
        else:
            print("✓ Real child speech patterns maintained in synthetic data")

        # Complexity analysis
        real_child_complexity = syntax_results['Real Child']['mean_utterance_length']
        synthetic_child_complexity = syntax_results['Synthetic Child']['mean_utterance_length']
        adult_complexity = syntax_results['Adult']['mean_utterance_length']
        
        print("\nComplexity Analysis:")
        if abs(real_child_complexity - synthetic_child_complexity) < 2:
            print("✓ Similar utterance complexity between real and synthetic child speech")
        else:
            print("✗ Different complexity patterns between real and synthetic child speech")
            
        if adult_complexity > max(real_child_complexity, synthetic_child_complexity):
            print("✓ Adult speech shows expected higher complexity than child speech")
        else:
            print("~ Unexpected complexity patterns between adult and child speech")

        # Statistical significance
        print("\nStatistical Significance:")
        for pair, test_results in distributions['pairwise_tests'].items():
            if test_results['ks_pvalue'] > 0.05:
                print(f"✓ {pair}: Similar distributions (p > 0.05)")
            else:
                print(f"✗ {pair}: Significantly different distributions (p ≤ 0.05)")

        return {
            'lexical_diversity': lexical_results,
            'syntactic_complexity': syntax_results,
            'vocabulary_overlaps': vocab_overlaps,
            'content_diversity': content_div,
            'statistical_tests': distributions
        }

    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for three-way comparison"""
        fig, axes = plt.subplots(3, 2, figsize=(18, 20))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        labels = self.dataset_labels

        # 1. Utterance length distribution
        all_lengths = []
        for i, dataset in enumerate(self.datasets):
            lengths = [len(word_tokenize(text)) for text in dataset]
            all_lengths.append(lengths)
            axes[0, 0].hist(lengths, bins=30, alpha=0.6, label=labels[i], 
                           density=True, color=colors[i])
        
        axes[0, 0].set_xlabel('Utterance Length (words)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Utterance Length Distribution Comparison')
        axes[0, 0].legend()

        # 2. Box plot for utterance lengths
        axes[0, 1].boxplot(all_lengths, labels=labels)
        axes[0, 1].set_ylabel('Utterance Length (words)')
        axes[0, 1].set_title('Utterance Length Box Plot')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Vocabulary size comparison
        lexical_results = {}
        vocab_sizes = []
        ttrs = []
        mattrs = []
        
        for i, dataset in enumerate(self.datasets):
            lexical_results[labels[i]] = self.calculate_lexical_diversity(dataset)
            vocab_sizes.append(lexical_results[labels[i]]['vocab_size'])
            ttrs.append(lexical_results[labels[i]]['ttr'])
            mattrs.append(lexical_results[labels[i]]['mattr'])

        x = np.arange(len(labels))
        width = 0.25

        axes[1, 0].bar(x - width, vocab_sizes, width, label='Vocab Size', alpha=0.8)
        axes[1, 0].bar(x, [t*1000 for t in ttrs], width, label='TTR×1000', alpha=0.8)
        axes[1, 0].bar(x + width, [m*1000 for m in mattrs], width, label='MATTR×1000', alpha=0.8)
        
        axes[1, 0].set_xlabel('Dataset')
        axes[1, 0].set_ylabel('Values')
        axes[1, 0].set_title('Lexical Diversity Metrics')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(labels)
        axes[1, 0].legend()

        # 4. Vocabulary overlap heatmap
        vocab_overlaps = self.analyze_all_vocabulary_overlaps()
        overlap_matrix = np.zeros((3, 3))
        
        # Fill diagonal with 1.0 (perfect self-overlap)
        np.fill_diagonal(overlap_matrix, 1.0)
        
        # Fill off-diagonal elements
        pair_mapping = {
            'Real Child vs Synthetic Child': (0, 1),
            'Real Child vs Adult': (0, 2),
            'Synthetic Child vs Adult': (1, 2)
        }
        
        for pair_name, (i, j) in pair_mapping.items():
            jaccard = vocab_overlaps[pair_name]['jaccard_similarity']
            overlap_matrix[i, j] = jaccard
            overlap_matrix[j, i] = jaccard

        im = axes[1, 1].imshow(overlap_matrix, cmap='Blues', aspect='equal')
        axes[1, 1].set_xticks(range(3))
        axes[1, 1].set_yticks(range(3))
        axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 1].set_yticklabels(labels)
        axes[1, 1].set_title('Vocabulary Overlap (Jaccard Similarity)')
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = axes[1, 1].text(j, i, f'{overlap_matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=axes[1, 1])

        # 5. Complexity metrics comparison
        syntax_results = {}
        mean_utterance_lengths = []
        flesch_scores = []
        pos_diversities = []
        
        for i, dataset in enumerate(self.datasets):
            syntax_results[labels[i]] = self.calculate_syntactic_complexity(dataset)
            mean_utterance_lengths.append(syntax_results[labels[i]]['mean_utterance_length'])
            flesch_scores.append(syntax_results[labels[i]]['flesch_reading_ease'])
            pos_diversities.append(syntax_results[labels[i]]['pos_tag_diversity'])

        x = np.arange(len(labels))
        width = 0.25

        axes[2, 0].bar(x - width, mean_utterance_lengths, width, 
                      label='Mean Utterance Length', alpha=0.8)
        axes[2, 0].bar(x, [f/10 for f in flesch_scores], width, 
                      label='Flesch Score/10', alpha=0.8)
        axes[2, 0].bar(x + width, pos_diversities, width, 
                      label='POS Tag Diversity', alpha=0.8)
        
        axes[2, 0].set_xlabel('Dataset')
        axes[2, 0].set_ylabel('Values')
        axes[2, 0].set_title('Syntactic Complexity Metrics')
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels(labels)
        axes[2, 0].legend()

        # 6. Content diversity (internal vs cross similarities)
        content_div = self.analyze_three_way_content_diversity()
        
        # Internal similarities
        internal_means = [content_div['internal_similarities'][label]['mean'] 
                         for label in labels]
        
        # Cross similarities 
        cross_means = []
        cross_labels = []
        for pair_name, similarity in content_div['cross_similarities'].items():
            cross_means.append(similarity['mean'])
            cross_labels.append(pair_name.replace(' vs ', '\nvs '))

        # Create grouped bar chart
        all_means = internal_means + cross_means
        all_labels = [f"{label}\n(Internal)" for label in labels] + cross_labels
        bar_colors = colors + ['#d62728', '#9467bd', '#8c564b']  # Different colors for cross-comparisons
        
        bars = axes[2, 1].bar(range(len(all_means)), all_means, color=bar_colors, alpha=0.8)
        axes[2, 1].set_xlabel('Comparison Type')
        axes[2, 1].set_ylabel('Mean Cosine Similarity')
        axes[2, 1].set_title('Content Diversity: Internal vs Cross-Dataset Similarities')
        axes[2, 1].set_xticks(range(len(all_means)))
        axes[2, 1].set_xticklabels(all_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, all_means):
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('three_way_speech_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with your three JSON file paths
    analyzer = ThreeWaySpeechAnalyzer('myst.json', 'tinyd_age51015.json', 'tedlium.json')

    # Generate comprehensive report
    results = analyzer.generate_comprehensive_report()

    # Create visualizations
    analyzer.create_comprehensive_visualizations()

    print(f"\n{'='*80}")
    print("Three-way analysis complete! Visualizations saved as 'three_way_speech_comparison.png'")
    print("="*80)
