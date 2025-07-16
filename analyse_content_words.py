import pandas as pd
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class CorpusContentWordsComparator:
    def __init__(self):
        self.corpus_words = set()
        self.content_words_data = []
        self.missing_words = []
        self.present_words = []
        
    def load_corpus(self, corpus_file_path):
        """Load and tokenize the corpus file."""
        print(f"Loading corpus from: {corpus_file_path}")
        
        try:
            with open(corpus_file_path, 'r', encoding='utf-8') as file:
                corpus_text = file.read()
            
            # Tokenize the corpus text
            print("Tokenizing corpus...")
            tokens = word_tokenize(corpus_text.lower())
            
            # Filter out non-alphabetic tokens and convert to set for faster lookup
            self.corpus_words = set(token for token in tokens if token.isalpha())
            
            print(f"Corpus loaded successfully!")
            print(f"Total unique words in corpus: {len(self.corpus_words):,}")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Could not find corpus file '{corpus_file_path}'")
            return False
        except Exception as e:
            print(f"Error loading corpus: {e}")
            return False
    
    def load_content_words_csv(self, csv_file_path):
        """Load the content words CSV file."""
        print(f"Loading content words CSV from: {csv_file_path}")
        
        try:
            df = pd.read_csv(csv_file_path)
            
            # Check if required columns exist
            required_columns = ['word', 'total_error_percentage']
            if not all(col in df.columns for col in required_columns):
                print(f"Error: CSV file must contain columns: {required_columns}")
                print(f"Found columns: {list(df.columns)}")
                return False
            
            # Convert to list of dictionaries
            self.content_words_data = df.to_dict('records')
            
            print(f"Content words CSV loaded successfully!")
            print(f"Total content words: {len(self.content_words_data):,}")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Could not find CSV file '{csv_file_path}'")
            return False
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def compare_words(self):
        """Compare content words with corpus words."""
        print("Comparing content words with corpus...")
        
        self.missing_words = []
        self.present_words = []
        
        for word_data in self.content_words_data:
            word = word_data['word'].lower()
            error_percentage = word_data['total_error_percentage']
            
            if word not in self.corpus_words:
                self.missing_words.append({
                    'word': word_data['word'],  # Keep original case
                    'total_error_percentage': error_percentage
                })
            else:
                self.present_words.append({
                    'word': word_data['word'],  # Keep original case
                    'total_error_percentage': error_percentage
                })
        
        # Sort by error percentage (descending)
        self.missing_words.sort(key=lambda x: x['total_error_percentage'], reverse=True)
        self.present_words.sort(key=lambda x: x['total_error_percentage'], reverse=True)
        
        print(f"Comparison completed!")
        print(f"Words NOT found in corpus: {len(self.missing_words):,}")
        print(f"Words found in corpus: {len(self.present_words):,}")
    
    def generate_report(self):
        """Generate a comprehensive comparison report."""
        print("\n" + "=" * 80)
        print("CORPUS vs CONTENT WORDS COMPARISON REPORT")
        print("=" * 80)
        
        total_content_words = len(self.content_words_data)
        missing_count = len(self.missing_words)
        present_count = len(self.present_words)
        
        print(f"\nSUMMARY:")
        print(f"Total content words analyzed: {total_content_words:,}")
        print(f"Words found in corpus: {present_count:,} ({present_count/total_content_words*100:.1f}%)")
        print(f"Words NOT found in corpus: {missing_count:,} ({missing_count/total_content_words*100:.1f}%)")
        print(f"Total unique words in corpus: {len(self.corpus_words):,}")
        
        # Show words NOT found in corpus
        if self.missing_words:
            print(f"\nWORDS NOT FOUND IN CORPUS (sorted by error percentage):")
            print("-" * 50)
            print(f"{'Word':<20} {'Error %':<10}")
            print("-" * 50)
            
            for word_data in self.missing_words:
                print(f"{word_data['word']:<20} {word_data['total_error_percentage']:<9.2f}%")
        
        # Show top words found in corpus (for comparison)
        if self.present_words:
            print(f"\nTOP 20 WORDS FOUND IN CORPUS (sorted by error percentage):")
            print("-" * 50)
            print(f"{'Word':<20} {'Error %':<10}")
            print("-" * 50)
            
            for word_data in self.present_words[:20]:
                print(f"{word_data['word']:<20} {word_data['total_error_percentage']:<9.2f}%")
        
        # Calculate total error percentage for missing words
        if self.missing_words:
            total_missing_error_pct = sum(word['total_error_percentage'] for word in self.missing_words)
            print(f"\nADDITIONAL STATISTICS:")
            print(f"Total error percentage from missing words: {total_missing_error_pct:.2f}%")
            print(f"Average error percentage per missing word: {total_missing_error_pct/len(self.missing_words):.2f}%")
            
            # Show highest error words not in corpus
            if len(self.missing_words) > 0:
                highest_error_missing = self.missing_words[0]
                print(f"Highest error word not in corpus: '{highest_error_missing['word']}' ({highest_error_missing['total_error_percentage']:.2f}%)")
    
    def export_missing_words_csv(self, output_file='missing_content_words.csv'):
        """Export missing words to CSV file."""
        if self.missing_words:
            df = pd.DataFrame(self.missing_words)
            df.to_csv(output_file, index=False)
            print(f"\nMissing words exported to: {output_file}")
        else:
            print("\nNo missing words to export.")
    
    def run_comparison(self, corpus_file_path, content_words_csv_path):
        """Run the complete comparison process."""
        print("Starting corpus vs content words comparison...")
        
        # Load corpus
        if not self.load_corpus(corpus_file_path):
            return False
        
        # Load content words CSV
        if not self.load_content_words_csv(content_words_csv_path):
            return False
        
        # Compare words
        self.compare_words()
        
        # Generate report
        self.generate_report()
        
        # Export missing words
        self.export_missing_words_csv()
        
        return True

def main():
    """Main function to run the comparison."""
    # File paths - update these to match your files
    corpus_file_path = 'corpus_clean_comb5.txt'
    content_words_csv_path = 'content_words.csv'
    
    # Create comparator instance
    comparator = CorpusContentWordsComparator()
    
    # Run comparison
    success = comparator.run_comparison(corpus_file_path, content_words_csv_path)
    
    if success:
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("COMPARISON FAILED - Check error messages above")
        print("=" * 80)

if __name__ == "__main__":
    main()
