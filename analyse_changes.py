#!/usr/bin/env python3
"""
ASR Language Model Impact Analysis
Compares ASR outputs before and after LM incorporation to analyze WER changes
"""

import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict
import difflib

@dataclass
class Utterance:
    id: str
    predicted: str
    actual: str
    wer: float

def calculate_wer(predicted: str, actual: str) -> float:
    """Calculate Word Error Rate between predicted and actual text"""
    pred_words = predicted.lower().split()
    actual_words = actual.lower().split()
    
    if len(actual_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    # Use edit distance (Levenshtein) for WER calculation
    d = [[0] * (len(pred_words) + 1) for _ in range(len(actual_words) + 1)]
    
    for i in range(len(actual_words) + 1):
        d[i][0] = i
    for j in range(len(pred_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(actual_words) + 1):
        for j in range(1, len(pred_words) + 1):
            if actual_words[i-1] == pred_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j] + 1,      # deletion
                             d[i][j-1] + 1,      # insertion
                             d[i-1][j-1] + 1)    # substitution
    
    return d[len(actual_words)][len(pred_words)] / len(actual_words)

def parse_file(filepath: str) -> Dict[str, Utterance]:
    """Parse the ASR output file and return dictionary of utterances"""
    utterances = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('<DIV>')
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            
            utt_id = parts[0].strip()
            predicted = parts[1].strip()
            actual = parts[2].strip()
            
            wer = calculate_wer(predicted, actual)
            utterances[utt_id] = Utterance(utt_id, predicted, actual, wer)
    
    return utterances

def get_word_changes(before_pred: str, after_pred: str) -> Tuple[List[str], List[str]]:
    """Get words that were changed between before and after predictions"""
    before_words = before_pred.lower().split()
    after_words = after_pred.lower().split()
    
    # Use difflib to find differences
    matcher = difflib.SequenceMatcher(None, before_words, after_words)
    
    removed_words = []
    added_words = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete':
            removed_words.extend(before_words[i1:i2])
        elif tag == 'insert':
            added_words.extend(after_words[j1:j2])
        elif tag == 'replace':
            removed_words.extend(before_words[i1:i2])
            added_words.extend(after_words[j1:j2])
    
    return removed_words, added_words

def analyze_changes(before_file: str, after_file: str):
    """Main analysis function"""
    print("Parsing files...")
    before_utts = parse_file(before_file)
    after_utts = parse_file(after_file)
    
    # Find common utterances
    common_ids = set(before_utts.keys()) & set(after_utts.keys())
    print(f"Found {len(common_ids)} common utterances")
    
    if len(common_ids) == 0:
        print("No common utterances found. Check file formats and utterance IDs.")
        return
    
    # Analyze changes
    improvements = []  # (utt_id, wer_before, wer_after, wer_change)
    degradations = []
    no_change = []
    
    word_changes_improved = defaultdict(int)  # (removed_word, added_word) -> count
    word_changes_degraded = defaultdict(int)
    
    for utt_id in common_ids:
        before = before_utts[utt_id]
        after = after_utts[utt_id]
        
        wer_change = after.wer - before.wer
        
        if abs(wer_change) < 1e-6:  # No change
            no_change.append((utt_id, before.wer, after.wer, wer_change))
        elif wer_change < 0:  # Improvement
            improvements.append((utt_id, before.wer, after.wer, wer_change))
            
            # Analyze word changes for improvements
            removed, added = get_word_changes(before.predicted, after.predicted)
            for r in removed:
                for a in added:
                    word_changes_improved[(r, a)] += 1
            # Also track single word removals/additions
            for r in removed:
                if not added:  # Only removals
                    word_changes_improved[(r, "")] += 1
            for a in added:
                if not removed:  # Only additions
                    word_changes_improved[("", a)] += 1
                    
        else:  # Degradation
            degradations.append((utt_id, before.wer, after.wer, wer_change))
            
            # Analyze word changes for degradations
            removed, added = get_word_changes(before.predicted, after.predicted)
            for r in removed:
                for a in added:
                    word_changes_degraded[(r, a)] += 1
            # Also track single word removals/additions
            for r in removed:
                if not added:  # Only removals
                    word_changes_degraded[(r, "")] += 1
            for a in added:
                if not removed:  # Only additions
                    word_changes_degraded[("", a)] += 1
    
    # Sort by WER change magnitude
    improvements.sort(key=lambda x: x[3])  # Most improvement first (most negative)
    degradations.sort(key=lambda x: x[3], reverse=True)  # Most degradation first (most positive)
    
    # Calculate overall statistics
    total_before_wer = sum(before_utts[uid].wer for uid in common_ids) / len(common_ids)
    total_after_wer = sum(after_utts[uid].wer for uid in common_ids) / len(common_ids)
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total utterances analyzed: {len(common_ids)}")
    print(f"Average WER before LM: {total_before_wer:.4f}")
    print(f"Average WER after LM: {total_after_wer:.4f}")
    print(f"Overall WER change: {total_after_wer - total_before_wer:.4f}")
    print(f"Utterances improved: {len(improvements)} ({len(improvements)/len(common_ids)*100:.1f}%)")
    print(f"Utterances degraded: {len(degradations)} ({len(degradations)/len(common_ids)*100:.1f}%)")
    print(f"Utterances unchanged: {len(no_change)} ({len(no_change)/len(common_ids)*100:.1f}%)")
    
    # Show most common word changes that led to improvements
    print("\n" + "="*80)
    print("MOST COMMON CHANGES LEADING TO WER IMPROVEMENTS")
    print("="*80)
    top_improved_changes = sorted(word_changes_improved.items(), key=lambda x: x[1], reverse=True)[:20]
    if top_improved_changes:
        for (removed, added), count in top_improved_changes:
            if removed == "":
                print(f"Added '{added}': {count} occurrences")
            elif added == "":
                print(f"Removed '{removed}': {count} occurrences")
            else:
                print(f"'{removed}' → '{added}': {count} occurrences")
    else:
        print("No word changes found that led to improvements.")
    
    # Show most common word changes that led to degradations
    print("\n" + "="*80)
    print("MOST COMMON CHANGES LEADING TO WER DEGRADATIONS")
    print("="*80)
    top_degraded_changes = sorted(word_changes_degraded.items(), key=lambda x: x[1], reverse=True)[:20]
    if top_degraded_changes:
        for (removed, added), count in top_degraded_changes:
            if removed == "":
                print(f"Added '{added}': {count} occurrences")
            elif added == "":
                print(f"Removed '{removed}': {count} occurrences")
            else:
                print(f"'{removed}' → '{added}': {count} occurrences")
    else:
        print("No word changes found that led to degradations.")

    # Show examples of top patterns
    if len(improvements) > 0:
        print("\n" + "="*80)
        print("EXAMPLE IMPROVEMENTS (Few Examples)")
        print("="*80)
        for i, (utt_id, wer_before, wer_after, change) in enumerate(improvements[:20]):
            print(f"\n{i+1}. Utterance ID: {utt_id}")
            print(f"   WER: {wer_before:.4f} → {wer_after:.4f} (Δ {change:.4f})")
            print(f"   Before: '{before_utts[utt_id].predicted}'")
            print(f"   After:  '{after_utts[utt_id].predicted}'")
            print(f"   Actual: '{before_utts[utt_id].actual}'")
    
    if len(degradations) > 0:
        print("\n" + "="*80)
        print("EXAMPLE DEGRADATIONS (Few Examples)")
        print("="*80)
        for i, (utt_id, wer_before, wer_after, change) in enumerate(degradations[:20]):
            print(f"\n{i+1}. Utterance ID: {utt_id}")
            print(f"   WER: {wer_before:.4f} → {wer_after:.4f} (Δ +{change:.4f})")
            print(f"   Before: '{before_utts[utt_id].predicted}'")
            print(f"   After:  '{after_utts[utt_id].predicted}'")
            print(f"   Actual: '{before_utts[utt_id].actual}'")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python asr_analysis.py <before_lm_file.txt> <after_lm_file.txt>")
        sys.exit(1)
    
    before_file = sys.argv[1]
    after_file = sys.argv[2]
    
    try:
        analyze_changes(before_file, after_file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")
