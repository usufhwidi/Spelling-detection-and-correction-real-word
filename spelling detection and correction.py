import collections
import itertools
import math
import re
import enchant
import nltk.data
from joblib.numpy_pickle_utils import xrange


class Corrector:
    def init(self, training_file):
        self.UnigramCounts = collections.defaultdict(lambda: 0)
        self.BigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.sentences = []
        self.dic = enchant.Dict("en_US")
        self.normalize_file(training_file)
        self.train()

    def normalize_file(self, file):
        f = open(file)
        content = f.read()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for sentence in tokenizer.tokenize(content):
            clean = [i.lower() for i in re.split('[^a-zA-Z]+', sentence) if i]
            self.sentences.append(clean)

    def train(self):
        for sentence in self.sentences:
            sentence.insert(0, '<s>')
            sentence.append('</s>')
            for i in xrange(len(sentence) - 1):
                word1 = sentence[i]
                word2 = sentence[i + 1]
                self.UnigramCounts[word1] += 1
                self.BigramCounts[(word1, word2)] += 1
                self.total += 1
            self.total += 1
            self.UnigramCounts[sentence[-1]] += 1

    def candidate_word(self, word):

        suggests = self.dic.suggest(word)
        suggests = [suggest.lower() for suggest in suggests][:4]
        suggests.append(word)
        suggests = list(set(suggests))

        return suggests, len(suggests)

    def candidate_sentence(self, sentence):
        candidate_sentences = []
        words_count = {}
        for word in sentence:
            candidate_sentences.append(self.candidate_word(word)[0])
            words_count[word] = self.candidate_word(word)[1]
        candidate_sentences = list(itertools.product(*candidate_sentences))
        return candidate_sentences, words_count

    def correction_score(self, words_count, old_sentence, new_sentence):
        score = 1
        for i in xrange(len(new_sentence)):
            if new_sentence[i] in words_count:
                score *= 0.95
            else:
                score *= (0.05 / (words_count[old_sentence[i]] - 1))
        return math.log(score)

    def score(self, sentence):
        score = 0.0
        for i in xrange(len(sentence) - 1):
            if self.BigramCounts[(sentence[i], sentence[i + 1])] > 0:
                score += math.log(self.BigramCounts[(sentence[i], sentence[i + 1])])
                score -= math.log(self.UnigramCounts[sentence[i]])
            else:
                score += (math.log(self.UnigramCounts[sentence[i + 1]] + 1) + math.log(0.4))
                score -= math.log(self.total + len(self.UnigramCounts))
        return score

    def return_best_sentence(self, old_sentence):
        bestScore = float('-inf')
        bestSentence = []
        old_sentence = [word.lower() for word in old_sentence.split()]
        sentences, word_count = self.candidate_sentence(old_sentence)
        for new_sentence in sentences:
            new_sentence = list(new_sentence)

            score = self.correction_score(word_count, new_sentence, old_sentence)
            new_sentence.insert(0, '<s>')
            new_sentence.append('</s>')
            score += self.score(new_sentence)
            if score >= bestScore:
                bestScore = score
                bestSentence = new_sentence
        bestSentence = ' '.join(bestSentence[1:-1])
        return bestSentence


corrector = Corrector('F:/big.txt')
v = corrector.return_best_sentence('uoy donky')
print(v)