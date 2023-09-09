import re

import nltk
from dostoevsky.models import FastTextSocialNetworkModel
from dostoevsky.tokenization import RegexTokenizer
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)
from nltk.corpus import stopwords
from spellchecker import SpellChecker

import src.framework.instance as instance
import os

PATH = os.path.dirname(__file__)


class Preprocessor:
    def __init__(self):
        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()
        self._emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._emb)
        self._syntax_parser = NewsSyntaxParser(self._emb)
        try:
            self._russian_stopwords = stopwords.words("russian")
        except LookupError:
            nltk.download("stopwords")
            self._russian_stopwords = stopwords.words("russian")

        self._spell = SpellChecker(language="ru")

    def _get_abbreviations(self, f):
        abbreviations = dict()
        with open(f) as file:
            for line in file.readlines():
                key, value = map(str.strip, line.split('->'))
                abbreviations[key] = value
        return abbreviations

    def _replace_abbreviations_str(self, sentence):
        abbreviations = self._get_abbreviations(f'{PATH}/data/сокращения.txt')
        words = sentence.lower().split()
        for i in range(len(words)):
            if words[i] in abbreviations:
                words[i] = abbreviations[words[i]]
        return ' '.join(words)

    def _replace_anglicisms_str(self, sentence):
        anglicisms = self._get_abbreviations(f'{PATH}/data/англицизмы.txt')
        words = sentence.lower().split()
        for i in range(len(words)):
            if words[i] in anglicisms:
                words[i] = anglicisms[words[i]]
        return ' '.join(words)

    @staticmethod
    def _get_keys(list_of_dicts):
        keys = []
        for d in list_of_dicts:
            for key in d:
                keys.append(key)
        return keys

    def replace_abbreviations(self, inst: instance.Instance):
        corrected_answers = list(
            map(lambda sentence: self._replace_abbreviations_str(sentence), inst.answers)
        )
        return instance.Instance(inst.question, inst.id_, corrected_answers, inst.counts, inst.sentiments)

    def token_lemmatization_spc_natasha(self, inst: instance.Instance):
        result = []
        for answer in inst.answers:
            tokens = re.sub("[^а-яА-Я]", " ", answer)

            doc = Doc(" ".join(tokens))
            doc.segment(self._segmenter)
            doc.tag_morph(self._morph_tagger)
            for token in doc.tokens:
                token.lemmatize(self._morph_vocab)
            doc = Doc(tokens)

            doc.segment(self._segmenter)
            doc.tag_morph(self._morph_tagger)

            for token in doc.tokens:
                token.lemmatize(self._morph_vocab)

            result.append(" ".join([_.lemma for _ in doc.tokens if _.lemma not in self._russian_stopwords]))
        return instance.Instance(inst.question, inst.id_, result, inst.counts, inst.sentiments)

    def token_lemmatization_natasha(self, inst: instance.Instance):
        result = []
        for answer in inst.answers:
            tokens = re.sub("[^а-яА-Яa-zA-Z]", " ", answer)
            first_lem = [list(self._spell.candidates(i))[0].lower()
                         if self._spell.candidates(i) is not None
                         else i.lower()
                         for i in tokens.split(" ")]
            doc = Doc(" ".join(first_lem))
            doc.segment(self._segmenter)
            doc.tag_morph(self._morph_tagger)
            for token in doc.tokens:
                token.lemmatize(self._morph_vocab)
            result.append(" ".join([_.lemma for _ in doc.tokens if _.lemma not in self._russian_stopwords]))

        return instance.Instance(inst.question, inst.id_, result, inst.counts, inst.sentiments)

    def token_lemmatization_spc(self, inst: instance.Instance):
        result = []
        for answer in inst.answers:
            tokens = re.sub("[^а-яА-Яa-zA-Z]", " ", answer)
            first_lem = [list(self._spell.candidates(i))[0].lower()
                         if self._spell.candidates(i) is not None
                         else i.lower()
                         for i in tokens.split(" ")]
            result.append(" ".join([_ for _ in first_lem if _ not in self._russian_stopwords]))
        return instance.Instance(inst.question, inst.id_, result, inst.counts, inst.sentiments)

    def replace_anglicisms(self, inst: instance.Instance):
        corrected_answers = list(
            map(lambda sentence: self._replace_anglicisms_str(sentence), inst.answers)
        )
        return instance.Instance(inst.question, inst.id_, corrected_answers, inst.counts, inst.sentiments)

    def get_sentiments(self, inst: instance.Instance):
        tokenizer = RegexTokenizer()
        model = FastTextSocialNetworkModel(tokenizer=tokenizer)
        list_of_sentiments = model.predict(inst.answers, k=1)
        sentiments = [
            sentiment if sentiment != 'speech' else 'positive' for sentiment in self._get_keys(list_of_sentiments)
        ]
        sentiments = [sentiment if sentiment != 'skip' else 'neutral' for sentiment in sentiments]
        return sentiments

    @staticmethod
    def delete_question_mark(inst: instance.Instance):
        p = re.compile('(Вопрос|Вопросы|Question|Questions)', flags=re.IGNORECASE)
        return instance.Instance(p.sub(" ", inst.question), inst.id_, inst.answers, inst.counts, inst.sentiments)

    @staticmethod
    def composition(prepr_func, inst):
        for func in prepr_func:
            inst = func(inst)
        return inst
