import instance
import nltk
import re
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
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

nltk.download("stopwords")

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
spell = SpellChecker(language="ru")
russian_stopwords = stopwords.words("russian")


def get_abbreviations(f):
    abbreviations = dict()
    with open(f) as file:
        for line in file.readlines():
            key, value = map(str.strip, line.split('->'))
            abbreviations[key] = value
    return abbreviations


def replace_abbreviations_str(sentence):
    abbreviations = get_abbreviations('../data/сокращения.txt')
    words = sentence.lower().split()
    for i in range(len(words)):
        if words[i] in abbreviations:
            words[i] = abbreviations[words[i]]
    return ' '.join(words)


def replace_anglicisms_str(sentence):
    anglicisms = get_abbreviations('../data/англицизмы.txt')
    words = sentence.lower().split()
    for i in range(len(words)):
        if words[i] in anglicisms:
            words[i] = anglicisms[words[i]]
    return ' '.join(words)


def get_keys(list_of_dicts):
    keys = []
    for d in list_of_dicts:
        for key in d:
            keys.append(key)
    return keys


class InstancePreprocessor:
    @staticmethod
    def to_lower(inst: instance.Instance):
        inst.question = inst.question.lower()
        inst.answers = [i.lower() for i in inst.answers]

    @staticmethod
    def delete_non_letters(inst: instance.Instance):
        for i in range(len(inst.answers)):
            inst.answers[i] = re.sub("[^а-яА-Яa-zA-Z]", " ", inst.answers[i])

    @staticmethod
    def replace_abbreviations(inst: instance.Instance):
        corrected_answers = list(
            map(lambda sentence: replace_abbreviations_str(sentence), inst.answers)
        )
        return instance.Instance(inst.question, inst.id, corrected_answers, inst.counts)

    @staticmethod
    def token_lemmatization_spc_natasha(inst: instance.Instance):
        result = []
        for answer in inst.answers:
            tokens = re.sub("[^а-яА-Я]", " ", answer)

            doc = Doc(" ".join(tokens))
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            for token in doc.tokens:
                token.lemmatize(morph_vocab)
            doc = Doc(tokens)

            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)

            for token in doc.tokens:
                token.lemmatize(morph_vocab)

            result.append(" ".join([_.lemma for _ in doc.tokens if _.lemma not in russian_stopwords]))
        return instance.Instance(inst.question, inst.id, result, inst.counts)

    @staticmethod
    def token_lemmatization_natasha(inst: instance.Instance):
        result = []
        for answer in inst.answers:
            tokens = re.sub("[^а-яА-Я]", " ", answer)
            first_lem = [list(spell.candidates(i))[0].lower()
                         if spell.candidates(i) is not None
                         else i.lower()
                         for i in tokens.split(" ")]
            doc = Doc(" ".join(first_lem))
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            for token in doc.tokens:
                token.lemmatize(morph_vocab)
            result.append(" ".join([_.lemma for _ in doc.tokens if _.lemma not in russian_stopwords]))

        return instance.Instance(inst.question, inst.id, result, inst.counts)

    @staticmethod
    def token_lemmatization_spc(inst: instance.Instance):
        result = []
        for answer in inst.answers:
            tokens = re.sub("[^а-яА-Я]", " ", answer)
            first_lem = [list(spell.candidates(i))[0].lower()
                         if spell.candidates(i) is not None
                         else i.lower()
                         for i in tokens.split(" ")]
            result.append(" ".join([_ for _ in first_lem if _ not in russian_stopwords]))
        return instance.Instance(inst.question, inst.id, result, inst.counts)

    @staticmethod
    def replace_anglicisms(inst: instance.Instance):
        corrected_answers = list(
            map(lambda sentence: replace_anglicisms_str(sentence), inst.answers)
        )
        return instance.Instance(inst.question, inst.id, corrected_answers, inst.counts)

    @staticmethod
    def get_sentiments(inst: instance.Instance):
        tokenizer = RegexTokenizer()
        model = FastTextSocialNetworkModel(tokenizer=tokenizer)
        list_of_sentiments = model.predict(inst.answers, k=1)
        sentiments = [sentiment if sentiment != 'speech' else 'positive' for sentiment in get_keys(list_of_sentiments)]
        sentiments = [sentiment if sentiment != 'skip' else 'neutral' for sentiment in sentiments]
        return instance.Instance(inst.question, inst.id, inst.answers, inst.counts, sentiments)

    @staticmethod
    def delete_question_mark(inst: instance.Instance):
        p = re.compile('(Вопрос|Вопросы|Question|Questions)', flags=re.IGNORECASE)
        return p.sub(" ", inst.question)
