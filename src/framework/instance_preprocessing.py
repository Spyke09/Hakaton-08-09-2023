import instance
from spellchecker import spellchecker
import re


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
    def correct_errors(inst: instance.Instance):
        rus_cheker = spellchecker.SpellChecker("ru")
        eng_cheker = spellchecker.SpellChecker("en")
        eng = set("qwertyuiopasdfghjklzxcvbnm")
        rus = set("йцукенгшщзхъфывапролджэячсмитьбю")

        for i in range(len(inst.answers)):
            new_line = rus_cheker.correction(inst.answers[i])
            if new_line is not None:
                inst.answers[i] = new_line



