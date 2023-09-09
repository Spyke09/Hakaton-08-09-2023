import instance
import re
import ABC


class InstancePreprocessor:
    @staticmethod
    def to_lower(inst: instance.Instance):
        inst.question = inst.question.lower()
        inst.answers = [i.lower() for i in inst.answers]

    @staticmethod
    def delete_non_letters(inst: instance.Instance):
        for i in range(len(inst.answers)):
            inst.answers[i] = re.sub("[^а-яА-Яa-zA-Z]", " ", inst.answers[i])




