import json

import instance


class SimpleJSONParser:
    def __init__(self, path: str):
        if path[-5:] != ".json":
            raise ValueError("File should have .json extension.")

        with open(path, "r") as file:
            j_file = json.load(file)

            self._question = j_file["question"]
            self._id = j_file["id"]
            self._answers = [i["answer"] for i in j_file["answers"]]
            self._counts = [i["count"] for i in j_file["answers"]]

    def get_instance(self) -> instance.Instance:
        return instance.Instance(self._question, self._id, self._answers, self._counts)
