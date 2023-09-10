import src.framework.main_model
import src.framework.json_parser


if __name__ == "__main__":
    path = "../data/all/14507.json"

    p = src.framework.json_parser.SimpleJSONParser(path)
    inst = p.get_instance()
    m = src.framework.main_model.SimpleModel()
    m.fit(inst)
    m.train()
    m.dump("../tests/t.json")
    m.show_stats()
