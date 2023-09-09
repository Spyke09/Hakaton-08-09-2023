import main_model
import parser

path = "../data/all/324.json"

p = parser.SimpleJSONParser(path)
inst = p.get_instance()
m = main_model.SimpleModel()
m.fit(inst)
m.train()
m.dump("t.json")
