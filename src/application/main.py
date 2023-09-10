import sys
sys.path.insert(0, '../../')

import src.framework.main_model
import src.framework.json_parser
import click




@click.command()
@click.argument("path_to_instance", type=click.Path())
@click.argument("path_to_result", type=click.Path())
def main(path_to_instance, path_to_result):
    p = src.framework.json_parser.SimpleJSONParser(path_to_instance)
    inst = p.get_instance()
    m = src.framework.main_model.SimpleModel()
    m.fit(inst)
    m.train()
    m.dump(path_to_result)
    m.draw_word_cloud(path_to_result)
    m.save_to_excel(path_to_result)


if __name__ == "__main__":
    main()
