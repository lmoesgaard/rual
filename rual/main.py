import sys
import json
import importlib
from rual.al.rual import RUAL


def get_scoring_tool(tool):
    module = importlib.import_module(tool["module"])
    classname = tool["classname"]
    try:
        return getattr(module, classname)
    except:
        print(f"Could not find scorer: {classname}")
        sys.exit()

def main(json_filename):
    input = json.load(open(json_filename))
    rual_instance = RUAL(input)
    rual_instance.scorer = get_scoring_tool(input['scorer'])(input)
    while not rual_instance.final:
        rual_instance.new_round()

if __name__ == '__main__':
    json_filename = sys.argv[2]
    main(json_filename)
