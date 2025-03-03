import json
import re
from functools import lru_cache

from evolutionary_forest.component.llm.gp_tree_converter import (
    population_to_json,
    generate_pattern,
    parent_to_json,
    generate_trees,
    json_to_individual,
    ChatGP,
)


@lru_cache(maxsize=None)
def get_llm_model():
    gpt = ChatGP()
    gpt.lazy_load()
    return gpt


def llm_pattern_extraction(population):
    gpt = get_llm_model()
    good_and_bad_individuals = population_to_json(population)
    pattern = generate_pattern(gpt, good_and_bad_individuals)
    return pattern


def llm_crossover(parent, pattern, pset):
    parent_json = parent_to_json(parent)
    gpt = get_llm_model()
    response = generate_trees(gpt, pattern, parent_json)
    parsed_json = extract_json_from_response(response)
    solution_list = []
    for k, v in parsed_json.items():
        solution_list.append(v["genes"])
    solutions = json_to_individual(solution_list, parent[0], pset)
    return solutions


def extract_json_from_response(response):
    # Extract JSON using regex
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        json_data = json_match.group()
        try:
            parsed_json = json.loads(json_data)
            print(parsed_json)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
    else:
        print("No JSON found in response.")
    return parsed_json
