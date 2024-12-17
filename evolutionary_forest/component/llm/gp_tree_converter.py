import json
import os

from deap import gp
from deap.gp import PrimitiveSet

from evolutionary_forest.component.llm.chatgpt import initialize_openai, load_api_key
from evolutionary_forest.utils import efficient_deepcopy


def tree_to_json(tree: gp.PrimitiveTree):
    return str(tree).replace("AQ", "Div")


# Function to convert JSON back to a DEAP tree
def json_to_tree(tree_json, pset: PrimitiveSet):
    return gp.PrimitiveTree.from_string(
        tree_json.replace("Arg", "ARG").replace("Div", "AQ"), pset
    )


# Function to serialize top individuals in the population to JSON
def population_to_json(population, top_k=5):
    top_individuals = sorted(
        population, key=lambda x: x.fitness.wvalues[0], reverse=True
    )[:top_k]
    bad_individuals = sorted(population, key=lambda x: x.fitness.wvalues[0])[:top_k]

    return_info = {}

    # Add top solutions with fitness
    for id, ind in enumerate(top_individuals):
        solution_key = f"Good Solution {id}"
        return_info[solution_key] = {
            "fitness": ind.fitness.wvalues[0],
            "genes": [tree_to_json(tree) for tree in ind.gene],
        }

    # Add bad solutions with fitness
    for id, ind in enumerate(bad_individuals):
        solution_key = f"Bad Solution {id}"
        return_info[solution_key] = {
            "fitness": ind.fitness.wvalues[0],
            "genes": [tree_to_json(tree) for tree in ind.gene],
        }

    return json.dumps(return_info, indent=4)


def parent_to_json(population, top_k=5):
    top_individuals = sorted(
        population, key=lambda x: x.fitness.wvalues[0], reverse=True
    )[:top_k]

    return_info = {}

    # Add current solutions with fitness
    for id, ind in enumerate(top_individuals):
        solution_key = f"Current Solution {id}"
        return_info[solution_key] = {
            "fitness": ind.fitness.wvalues[0],
            "genes": [tree_to_json(tree) for tree in ind.gene],
        }

    return json.dumps(return_info, indent=4)


# Function to deserialize JSON back to DEAP individuals
def json_to_individual(data, template, pset: PrimitiveSet):
    population = []
    for tree_list in data:
        trees = []
        for tree_json in tree_list:
            trees.append(json_to_tree(tree_json, pset))
        individual = efficient_deepcopy(template)
        individual.gene = trees
        population.append(individual)
    return population


class ChatGP:
    def __init__(self):
        self.client = None

    def lazy_load(self):
        if self.client is not None:
            return
        config_file = "config.yaml"
        api_key = load_api_key(config_file)
        os.environ["OPENAI_API_KEY"] = api_key

        # Initialize OpenAI client
        self.client = initialize_openai()


def generate_trees(chatgp: ChatGP, pattern, parent_json, tree_num=10):
    chatgp.lazy_load()
    if pattern == "":
        llm_text = f"""
You are a helpful assistant tasked with extracting patterns from good and bad solutions in the context of retrieval symbolic regression. Each solution contains multiple features, and the objective of feature construction is to enable effective contrastive learning. The goal is to construct a feature space where samples with similar labels are positioned closer together.

Given several patterns and {2} parent solutions, each with {tree_num} features provided in JSON format, analyze the input to generate a list of {2} better solutions, output in JSON format.

Parent Solutions:
{parent_json}

Please only output the generated solutions. Ensure the solutions follows the grammar of parent solution, avoid use non-exist functions and non-exist terminals, which lead to python exception.
                    """
    else:
        llm_text = f"""
You are a helpful assistant tasked with extracting patterns from good and bad solutions in the context of retrieval symbolic regression. Each solution contains multiple features, and the objective of feature construction is to enable effective contrastive learning. The goal is to construct a feature space where samples with similar labels are positioned closer together.

Given several patterns and {2} parent solutions, each with {tree_num} features provided in JSON format, analyze the input to generate a list of {2} better solutions, output in JSON format. Ensure the output solutions incorporate the learned patterns to enhance quality.

Pattern:
{pattern}

Parent Solutions:
{parent_json}

Please only output the generated solutions. Ensure the solutions follows the grammar of parent solution, avoid use non-exist functions and non-exist terminals, which lead to python exception.
                    """
    chat_completion = chatgp.client.chat.completions.create(
        messages=[{"role": "user", "content": llm_text}],
        model="gpt-4o-mini",
        seed=0,
    )
    return chat_completion.choices[0].message.content


def generate_pattern(chatgpt: ChatGP, good_bad_json, top_k=5, tree_num=10):
    chatgpt.lazy_load()
    chat_completion = chatgpt.client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
    You are a helpful assistant tasked with extracting patterns from good and bad solutions in the context of retrieval symbolic regression. Each solution contains multiple features, and the objective of feature construction is to enable effective contrastive learning. The goal is to construct a feature space where samples with similar labels are positioned closer together.

    Given {top_k} good solutions and {top_k} bad solutions, each with {tree_num} features provided in JSON format, analyze the solutions to identify patterns that differentiate good solutions from bad ones. Use these patterns to provide actionable insights for guiding evolutionary algorithms to generate better solutions. The output should be in JSON format.

    {good_bad_json}
                """,
            }
        ],
        model="gpt-4o-mini",
        seed=0,
    )
    return chat_completion.choices[0].message.content
