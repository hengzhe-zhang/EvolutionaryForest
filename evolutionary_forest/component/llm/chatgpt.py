import os
from openai import OpenAI
import yaml


# Function to load API key from config.yaml
def load_api_key(config_file="config.yaml"):
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            return config.get("api_key")
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    except KeyError:
        raise KeyError("'api_key' not found in the configuration file.")


# Initialize OpenAI client using the new API structure
def initialize_openai():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is required to use the OpenAI API.")
    return OpenAI(api_key=api_key)


# Function to generate a response from ChatGPT using the new API
def generate_response(client, prompt, model="gpt-4o-mini"):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            seed=0,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to generate a response: {e}")


# Main function to demonstrate usage
def main():
    config_file = "config.yaml"
    try:
        api_key = load_api_key(config_file)
        os.environ["OPENAI_API_KEY"] = api_key

        # Initialize OpenAI client
        client = initialize_openai()

        prompt = "What is the capital of China?"
        response = generate_response(client, prompt)
        print(f"ChatGPT Response: {response}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
