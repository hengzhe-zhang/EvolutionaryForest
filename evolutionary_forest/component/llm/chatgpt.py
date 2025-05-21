from openai import OpenAI
import yaml

open_router = False
use_gemini = True


# Function to load API key from config.yaml
def load_api_key(config_file="config.yaml", service=None):
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            if service == "Gemini":
                return config.get("gemini_key")
            elif service == "DeepSeek":
                return config.get("deepseek_key")
            elif service == "OpenAI":
                return config.get("api_key")
            else:
                raise ValueError("Unknown service")
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    except KeyError:
        raise KeyError("'api_key' not found in the configuration file.")


# Initialize OpenAI client using the new API structure
def initialize_openai(api_key, service):
    if service == "Gemini":
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif service == "DeepSeek":
        return OpenAI(
            api_key=api_key,
            base_url="https://llm.chutes.ai/v1/",
        )
    elif service == "OpenAI":
        return OpenAI(api_key=api_key)
    else:
        raise Exception(f"Unknown service '{service}'")


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

        # Initialize OpenAI client
        client = initialize_openai(api_key)

        prompt = "What is the capital of China?"
        response = generate_response(client, prompt)
        print(f"ChatGPT Response: {response}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
