from .llm_sdk import Small_LLM_Model

def main():
    print("Hello from ai-beta-llm!")

    model = Small_LLM_Model()
    path = model.get_path_to_vocabulary_json()
    print(f"vocabulary path: {path}")


if __name__ == "__main__":
    main()
