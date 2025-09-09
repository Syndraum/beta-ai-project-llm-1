from .LLM_Model import LLM_Model

DEFAULT_QUESTION = "Generate a small json output with at least, a random string and a number."

def main():
    model = LLM_Model()

    question = input("What kind of json do you want : ")
    if question == "":
        question = DEFAULT_QUESTION
    model.apply_json_creation_template(question)
    for token_id in model.generate():
        token = model.decode(token_id)
        print(f"{token}", end='', flush=True)
    print()

if __name__ == "__main__":
    main()