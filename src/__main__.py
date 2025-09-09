from .LLM_Model import LLM_Model

DEFAULT_QUESTION = "Generate a small json output with at least, a random string and a number."

def clear_json_output(output: str):
	return output.replace("```json", "").replace("```", "").strip()

def main():
	model = LLM_Model()

	question = input("What kind of json do you want : ")
	if question == "":
		question = DEFAULT_QUESTION
	model.apply_json_creation_template(question)

	response_token_id = []
	for token_id in model.generate():
		response_token_id.append(token_id)
	# 	token = model.decode(token_id)
	# 	print(f"{token}", end='', flush=True)
	# print()
	output_file = open("./output/test.json", "w")
	output_file.write(clear_json_output(model._model._decode(response_token_id)))

if __name__ == "__main__":
	main()