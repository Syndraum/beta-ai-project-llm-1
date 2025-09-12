from .LLM_Model import Qwen3_Model

def clear_json_output(output: str):
	return output.replace("```json", "").replace("```", "").strip()

def main():
	model = Qwen3_Model()
	response_token_id = []

	question = input("How can I help you today?\n>> ")
	model.apply_json_function_creation(question)
	for token_id in model.generate():
		response_token_id.append(token_id)
		token = model.decode(token_id)
		print(f"{token}", end='', flush=True)
	print()

	output_file = open("./output/response.json", "w")
	output_file.write(clear_json_output(model._model._decode(response_token_id)))

if __name__ == "__main__":
	main()