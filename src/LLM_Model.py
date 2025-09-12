from .llm_sdk import Small_LLM_Model
from .utils import top_k, softmax

QWEN3_DEFAULT_OUTPUT_LENGHT = 32768

QWEN3_BOS_TOKEN_ID = 151643
QWEN3_EOS_TOKEN_ID = 151645

DEFAULT_CHAT_PROMPT = """
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. terminate your response with <|im_end|> tag

### Instruction:
{instruction}

<|im_start|>user
### Question:
{question}<|im_end|>
<|im_start|>system
### Response:
<think>
</think>

"""

class Qwen3_Model:
	def __init__(self, temperature: float = 0.6, top_k: int = 20):
		self._model = Small_LLM_Model()
		self._prompt: str = ""
		self._input_ids: list[int] = []
		self.temperature = temperature
		self.top_k = top_k

	def _apply_default_chat_template(self, instruction: str, question: str):
		"""
		Apply default chat template with specific instruction and question.

		Warning: Prompt injection
		"""

		self._prompt = DEFAULT_CHAT_PROMPT.format_map({
			"instruction": instruction, 
			"question": question
		})

	def apply_json_function_creation(self, question: str):
		"""
		Apply template for json function creation
		"""

		function_definition = open("./input/functions_definition.json").read()
		instruction = f"""Your role is to create json object containing the prompt of the user, the function name responding to the prompt and the args of the function.
		### Functions
		{function_definition}

		### Examples
		{{
			"prompt": "What is the sum of 2 and 3?"
			"fn_name": "fn_add_numbers",
			"args": {{ "a": 2.0, "b": 3.0 }}
		}}
		"""
		self._apply_default_chat_template(instruction, question)

	def generate(self, max_token: int = QWEN3_DEFAULT_OUTPUT_LENGHT):
		"""
		Generate token until eos token is found.
		"""

		tensor = self._model._encode(self._prompt)
		self._input_ids = tensor.tolist()[0]
		for i in range(max_token):
			logits = self._model.get_logits_from_input_ids(self._input_ids)
			probs = softmax(logits, self.temperature)
			[pick, filtered_probs] = top_k(probs, self.top_k)
			self._input_ids.append(pick)
			if pick == QWEN3_EOS_TOKEN_ID:
				break
			yield pick

	def decode(self, token_id: int | list[int]):
		"""
		Decode wrapper
		"""

		if isinstance(token_id, int):
			token_id = [token_id]
		return self._model._decode(token_id)