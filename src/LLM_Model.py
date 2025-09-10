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

class LLM_Model:
	def __init__(self, temperature: float = 0.6, top_k: int = 20):
		self._model = Small_LLM_Model()
		self._prompt: str = ""
		self._input_ids: list[int] = []
		self.temperature = temperature
		self.top_k = top_k

	def _apply_default_chat_template(self, instruction: str, question: str):
		self._prompt = DEFAULT_CHAT_PROMPT.format_map({
			"instruction": instruction, 
			"question": question
		})

	def apply_json_creation_template(self, question: str):
		instruction = "Your role is to create json object. The output must be valid JSON (no trailing commas, no comments). No extra keys or prose are allowed anywhere in the output."
		self._apply_default_chat_template(instruction, question)

	def generate(self, max_token: int = QWEN3_DEFAULT_OUTPUT_LENGHT):
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

	def decode(self, token_id: int):
		return self._model._decode([token_id])