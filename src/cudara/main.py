"""
Cudara Inference Server (Transformers Native)
=============================================
Definitive Unified Edition.
"""

import asyncio
import base64
import contextlib
import datetime
import gc
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import warnings
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Literal, Optional, Union

import torch
import transformers.activations
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import login, snapshot_download
from pydantic import BaseModel, ConfigDict, Field
from qwen_vl_utils import process_vision_info  # type: ignore
from starlette.exceptions import HTTPException as StarletteHTTPException
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
    pipeline,
)

if not hasattr(transformers.activations, "PytorchGELUTanh"):
    transformers.activations.PytorchGELUTanh = type("PytorchGELUTanh", (object,), {})  # type: ignore
warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cudara")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if HF_TOKEN := os.getenv("HF_TOKEN"):
    login(token=HF_TOKEN, add_to_git_credential=False)

MODELS_DIR, TEMP_DIR = Path("models"), Path("temp_uploads")
for d in (MODELS_DIR, TEMP_DIR):
    d.mkdir(exist_ok=True)

GPU_LOCK = asyncio.Lock()
IDLE_TIMEOUT_SECONDS = 300


class AppError(Exception):
    """Custom application error mapped to standard JSON error format."""

    def __init__(self, message: str, status_code: int = 500, code: str = "error") -> None:
        self.message = message
        self.status_code = status_code
        self.code = code


class TokenLogprob(BaseModel):
    """Log probability information for a single token alternative."""

    token: str = Field(..., description="Text representation of the token")
    logprob: float = Field(..., description="Log probability of this token")
    bytes: Optional[List[int]] = Field(None, description="Raw byte representation")


class Logprob(BaseModel):
    """Log probability information for a generated token."""

    token: str = Field(..., description="Text representation of the token")
    logprob: float = Field(..., description="Log probability of this token")
    bytes: Optional[List[int]] = Field(None, description="Raw byte representation")
    top_logprobs: Optional[List[TokenLogprob]] = Field(
        None, description="Most likely tokens and their log probabilities"
    )


class ModelOptions(BaseModel):
    """Runtime options that control text generation."""

    seed: Optional[int] = Field(None, description="Random seed for reproducible outputs")
    temperature: Optional[float] = Field(None, description="Randomness in generation")
    top_k: Optional[int] = Field(None, description="Limits next token selection to K most likely")
    top_p: Optional[float] = Field(None, description="Cumulative probability threshold")
    min_p: Optional[float] = Field(None, description="Minimum probability threshold")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    num_ctx: Optional[int] = Field(None, description="Context length size")
    num_predict: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    model_config = ConfigDict(extra="allow")


class GenerateRequest(BaseModel):
    """Payload for generating a response from a model."""

    model: str = Field(..., description="Model name")
    prompt: Optional[str] = Field(None, description="Text for the model to generate a response from")
    suffix: Optional[str] = Field(None, description="Text that appears after the user prompt for FIM models")
    images: Optional[List[str]] = Field(None, description="Base64-encoded images")
    format: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Structured output format constraints")
    system: Optional[str] = Field(None, description="System prompt")
    stream: bool = Field(True, description="Returns a stream of partial responses if true")
    think: Optional[Union[bool, Literal["high", "medium", "low"]]] = Field(
        None, description="Returns separate thinking output"
    )
    raw: bool = Field(False, description="Returns raw response without prompt templating")
    keep_alive: Optional[Union[str, int]] = Field(None, description="Model keep-alive duration")
    options: Optional[ModelOptions] = Field(None, description="Runtime options")
    logprobs: Optional[bool] = Field(None, description="Return log probabilities")
    top_logprobs: Optional[int] = Field(None, description="Number of top logprobs to return")


class GenerateResponse(BaseModel):
    """Standard, non-streaming text generation response."""

    model: str = Field(..., description="Model name")
    created_at: str = Field(..., description="ISO 8601 timestamp")
    response: str = Field(..., description="Generated text response")
    thinking: Optional[str] = Field(None, description="Generated thinking output")
    done: bool = Field(..., description="Indicates whether generation has finished")
    done_reason: Optional[str] = Field(None, description="Reason the generation stopped")
    total_duration: Optional[int] = Field(None, description="Total duration in nanoseconds")
    load_duration: Optional[int] = Field(None, description="Load duration in nanoseconds")
    prompt_eval_count: Optional[int] = Field(None, description="Prompt token count")
    prompt_eval_duration: Optional[int] = Field(None, description="Prompt evaluation duration in nanoseconds")
    eval_count: Optional[int] = Field(None, description="Generated token count")
    eval_duration: Optional[int] = Field(None, description="Generation duration in nanoseconds")
    logprobs: Optional[List[Logprob]] = Field(None, description="Log probability information")


class GenerateStreamEvent(BaseModel):
    """A single event chunk from a streaming generation response."""

    model: str = Field(..., description="Model name")
    created_at: str = Field(..., description="ISO 8601 timestamp")
    response: str = Field(..., description="Generated text response chunk")
    thinking: Optional[str] = Field(None, description="Generated thinking output chunk")
    done: bool = Field(..., description="Indicates whether stream has finished")
    done_reason: Optional[str] = Field(None, description="Reason stream stopped")
    total_duration: Optional[int] = Field(None)
    load_duration: Optional[int] = Field(None)
    prompt_eval_count: Optional[int] = Field(None)
    prompt_eval_duration: Optional[int] = Field(None)
    eval_count: Optional[int] = Field(None)
    eval_duration: Optional[int] = Field(None)


class ToolCallFunction(BaseModel):
    """Details of the function to call produced by the model."""

    name: str = Field(..., description="Name of the function to call")
    description: Optional[str] = Field(None, description="Function description")
    arguments: Optional[Dict[str, Any]] = Field(None, description="Arguments to pass")


class ToolCall(BaseModel):
    """A tool call request produced by the model."""

    function: ToolCallFunction


class ToolDefinitionFunction(BaseModel):
    """Function details exposed to the model for tool calling."""

    name: str = Field(..., description="Function name exposed to the model")
    description: Optional[str] = Field(None, description="Description of the function")
    parameters: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for parameters")


class ToolDefinition(BaseModel):
    """Definition of a tool the model can use."""

    type: Literal["function"] = Field("function", description="Type of tool")
    function: ToolDefinitionFunction


class ChatMessage(BaseModel):
    """A single message in the chat history."""

    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="Message author")
    content: str = Field(..., description="Message text content")
    images: Optional[List[str]] = Field(None, description="Base64-encoded images")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls")


class ChatRequest(BaseModel):
    """Payload for generating a chat response."""

    model: str = Field(..., description="Model name")
    messages: List[ChatMessage] = Field(..., description="Chat history")
    tools: Optional[List[ToolDefinition]] = Field(None, description="Available functions")
    format: Optional[Union[Literal["json"], Dict[str, Any]]] = Field(None, description="Response format constraint")
    options: Optional[ModelOptions] = Field(None, description="Runtime options")
    stream: bool = Field(True, description="Stream responses if true")
    think: Optional[Union[bool, Literal["high", "medium", "low"]]] = Field(None, description="Include thinking trace")
    keep_alive: Optional[Union[str, int]] = Field(None, description="Model keep-alive duration")
    logprobs: Optional[bool] = Field(None, description="Return log probabilities")
    top_logprobs: Optional[int] = Field(None, description="Number of top logprobs")


class ChatResponseMessage(BaseModel):
    """The generated message content inside the response."""

    role: Literal["assistant"] = Field("assistant", description="Message author")
    content: str = Field(..., description="Assistant message text")
    thinking: Optional[str] = Field(None, description="Deliberate thinking trace")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Requested tool calls")
    images: Optional[List[str]] = Field(None, description="Base64-encoded images")


class ChatResponse(BaseModel):
    """Standard, non-streaming chat response object."""

    model: str = Field(..., description="Model name")
    created_at: str = Field(..., description="ISO 8601 timestamp")
    message: ChatResponseMessage
    done: bool = Field(..., description="Indicates completion")
    done_reason: Optional[str] = Field(None, description="Reason for completion")
    total_duration: Optional[int] = Field(None)
    load_duration: Optional[int] = Field(None)
    prompt_eval_count: Optional[int] = Field(None)
    prompt_eval_duration: Optional[int] = Field(None)
    eval_count: Optional[int] = Field(None)
    eval_duration: Optional[int] = Field(None)
    logprobs: Optional[List[Logprob]] = Field(None)


class ChatStreamEvent(BaseModel):
    """A single event chunk from a streaming chat response."""

    model: str = Field(..., description="Model name")
    created_at: str = Field(..., description="ISO 8601 timestamp")
    message: ChatResponseMessage
    done: bool = Field(..., description="Indicates completion")
    done_reason: Optional[str] = Field(None, description="Reason for completion")
    total_duration: Optional[int] = Field(None)
    load_duration: Optional[int] = Field(None)
    prompt_eval_count: Optional[int] = Field(None)
    prompt_eval_duration: Optional[int] = Field(None)
    eval_count: Optional[int] = Field(None)
    eval_duration: Optional[int] = Field(None)


class ModelStatus(str, Enum):
    """Current state of a model in the local registry."""

    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


class ModelConfig(BaseModel):
    """Configuration mapping for downloaded and available models."""

    description: Optional[str] = None
    task: str = "text-generation"
    backend: str = "transformers"
    system_prompt: Optional[str] = None


class RegistryItem(BaseModel):
    """Tracks a model's installation progress and local path."""

    status: ModelStatus
    local_path: Optional[str] = None
    error_message: Optional[str] = None


class EmbedRequest(BaseModel):
    """Payload for generating embeddings."""

    model: str = Field(..., description="Model name")
    input: Union[str, List[str]] = Field(..., description="Text or array of texts to embed")
    truncate: bool = Field(True, description="Truncate inputs exceeding context window")
    dimensions: Optional[int] = Field(None, description="Output dimensions")
    keep_alive: Optional[Union[str, int]] = Field(None, description="Keep-alive duration")
    options: Optional[ModelOptions] = Field(None, description="Runtime options")


class EmbedResponse(BaseModel):
    """Response object containing embeddings or rerank scores."""

    model: str = Field(..., description="Model name")
    embeddings: Optional[List[List[float]]] = Field(None, description="Array of vector embeddings")
    scores: Optional[List[float]] = Field(None, description="Relevance scores for reranking")
    total_duration: Optional[int] = Field(None, description="Total duration in nanoseconds")
    load_duration: Optional[int] = Field(None, description="Load duration in nanoseconds")
    prompt_eval_count: Optional[int] = Field(None, description="Tokens processed")


class ModelDetails(BaseModel):
    """Metadata detailing the model's format and architecture."""

    format: str = Field(..., description="Model file format")
    family: str = Field(..., description="Primary model family")
    families: Optional[List[str]] = Field(None, description="All families")
    parameter_size: Optional[str] = Field(None, description="Approximate parameter count")
    quantization_level: Optional[str] = Field(None, description="Quantization level")


class ModelSummary(BaseModel):
    """Summary information for a locally available model."""

    name: str = Field(..., description="Model name")
    model: str = Field(..., description="Model name alias")
    modified_at: str = Field(..., description="ISO 8601 last modified timestamp")
    size: int = Field(..., description="Size on disk in bytes")
    digest: str = Field(..., description="Model contents digest")
    details: ModelDetails


class ListResponse(BaseModel):
    """Response containing the list of available models."""

    models: List[ModelSummary]


class Ps(BaseModel):
    """Information about a currently running model."""

    name: str = Field(..., description="Name of the running model")
    model: str = Field(..., description="Name of the running model")
    size: int = Field(..., description="Size of the model in bytes")
    digest: str = Field(..., description="SHA256 digest")
    details: ModelDetails = Field(..., description="Model details")
    expires_at: str = Field(..., description="Time when the model will be unloaded")
    size_vram: int = Field(..., description="VRAM usage in bytes")
    context_length: int = Field(0, description="Context length capacity")


class PsResponse(BaseModel):
    """Response containing the list of models currently loaded into memory."""

    models: List[Ps] = Field(..., description="Currently running models")


class ShowRequest(BaseModel):
    """Payload for showing model details."""

    model: str = Field(..., description="Model name to show")
    verbose: bool = Field(False, description="Include large verbose fields")


class ShowResponse(BaseModel):
    """Response containing detailed model information."""

    parameters: Optional[str] = Field(None, description="Parameter settings")
    license: Optional[str] = Field(None, description="Model license")
    modified_at: str = Field(..., description="Last modified timestamp")
    details: ModelDetails = Field(..., description="High-level details")
    template: Optional[str] = Field(None, description="Prompt template")
    capabilities: List[str] = Field(..., description="Supported features")
    model_info: Dict[str, Any] = Field(..., description="Additional metadata")


class PullRequest(BaseModel):
    """Payload for pulling a model from a remote registry."""

    model: str = Field(..., description="Name of the model to download")
    insecure: bool = Field(False, description="Allow insecure connections")
    stream: bool = Field(True, description="Stream progress updates")


class DeleteRequest(BaseModel):
    """Payload for deleting a model."""

    model: str = Field(..., description="Model name to delete")


class ModelManager:
    """Manages the local model registry, configurations, and downloads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        if not Path("registry.json").exists():
            self._save_json(Path("registry.json"), {})

    def _save_json(self, path: Path, data: Any) -> None:
        """Atomically saves JSON data to disk."""
        with self._lock:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Loads JSON data safely."""
        with self._lock:
            if not path.exists():
                return {}
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}

    def get_allowed(self) -> Dict[str, ModelConfig]:
        """Retrieves allowed models from models.json."""
        data = self._load_json(Path("models.json"))
        return {k: ModelConfig(**v) for k, v in data.items() if not k.startswith("_")}

    def get_registry(self) -> Dict[str, RegistryItem]:
        """Retrieves the current downloaded model registry."""
        return {k: RegistryItem(**v) for k, v in self._load_json(Path("registry.json")).items()}

    def update_registry(self, model_id: str, **kwargs: Any) -> None:
        """Updates specific attributes of a model in the registry."""
        reg = self._load_json(Path("registry.json"))
        reg.setdefault(model_id, {}).update(kwargs)
        self._save_json(Path("registry.json"), reg)

    def download_task(self, model_id: str) -> None:
        """Executes model download from Hugging Face."""
        try:
            local_dir = MODELS_DIR / model_id.replace("/", "--")
            snapshot_download(repo_id=model_id, local_dir=local_dir, ignore_patterns=["*.msgpack", "*.h5", "*.bin"])
            self.update_registry(model_id, status=ModelStatus.READY, local_path=str(local_dir))
        except Exception as e:
            self.update_registry(model_id, status=ModelStatus.ERROR, error_message=str(e))

    def delete_model(self, model_id: str) -> None:
        """Removes model assets and registry entries."""
        reg = self._load_json(Path("registry.json"))
        if model_id in reg:
            path = reg[model_id].get("local_path")
            if path and Path(path).exists():
                shutil.rmtree(path)
            del reg[model_id]
            self._save_json(Path("registry.json"), reg)


class InferenceEngine:
    """Handles PyTorch and Transformers inference operations."""

    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager
        self.active_id: Optional[str] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.processor: Optional[Any] = None
        self._lock = threading.RLock()
        self.last_interaction: float = 0.0
        self.active_requests: int = 0
        self.keep_alive_timeout: float = float(IDLE_TIMEOUT_SECONDS)

    @contextlib.contextmanager
    def request_context(self):
        """Context manager to safely track when the GPU is busy."""
        with self._lock:
            self.active_requests += 1
        try:
            yield
        finally:
            with self._lock:
                self.active_requests -= 1
                # Reset the idle timer ONLY when the generation actually finishes
                self.last_interaction = time.time()

    def unload(self) -> None:
        """Releases VRAM by safely severing references and forcing PyTorch cleanup."""
        with self._lock:  # Secure the thread so we don't unload during active use
            if self.model is not None:
                logger.info(f"Unloading model {self.active_id} from VRAM.")

                # 1. Reassigning to None is safer and cleaner than using `del`
                self.model = None
                self.tokenizer = None
                self.processor = None
                self.active_id = None

                # 2. Double GC pass. PyTorch cyclical references often survive a single pass.
                gc.collect()
                gc.collect()

                # 3. Aggressively clear CUDA caches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()  # Clears dead inter-process memory segments

    def load(self, model_id: str) -> None:
        """Loads model weights into VRAM."""
        self.last_interaction = time.time()
        if self.active_id == model_id:
            return

        config = self.manager.get_allowed().get(model_id)
        reg = self.manager.get_registry().get(model_id)
        if not config or not reg or reg.status != "ready" or not reg.local_path:
            raise AppError(f"Model '{model_id}' not found or unavailable.", 404)

        with self._lock:
            self.unload()
            path = reg.local_path
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

            logger.info(f"Loading {model_id} into VRAM...")
            is_awq = "AWQ" in model_id.upper()
            attn_impl = "eager" if is_awq else "sdpa"

            if config.task == "text-generation":
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    path, device_map="auto", dtype=dtype, attn_implementation=attn_impl
                )
            elif config.task == "image-to-text":
                self.processor = AutoProcessor.from_pretrained(path)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    path, device_map="auto", dtype=dtype, attn_implementation=attn_impl
                )
            elif config.task == "automatic-speech-recognition":
                self.model = pipeline("automatic-speech-recognition", model=path, device=device, dtype=dtype)
            elif config.task in ["feature-extraction", "text-classification"]:
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                m_cls = AutoModel if config.task == "feature-extraction" else AutoModelForSequenceClassification
                self.model = m_cls.from_pretrained(path, dtype=dtype).to(device)
            self.active_id = model_id

    def chat(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        options: Dict[str, Any],
        stream: bool,
        is_chat: bool,
        force_json: bool,
    ) -> Union[Iterator[str], Dict[str, Any]]:
        """Core logic for text and multimodal generation operations."""
        start_t = time.perf_counter()
        self.load(model_id)
        load_t = time.perf_counter() - start_t
        config = self.manager.get_allowed()[model_id]

        if config.task == "image-to-text" and self.processor and self.model:
            hf_msgs = []
            for m in messages:
                content = [{"type": "text", "text": m["content"]}]
                if m.get("images"):
                    for img in m["images"]:
                        content.append({"type": "image", "image": f"data:image/jpeg;base64,{img}"})
                hf_msgs.append({"role": m["role"], "content": content})
            text = self.processor.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(hf_msgs)
            inputs = self.processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(
                self.model.device
            )
            tok = self.processor.tokenizer
        elif self.tokenizer and self.model:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            tok = self.tokenizer
        else:
            raise AppError("Model configuration invalid for text generation.", 500)

        prompt_eval_count = inputs.input_ids.shape[1]
        eval_start_t = time.perf_counter()

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=15.0)
        temp = options.get("temperature", 0.8)

        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": options.get("num_predict", 512),
        }

        if temp <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            if "top_p" in options:
                gen_kwargs["top_p"] = options["top_p"]
            if "top_k" in options:
                gen_kwargs["top_k"] = options["top_k"]

        def _generate_with_error_handling():
            try:
                self.model.generate(**gen_kwargs)
            except Exception as e:
                logger.error(f"Generation thread crashed: {e}")
                # Push the stop signal into the streamer queue to unblock the main thread
                streamer.text_queue.put(streamer.stop_signal, timeout=streamer.timeout)

        threading.Thread(target=_generate_with_error_handling).start()

        def sync_gen() -> Iterator[Any]:
            with self.request_context():
                full_txt = ""
                thinking = ""
                in_thought = False
                t_count = 0
                first_tok = None

                try:
                    for new_text in streamer:
                        if first_tok is None:
                            first_tok = time.perf_counter()
                        t_count += 1

                        if "<think>" in new_text:
                            in_thought = True
                            continue
                        if "</think>" in new_text:
                            in_thought = False
                            continue

                        if in_thought:
                            thinking += new_text
                        else:
                            full_txt += new_text

                        if stream:
                            chunk: Dict[str, Any] = {
                                "model": model_id,
                                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "done": False,
                            }
                            if is_chat:
                                chunk["message"] = {"role": "assistant", "content": new_text}
                            else:
                                chunk["response"] = new_text
                            if thinking:
                                chunk["thinking"] = thinking
                            yield json.dumps(chunk) + "\n"

                except queue.Empty:
                    logger.error("TextIteratorStreamer timed out. Model likely crashed.")

            eval_end = time.perf_counter()

            if force_json and not stream:
                match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", full_txt, re.DOTALL)
                if match:
                    full_txt = match.group(1).strip()
                else:
                    match = re.search(r"(\{.*?\}|\[.*?\])", full_txt, re.DOTALL)
                    if match:
                        full_txt = match.group(1).strip()

            res: Dict[str, Any] = {
                "model": model_id,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "done": True,
                "done_reason": "stop",
                "total_duration": int((eval_end - start_t) * 1e9),
                "load_duration": int(load_t * 1e9),
                "prompt_eval_count": prompt_eval_count,
                "eval_count": t_count,
                "eval_duration": int((eval_end - (first_tok or eval_start_t)) * 1e9),
            }

            if is_chat:
                res["message"] = {"role": "assistant", "content": full_txt}
            else:
                res["response"] = full_txt

            if thinking:
                res["thinking"] = thinking

            if stream:
                yield json.dumps(res) + "\n"
            else:
                yield res

        if stream:
            return sync_gen()
        return next(sync_gen())

    def embeddings(
        self, model_id: str, texts: List[str], truncate: bool = True, dimensions: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculates vector embeddings for a given input."""
        start = time.perf_counter()
        self.load(model_id)
        load_t = time.perf_counter() - start

        if not self.tokenizer or not self.model:
            raise AppError("Model configuration invalid for embeddings.", 500)

        try:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=truncate).to(self.model.device)
        except Exception as e:
            raise AppError(f"Tokenization failed: {str(e)}", 400)

        if not truncate and hasattr(self.tokenizer, "model_max_length"):
            if inputs.input_ids.shape[1] > self.tokenizer.model_max_length:
                raise AppError(
                    f"Input exceeds maximum context length of {self.tokenizer.model_max_length}. "
                    "Set truncate=True to ignore.",
                    400,
                )

        with torch.no_grad():
            output = self.model(**inputs).last_hidden_state.mean(dim=1)
            if dimensions is not None:
                output = output[:, :dimensions]
            embeddings_list = output.cpu().tolist()

        return {
            "model": model_id,
            "embeddings": embeddings_list,
            "total_duration": int((time.perf_counter() - start) * 1e9),
            "load_duration": int(load_t * 1e9),
            "prompt_eval_count": inputs.input_ids.numel(),
        }

    def rerank(self, model_id: str, query: str, docs: List[str]) -> Dict[str, Any]:
        """Calculates relevance scores for reranking workflows."""
        start = time.perf_counter()
        self.load(model_id)
        if not self.tokenizer or not self.model:
            raise AppError("Model configuration invalid for reranking.", 500)

        pairs = [[query, doc] for doc in docs]
        inputs = self.tokenizer(pairs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            self.model.device
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits.view(-1).float()
            scores = torch.sigmoid(logits).cpu().tolist()

        return {"model": model_id, "scores": scores, "total_duration": int((time.perf_counter() - start) * 1e9)}

    def transcribe(self, model_id: str, path: str) -> Dict[str, Any]:
        """Processes audio files using Automatic Speech Recognition."""
        start = time.perf_counter()
        self.load(model_id)
        if not self.model:
            raise AppError("Model configuration invalid for transcription.", 500)

        res = self.model(path, chunk_length_s=30, generate_kwargs={"condition_on_prev_tokens": False})
        return {
            "model": model_id,
            "text": res.get("text", "").strip(),
            "total_duration": int((time.perf_counter() - start) * 1e9),
        }


def inject_json_instruction(messages: List[Dict[str, Any]], format_req: Union[str, Dict[str, Any], None]) -> bool:
    """Injects JSON constraints into system prompts if strict formatting is requested."""
    if not format_req:
        return False

    instruction = (
        "You are a strict data-generating API. You must respond ONLY with valid, parseable JSON. "
        "Ensure all JSON keys and string values are strictly enclosed in double quotes. "
        "Do not include any explanations, markdown formatting, or conversational text."
    )

    if isinstance(format_req, dict):
        schema_str = json.dumps(format_req, indent=2)
        instruction = (
            f"You are a strict data-generating API. You must respond ONLY with valid, parseable JSON "
            f"that adheres exactly to the following JSON Schema:\n{schema_str}\n"
            "Ensure all JSON keys and string values are strictly enclosed in double quotes. "
            "Do not include any explanations, markdown formatting, or conversational text."
        )

    if messages and messages[0].get("role") == "system":
        messages[0]["content"] += f"\n\n{instruction}"
    else:
        messages.insert(0, {"role": "system", "content": instruction})

    return True


async def _handle_keep_alive(keep_alive: Union[str, int, None], model: str, engine_instance: InferenceEngine) -> bool:
    """Updates the dynamic keep-alive timeout or unloads the model immediately."""
    if keep_alive is None:
        return False

    # Parse string formats like "5m", "1h", "-1"
    timeout_seconds = 0.0
    if isinstance(keep_alive, str):
        keep_alive = keep_alive.lower().strip()
        if keep_alive in ["0", "0s", "0m"]:
            timeout_seconds = 0.0
        elif keep_alive == "-1":
            timeout_seconds = -1.0  # Infinite
        elif keep_alive.endswith("h"):
            timeout_seconds = float(keep_alive[:-1]) * 3600
        elif keep_alive.endswith("m"):
            timeout_seconds = float(keep_alive[:-1]) * 60
        elif keep_alive.endswith("s"):
            timeout_seconds = float(keep_alive[:-1])
        else:
            try:
                timeout_seconds = float(keep_alive)
            except ValueError:
                timeout_seconds = float(IDLE_TIMEOUT_SECONDS)
    else:
        timeout_seconds = float(keep_alive)

    # If timeout is 0, rip it out of VRAM immediately
    if timeout_seconds == 0.0:
        if engine_instance.active_id == model:
            async with GPU_LOCK:
                engine_instance.unload()
        return True

    # Otherwise, update the engine's timeout clock
    with engine_instance._lock:
        engine_instance.keep_alive_timeout = timeout_seconds

    return False


def _extract_options(
    options: Optional[ModelOptions], logprobs: Optional[bool], top_logprobs: Optional[int]
) -> Dict[str, Any]:
    """Extracts and consolidates inference execution options."""
    options_dict = options.model_dump(exclude_unset=True) if options else {}
    if logprobs:
        options_dict["logprobs"] = True
        if top_logprobs:
            options_dict["top_logprobs"] = top_logprobs
    return options_dict


async def monitor_idle(eng: InferenceEngine) -> None:
    """Background task to automatically release VRAM based on inactivity."""
    while True:
        await asyncio.sleep(15)  # Check more frequently (every 15s instead of 60s)

        with eng._lock:
            # 1. Is a model loaded?
            if not eng.active_id:
                continue

            # 2. Is the GPU currently busy with a request?
            if eng.active_requests > 0:
                continue

            # 3. Is it set to never expire? (-1.0)
            if eng.keep_alive_timeout < 0:
                continue

            # 4. Has it exceeded the dynamic timeout?
            time_idle = time.time() - eng.last_interaction
            should_unload = time_idle > eng.keep_alive_timeout

        # Unload OUTSIDE the sync lock using the async GPU lock
        if should_unload:
            async with GPU_LOCK:
                eng.unload()


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Manages application lifespan, particularly background memory management."""
    task = asyncio.create_task(monitor_idle(engine))
    yield
    task.cancel()
    if engine.active_id:
        engine.unload()


manager = ModelManager()
engine = InferenceEngine(manager)
app = FastAPI(title="Cudara Transformers Native", lifespan=lifespan)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Captures structured AppErrors and maps them to JSON."""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handles malformed requests and maps to a 400 JSON error."""
    return JSONResponse(status_code=400, content={"error": f"Invalid request format: {str(exc)}"})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Captures HTTP exceptions seamlessly for standard JSON error formatting."""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global catch-all for unexpected internal server errors."""
    return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(exc)}"})


@app.middleware("http")
async def force_json_content_type(request: Request, call_next: Callable[[Request], Any]) -> Any:
    """Enforces application/json content type for compatibility with stateless CLIs."""
    if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            new_headers = [(k, v) for k, v in request.scope.get("headers", []) if k.lower() != b"content-type"]
            new_headers.append((b"content-type", b"application/json"))
            request.scope["headers"] = new_headers

    return await call_next(request)


@app.get("/api/tags", tags=["Models"], summary="List models", response_model=ListResponse)
async def list_models() -> ListResponse:
    """Fetch a list of models locally available and their configurations."""
    reg = manager.get_registry()
    allowed = manager.get_allowed()
    downloaded = []

    for m_id, item in reg.items():
        if item.status == ModelStatus.READY and item.local_path:
            model_path = Path(item.local_path)
            if model_path.exists():
                size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
                config = allowed.get(m_id)
                family = config.task if config else "unknown"
                mtime = datetime.datetime.fromtimestamp(
                    model_path.stat().st_mtime, tz=datetime.timezone.utc
                ).isoformat()

                downloaded.append(
                    ModelSummary(
                        name=m_id,
                        model=m_id,
                        modified_at=mtime,
                        size=size,
                        digest="sha256:" + base64.b16encode(m_id.encode()).decode().lower()[:64],
                        details=ModelDetails(
                            format="safetensors",
                            family=family,
                            families=[family],
                            parameter_size="unknown",
                            quantization_level="AWQ" if "AWQ" in m_id.upper() else "None",
                        ),
                    )
                )

    return ListResponse(models=downloaded)


@app.post("/api/show", tags=["Models"], summary="Show model details", response_model=ShowResponse)
async def show_model(req: ShowRequest) -> ShowResponse:
    """Displays detailed schema, architecture, and formatting configuration for a model."""
    allowed = manager.get_allowed().get(req.model)
    reg = manager.get_registry().get(req.model)

    if not allowed or not reg or not reg.local_path:
        raise AppError(f"Model '{req.model}' not found", 404)

    model_path = Path(reg.local_path)
    info = {}
    config_path = model_path / "config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            c = json.load(f)
            info = {
                f"{allowed.task}.attention.head_count": c.get("num_attention_heads"),
                f"{allowed.task}.context_length": c.get("max_position_embeddings"),
                "general.architecture": c.get("model_type"),
                "general.parameter_count": c.get("num_parameters", "unknown"),
            }

    caps = ["completion"]
    if allowed.task == "image-to-text":
        caps.append("vision")
    if allowed.task == "automatic-speech-recognition":
        caps.append("audio")

    mtime = datetime.datetime.fromtimestamp(model_path.stat().st_mtime, tz=datetime.timezone.utc).isoformat()

    return ShowResponse(
        parameters="temperature 0.7",
        license="See Hugging Face repository terms.",
        modified_at=mtime,
        details=ModelDetails(
            format="safetensors",
            family=allowed.task,
            families=[allowed.task],
            parameter_size="unknown",
            quantization_level="AWQ" if "AWQ" in req.model.upper() else "None",
        ),
        capabilities=caps,
        model_info=info,
        template=None,
    )


@app.post(
    "/api/generate",
    tags=["Generation"],
    summary="Generate a response",
    response_model=None,
    openapi_extra={"responses": {"200": {"description": "Generation responses."}}},
)
async def generate(req: GenerateRequest, bg: BackgroundTasks) -> Union[JSONResponse, StreamingResponse]:
    """Generates a text completion for the provided prompt parameters."""

    if await _handle_keep_alive(req.keep_alive, req.model, engine):
        return JSONResponse(
            content={
                "model": req.model,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "response": "",
                "done": True,
                "done_reason": "unload",
            }
        )

    options_dict = _extract_options(req.options, req.logprobs, req.top_logprobs)

    async with GPU_LOCK:
        if options_dict.get("is_audio") and req.images:
            audio_bytes = base64.b64decode(req.images[0])
            ext = "wav"

            if (
                audio_bytes.startswith(b"ID3")
                or audio_bytes.startswith(b"\xff\xfb")
                or audio_bytes.startswith(b"\xff\xf3")
            ):
                ext = "mp3"
            elif audio_bytes.startswith(b"OggS"):
                ext = "ogg"
            elif audio_bytes[4:8] == b"ftyp":
                ext = "m4a"

            tmp = TEMP_DIR / f"asr_{int(time.time())}.{ext}"
            with open(tmp, "wb") as f:
                f.write(audio_bytes)

            try:
                res = await asyncio.to_thread(engine.transcribe, req.model, str(tmp))
                return JSONResponse(content={"model": req.model, "response": res["text"], "done": True})
            finally:
                tmp.unlink(missing_ok=True)

        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})

        full_prompt = req.prompt or ""
        if req.suffix:
            full_prompt = f"<|fim_prefix|>{full_prompt}<|fim_suffix|>{req.suffix}<|fim_middle|>"

        if full_prompt:
            msg: Dict[str, Any] = {"role": "user", "content": full_prompt}
            if req.images:
                msg["images"] = req.images
            messages.append(msg)

        force_json = inject_json_instruction(messages, req.format)

        gen = await asyncio.to_thread(
            engine.chat,
            model_id=req.model,
            messages=messages,
            options=options_dict,
            stream=req.stream,
            is_chat=False,
            force_json=force_json,
        )

        if req.stream:
            return StreamingResponse(gen, media_type="application/x-ndjson")  # type: ignore
        return JSONResponse(content=gen)  # type: ignore


@app.post(
    "/api/chat",
    tags=["Generation"],
    summary="Generate a chat message",
    response_model=None,
    openapi_extra={"responses": {"200": {"description": "Chat response stream or payload."}}},
)
async def chat(req: ChatRequest) -> Union[JSONResponse, StreamingResponse]:
    """Appends conversational context and generates the next chat message."""

    if await _handle_keep_alive(req.keep_alive, req.model, engine):
        return JSONResponse(
            content={
                "model": req.model,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "unload",
            }
        )

    options_dict = _extract_options(req.options, req.logprobs, req.top_logprobs)

    async with GPU_LOCK:
        msgs = [m.model_dump(exclude_unset=True) for m in req.messages]
        force_json = inject_json_instruction(msgs, req.format)

        gen = await asyncio.to_thread(
            engine.chat,
            model_id=req.model,
            messages=msgs,
            options=options_dict,
            stream=req.stream,
            is_chat=True,
            force_json=force_json,
        )

        if req.stream:
            return StreamingResponse(gen, media_type="application/x-ndjson")  # type: ignore
        return JSONResponse(content=gen)  # type: ignore


@app.post("/api/embed", tags=["Generation"], summary="Generate embeddings", response_model=EmbedResponse)
@app.post("/api/embeddings", include_in_schema=False)
async def embed(req: EmbedRequest) -> JSONResponse:
    """Generates numerical vector arrays from structural text inputs."""
    if await _handle_keep_alive(req.keep_alive, req.model, engine):
        return JSONResponse(content={"model": req.model, "done": True})

    options_dict = _extract_options(req.options, False, None)

    async with GPU_LOCK:
        if options_dict.get("is_rerank") and isinstance(req.input, list) and len(req.input) > 1:
            res = await asyncio.to_thread(engine.rerank, req.model, req.input[0], req.input[1:])
            return JSONResponse(content=res)

        input_texts: List[str]
        if isinstance(req.input, str):
            input_texts = [req.input]
        else:
            input_texts = req.input

        res = await asyncio.to_thread(
            engine.embeddings, model_id=req.model, texts=input_texts, truncate=req.truncate, dimensions=req.dimensions
        )
        return JSONResponse(content=res)


@app.get("/api/version", tags=["System"])
def get_version() -> Dict[str, str]:
    """Returns the current API version for clients to align with."""
    return {"version": "0.1.44"}


@app.post("/api/pull", tags=["Models"], summary="Pull model", response_model=None)
async def pull_model(req: PullRequest, bg: BackgroundTasks) -> Union[Dict[str, str], StreamingResponse]:
    """Downloads model weights into the local registry."""
    allowed_models = manager.get_allowed()
    if req.model not in allowed_models:
        raise AppError(f"Model '{req.model}' not in allowed config list", 400)

    if not req.stream:
        bg.add_task(manager.download_task, req.model)
        return {"status": "success"}

    async def generate_pull_progress() -> AsyncIterator[str]:
        threading.Thread(target=manager.download_task, args=(req.model,), daemon=True).start()

        last_status = None
        while True:
            reg = manager.get_registry().get(req.model)
            if not reg:
                yield json.dumps({"status": "initializing..."}) + "\n"
            else:
                event = {
                    "status": reg.status.value,
                    "digest": "sha256:" + base64.b16encode(req.model.encode()).decode().lower()[:64],
                }

                if reg.status == ModelStatus.READY:
                    event["status"] = "success"
                    yield json.dumps(event) + "\n"
                    break

                if reg.status == ModelStatus.ERROR:
                    event["status"] = f"error: {reg.error_message}"
                    yield json.dumps(event) + "\n"
                    break

                if reg.status.value != last_status:
                    yield json.dumps(event) + "\n"
                    last_status = reg.status.value

            await asyncio.sleep(1)

    return StreamingResponse(generate_pull_progress(), media_type="application/x-ndjson")


@app.get("/api/ps", tags=["System"], summary="List running models", response_model=PsResponse)
async def list_running() -> PsResponse:
    """Retrieve memory footprints for actively loaded models."""
    if not engine.active_id:
        return PsResponse(models=[])

    model_id = engine.active_id
    reg = manager.get_registry().get(model_id)
    allowed = manager.get_allowed().get(model_id)

    size = 0
    if reg and reg.local_path:
        model_path = Path(reg.local_path)
        size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

    expires_delta = datetime.timedelta(seconds=IDLE_TIMEOUT_SECONDS)
    expiration = datetime.datetime.fromtimestamp(engine.last_interaction, tz=datetime.timezone.utc) + expires_delta

    vram_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    family = allowed.task if allowed else "unknown"

    running_model = Ps(
        name=model_id,
        model=model_id,
        size=size,
        digest="sha256:" + base64.b16encode(model_id.encode()).decode().lower()[:64],
        details=ModelDetails(
            format="safetensors",
            family=family,
            families=[family],
            parameter_size="unknown",
            quantization_level="AWQ" if "AWQ" in model_id.upper() else "None",
        ),
        expires_at=expiration.isoformat(),
        size_vram=vram_usage,
        context_length=8192,
    )

    return PsResponse(models=[running_model])


@app.delete("/api/delete", tags=["Models"], summary="Delete a model", status_code=200)
async def delete_model(req: DeleteRequest) -> JSONResponse:
    """Removes model weights and configuration from standard storage paths."""
    if engine.active_id == req.model:
        async with GPU_LOCK:
            engine.unload()

    try:
        manager.delete_model(req.model)
        return JSONResponse(status_code=200, content={"status": "success"})
    except Exception as e:
        raise AppError(f"Failed to delete model {req.model}: {str(e)}", 500)


@app.post("/api/create", tags=["Models"])
async def create_model_placeholder() -> JSONResponse:
    """Placeholder endpoint for building models from Modelfiles."""
    return JSONResponse(status_code=501, content={"error": "The /api/create endpoint is not yet implemented."})


@app.post("/api/copy", tags=["Models"])
async def copy_model_placeholder() -> JSONResponse:
    """Placeholder endpoint for aliasing an existing model."""
    return JSONResponse(status_code=501, content={"error": "The /api/copy endpoint is not yet implemented."})


@app.post("/api/push", tags=["Models"])
async def push_model_placeholder() -> JSONResponse:
    """Placeholder endpoint for uploading a local model to a remote registry."""
    return JSONResponse(status_code=501, content={"error": "The /api/push endpoint is not yet implemented."})
