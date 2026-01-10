"""
Cudara Inference Server.
========================

A lightweight, CUDA-accelerated inference server designed for Linux/NVIDIA environments.
It provides an Ollama-compatible API for GGUF (via llama.cpp) and Transformers models.

Key Features:
- STRICT Single-Model Concurrency: Only one model is loaded on the GPU at a time.
- Sequential Processing: Requests are serialized via an async lock.
- Automatic Cleanup: Models unload after 5 minutes of inactivity to free VRAM.
"""

import asyncio
import base64
import datetime
import fnmatch
import gc
import json
import logging
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# Backends
import torch

# FastAPI & Tools
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download, list_repo_files, login, snapshot_download
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification

# GGUF Backend (Soft Import)
Llama = None
Llava15ChatHandler = None

try:
    from llama_cpp import Llama

    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
    except ImportError:
        pass
except ImportError:
    pass

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cudara")

# Optimize for NVIDIA Ampere+
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

MODELS_DIR = Path("models")
REGISTRY_FILE = Path("registry.json")
ALLOWED_MODELS_FILE = Path("models.json")
TEMP_DIR = Path("temp_uploads")

MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Global Lock
GPU_LOCK = asyncio.Lock()
IDLE_TIMEOUT_SECONDS = 300  # 5 minutes


# =============================================================================
# DATA MODELS
# =============================================================================
class AppError(Exception):
    def __init__(self, message: str, status_code: int = 500, code: str = "error"):
        self.message = message
        self.status_code = status_code
        self.code = code


class ModelStatus(str, Enum):
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


class ModelConfig(BaseModel):
    description: Optional[str] = None
    task: str = "text-generation"
    backend: str = "llama-cpp"
    filename: Optional[str] = None
    projector_filename: Optional[str] = None
    system_prompt: Optional[str] = None
    generation_defaults: Dict[str, Any] = {}


class RegistryItem(BaseModel):
    status: ModelStatus
    local_path: Optional[str] = None
    projector_path: Optional[str] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = {}


class ModelIdentifier(BaseModel):
    name: str


class PullRequest(BaseModel):
    name: str
    stream: bool = False


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    images: Optional[List[str]] = None
    stream: bool = False
    options: Dict[str, Any] = {}


class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    options: Dict[str, Any] = {}


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    options: Dict[str, Any] = {}


# =============================================================================
# MODEL MANAGER
# =============================================================================
class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        if not REGISTRY_FILE.exists():
            self._save_json(REGISTRY_FILE, {})
        if not ALLOWED_MODELS_FILE.exists():
            self._save_json(ALLOWED_MODELS_FILE, {})

    def _save_json(self, path: Path, data: Any) -> None:
        with self._lock:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with self._lock:
            if not path.exists():
                return {}
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}

    def get_allowed(self) -> Dict[str, ModelConfig]:
        data = self._load_json(ALLOWED_MODELS_FILE)
        return {k: ModelConfig(**v) for k, v in data.items() if not k.startswith("_")}

    def get_registry(self) -> Dict[str, RegistryItem]:
        data = self._load_json(REGISTRY_FILE)
        return {k: RegistryItem(**v) for k, v in data.items()}

    def update_registry(self, model_id: str, **kwargs: Any) -> None:
        reg = self._load_json(REGISTRY_FILE)
        entry = reg.get(model_id, {})
        entry.update(kwargs)
        reg[model_id] = entry
        self._save_json(REGISTRY_FILE, reg)

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        config = self.get_allowed().get(model_id)
        reg = self.get_registry().get(model_id)
        if not config:
            return None
        return {
            "modelfile": f"# {model_id}\n# {config.description}",
            "parameters": json.dumps(config.generation_defaults),
            "template": config.system_prompt or "",
            "details": {
                "format": "gguf" if config.backend != "transformers" else "transformers",
                "family": config.task,
                "quantization_level": reg.details.get("quantization", "unknown")
                if reg
                else "unknown",
            },
            "model_info": config.model_dump(),
        }

    def download_task(self, model_id: str) -> None:
        config = self.get_allowed().get(model_id)
        if not config:
            return

        try:
            logger.info(f"⬇️ Starting download: {model_id}")
            if config.backend == "transformers" or config.task in [
                "automatic-speech-recognition",
                "feature-extraction",
                "text-classification",
            ]:
                local_dir = MODELS_DIR / model_id.replace("/", "--")
                snapshot_download(repo_id=model_id, local_dir=local_dir)
                self.update_registry(model_id, status=ModelStatus.READY, local_path=str(local_dir))
            else:
                if not config.filename:
                    raise ValueError("GGUF models must specify 'filename'")
                files = list_repo_files(repo_id=model_id)
                matches = fnmatch.filter(files, config.filename)
                if not matches:
                    raise FileNotFoundError(f"No file matching {config.filename}")
                target_file = matches[0]

                path = hf_hub_download(repo_id=model_id, filename=target_file, local_dir=MODELS_DIR)

                proj_path = None
                if config.projector_filename:
                    p_matches = fnmatch.filter(files, config.projector_filename)
                    if p_matches:
                        proj_path = hf_hub_download(
                            repo_id=model_id, filename=p_matches[0], local_dir=MODELS_DIR
                        )

                quant = target_file.split(".")[-2] if "Q" in target_file else "unknown"
                self.update_registry(
                    model_id,
                    status=ModelStatus.READY,
                    local_path=path,
                    projector_path=proj_path,
                    details={"quantization": quant},
                )

            logger.info(f"✅ Download complete: {model_id}")

        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            self.update_registry(model_id, status=ModelStatus.ERROR, error_message=str(e))

    def delete_model(self, model_id: str) -> None:
        reg = self.get_registry().get(model_id)
        if reg and reg.local_path:
            p = Path(reg.local_path)
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            if reg.projector_path:
                Path(reg.projector_path).unlink(missing_ok=True)

        full_reg = self._load_json(REGISTRY_FILE)
        if model_id in full_reg:
            del full_reg[model_id]
            self._save_json(REGISTRY_FILE, full_reg)


# =============================================================================
# INFERENCE ENGINE
# =============================================================================
class InferenceEngine:
    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager
        self.active_id: Optional[str] = None
        self.model_instance: Any = None
        self._lock = threading.Lock()
        self.last_interaction: float = 0.0

    def touch(self) -> None:
        """Update the last interaction timestamp to prevent idle cleanup."""
        self.last_interaction = time.time()

    def _unload(self) -> None:
        if self.model_instance:
            logger.info(f"♻️ Unloading {self.active_id}...")
            del self.model_instance
            self.model_instance = None
            self.active_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def load(self, model_id: str) -> None:
        # Implicitly touch when checking/loading
        self.touch()

        if self.active_id == model_id:
            return

        config = self.manager.get_allowed().get(model_id)
        reg = self.manager.get_registry().get(model_id)

        if not config or not reg or reg.status != ModelStatus.READY:
            raise AppError(f"Model {model_id} not ready", 404, "model_not_ready")

        with self._lock:
            if self.active_id == model_id:
                return
            self._unload()

            logger.info(f"🚀 Loading {model_id} ({config.backend})...")
            start = time.perf_counter()

            if config.backend == "transformers" or config.task in [
                "automatic-speech-recognition",
                "feature-extraction",
                "text-classification",
            ]:
                self._load_transformers(config, reg.local_path)
            else:
                self._load_gguf(config, reg.local_path, reg.projector_path)

            self.active_id = model_id
            logger.info(f"✅ Loaded in {time.perf_counter() - start:.2f}s")

    def _load_transformers(self, config: ModelConfig, path: str) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.task == "automatic-speech-recognition":
            self.model_instance = pipeline(
                "automatic-speech-recognition",
                model=path,
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        elif config.task == "feature-extraction":
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device)
            self.model_instance = (model, tokenizer)
        elif config.task == "text-classification":
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(
                path, trust_remote_code=True
            ).to(device)
            self.model_instance = (model, tokenizer)
        else:
            raise AppError(f"Task {config.task} not supported", 500)

    def _load_gguf(self, config: ModelConfig, path: str, projector_path: Optional[str]) -> None:
        if not Llama:
            raise AppError("llama-cpp-python missing", 500)

        params = config.generation_defaults.copy()
        n_gpu = params.pop("n_gpu_layers", -1)
        n_ctx = params.pop("n_ctx", 4096)

        # Auto-detect context window if 0
        if n_ctx == 0:
            n_ctx = 0

        chat_handler = None
        if config.task == "image-to-text" and projector_path:
            if Llava15ChatHandler:
                chat_handler = Llava15ChatHandler(clip_model_path=projector_path)

        self.model_instance = Llama(
            model_path=path,
            n_gpu_layers=n_gpu,
            n_ctx=n_ctx,
            chat_handler=chat_handler,
            verbose=False,
        )

    # --- Methods (Updated to call self.touch()) ---

    def chat(self, model_id: str, messages: List[Dict], options: Dict[str, Any]) -> Dict[str, Any]:
        self.load(model_id)
        if not isinstance(self.model_instance, Llama):
            raise AppError("Chat supported only for GGUF", 400)

        # Prepare messages
        fmt_msgs = []
        for m in messages:
            if m.get("images"):
                content = [{"type": "text", "text": m["content"]}]
                for img in m["images"]:
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                    )
                fmt_msgs.append({"role": m["role"], "content": content})
            else:
                fmt_msgs.append({"role": m["role"], "content": m["content"]})

        defaults = self.manager.get_allowed()[model_id].generation_defaults or {}
        kwargs = {**defaults, **options}
        for k in ["n_gpu_layers", "n_ctx", "n_batch"]:
            kwargs.pop(k, None)

        res = self.model_instance.create_chat_completion(messages=fmt_msgs, **kwargs)
        duration = int(
            (time.perf_counter() - self.last_interaction) * 1e9
        )  # Approx since load calls touch

        return {
            "model": model_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "message": res["choices"][0]["message"],
            "done": True,
            "total_duration": duration,
            "eval_count": res["usage"]["completion_tokens"],
        }

    def embeddings(
        self, model_id: str, texts: List[str], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.load(model_id)
        start = time.perf_counter()

        if isinstance(self.model_instance, Llama):
            raise AppError("Use transformers for embeddings", 400)

        model, tokenizer = self.model_instance
        device = model.device

        sep = options.get("pair_sep", "\u241e")
        if any(sep in t for t in texts):
            pairs = [t.split(sep, 1) for t in texts]
            inputs = tokenizer(
                pairs, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                scores = model(**inputs).logits.view(-1).float().cpu().tolist()
            output = [[s] for s in scores]
        else:
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                if hasattr(model, "last_hidden_state"):
                    output = model(**inputs).last_hidden_state.mean(dim=1).cpu().tolist()
                elif hasattr(model, "base_model"):
                    output = model.base_model(**inputs).last_hidden_state.mean(dim=1).cpu().tolist()
                else:
                    output = model(**inputs)[0].mean(dim=1).cpu().tolist()

        return {
            "model": model_id,
            "embeddings": output,
            "total_duration": int((time.perf_counter() - start) * 1e9),
        }

    def transcribe(self, model_id: str, file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        self.load(model_id)
        start = time.perf_counter()
        res = self.model_instance(file_path, **options)
        return {
            "model": model_id,
            "text": res.get("text", "").strip(),
            "total_duration": int((time.perf_counter() - start) * 1e9),
        }


# =============================================================================
# APP LIFECYCLE & MONITORING
# =============================================================================
async def monitor_idle_models(engine_ref: InferenceEngine):
    """Background task to unload models after inactivity."""
    logger.info("🕒 Idle monitor started (Timeout: 5min)")
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            if engine_ref.active_id:
                idle_time = time.time() - engine_ref.last_interaction
                if idle_time > IDLE_TIMEOUT_SECONDS:
                    logger.info(
                        f"⏱️ Idle timeout ({idle_time:.0f}s > {IDLE_TIMEOUT_SECONDS}s). Unloading..."
                    )
                    async with GPU_LOCK:
                        engine_ref._unload()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    monitor_task = asyncio.create_task(monitor_idle_models(engine))
    yield
    # Shutdown
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    if engine.active_id:
        engine._unload()


manager = ModelManager()
engine = InferenceEngine(manager)
app = FastAPI(
    title="Cudara API",
    version="1.0.0",
    description="Sequential GPU Inference Server",
    lifespan=lifespan,
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}}
    )


# --- Routes (Same as before) ---


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "active_model": engine.active_id,
        "cuda_available": torch.cuda.is_available(),
        "vram_used": f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
        if torch.cuda.is_available()
        else "0GB",
    }


@app.get("/api/tags", tags=["Models"])
def list_models() -> Dict[str, List[Dict]]:
    allowed = manager.get_allowed()
    reg = manager.get_registry()
    models_list = []
    for k, v in allowed.items():
        r = reg.get(k)
        models_list.append(
            {
                "name": k,
                "status": r.status.value if r else "not_downloaded",
                "details": r.details if r else {"format": "gguf"},
                "description": v.description,
            }
        )
    return {"models": models_list}


@app.post("/api/show", tags=["Models"])
def show_model(req: ModelIdentifier) -> Dict[str, Any]:
    info = manager.get_model_info(req.name)
    if not info:
        raise AppError("Model not found", 404, "model_not_found")
    return info


@app.post("/api/pull", tags=["Models"])
def pull_model(req: PullRequest, bg: BackgroundTasks) -> Dict[str, str]:
    if req.name not in manager.get_allowed():
        raise AppError("Model not allowed", 403, "model_not_allowed")
    manager.update_registry(req.name, status=ModelStatus.DOWNLOADING)
    bg.add_task(manager.download_task, req.name)
    return {"status": "downloading"}


@app.delete("/api/delete", tags=["Models"])
def delete_model(req: ModelIdentifier) -> Dict[str, str]:
    if engine.active_id == req.name:
        engine._unload()
    manager.delete_model(req.name)
    return {"status": "deleted"}


@app.post("/api/generate", tags=["Inference"])
async def generate(req: GenerateRequest) -> Dict[str, Any]:
    async with GPU_LOCK:
        msg = {"role": "user", "content": req.prompt}
        if req.images:
            msg["images"] = req.images
        return engine.chat(req.model, [msg], req.options)


@app.post("/api/chat", tags=["Inference"])
async def chat(req: ChatRequest) -> Dict[str, Any]:
    async with GPU_LOCK:
        return engine.chat(req.model, [m.model_dump() for m in req.messages], req.options)


@app.post("/api/embeddings", tags=["Inference"])
@app.post("/api/embed", tags=["Inference"], include_in_schema=False)
async def embeddings(req: EmbeddingRequest) -> Dict[str, Any]:
    async with GPU_LOCK:
        inp = [req.input] if isinstance(req.input, str) else req.input
        return engine.embeddings(req.model, inp, req.options)


@app.post("/api/transcribe", tags=["Inference"])
async def transcribe(
    model: str = Form(...), file: UploadFile = File(...), options: str = Form("{}")
) -> Dict[str, Any]:
    try:
        opt = json.loads(options)
    except Exception:
        opt = {}
    temp_path = TEMP_DIR / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        async with GPU_LOCK:
            return engine.transcribe(model, str(temp_path), opt)
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/vision", tags=["Inference"])
async def vision(
    model: str = Form(...),
    prompt: str = Form(...),
    file: UploadFile = File(...),
    options: str = Form("{}"),
) -> Dict[str, Any]:
    try:
        opt = json.loads(options)
    except Exception:
        opt = {}
    content = await file.read()
    b64 = base64.b64encode(content).decode("utf-8")
    async with GPU_LOCK:
        res = engine.chat(model, [{"role": "user", "content": prompt, "images": [b64]}], opt)
        return {
            "model": res["model"],
            "created_at": res["created_at"],
            "response": res["message"]["content"],
            "done": True,
            "total_duration": res["total_duration"],
            "eval_count": res["eval_count"],
        }


# Legacy
@app.get("/available-models", tags=["Models"], deprecated=True)
def list_available_legacy() -> Dict[str, Any]:
    return {k: v.model_dump() for k, v in manager.get_allowed().items()}


@app.get("/models", tags=["Models"], deprecated=True)
def list_downloaded_legacy() -> Dict[str, Any]:
    return {k: v.model_dump() for k, v in manager.get_registry().items()}
