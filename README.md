# KITScenes

Code for the [KITScenes LongTail Challenge](https://huggingface.co/spaces/KIT-MRT/KITScenes-LongTail-Challenge).

**Task:** Given a front-view image/video, past trajectory (21 waypoints at 5Hz), and a natural language driving instruction (e.g. "turn left", "overtake truck on the right"), predict the next 25 waypoints (5 seconds at 5Hz).

## Dataset

Request access at [KIT-MRT/KITScenes-LongTail](https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail), then:

```bash
pip install -r requirements.txt
huggingface-cli login
python scripts/download_data.py --splits train  # ~245 MB
python scripts/download_data.py --splits test   # ~34 GB
```

Data lands in `data/`. The train split has **3 scenarios**; the test split has **400 scenarios**.

## Prompt generation (`src/prompts.py` + `src/dataset.py`)

These modules convert dataset rows into prompts ready to send to any VLM API.

### Loading instances

```python
from src.dataset import load_instance, iter_instances

# single row by index
inst = load_instance("train", 0)   # dict with all fields
inst = load_instance("test",  42)

# iterate without loading everything into memory
for inst in iter_instances("train"):
    ...

# limit how many rows to load
for inst in iter_instances("test", limit=10):
    ...
```

Each `inst` is a plain Python dict with these fields:

| Field | Type | Description |
|---|---|---|
| `scenario_id` | `str` | Unique scenario identifier |
| `frames_camera_front` | `list[{bytes, path}]` | 21 JPEG frames, front camera |
| `frames_camera_{front_left,front_right,rear,rear_left,rear_right}` | same | Other cameras |
| `driving_instruction` | `str` | e.g. `"turn left"` |
| `scenario_type` | `str` | e.g. `"4 intersection"` |
| `trajectory.past` | `list[[x,y]]` × 21 | Past waypoints (ego frame, metres) |
| `trajectory.expert_like` | `list[[x,y]]` × 25 | Ground-truth future *(train only)* |
| `reasoning.{english,spanish,chinese}` | dict | Chain-of-thought annotation *(train only)* |

### Building prompts

```python
from src.prompts import build_prompt

prompt = build_prompt(inst)
```

`build_prompt` returns a `Prompt` dataclass:

```python
@dataclass
class Prompt:
    system: str                      # system message
    user_text: str                   # user text (instruction + past trajectory)
    images: list[ImagePart]          # images as raw bytes (JPEG)
    few_shot: list[FewShotExample]   # empty by default
    scenario_id: str | None
```

All parameters have sensible defaults and are keyword-only:

| Parameter | Default | Options |
|---|---|---|
| `cameras` | `("front",)` | any subset of the 6 cameras |
| `frame_strategy` | `"last"` | `"last"`, `"all"`, `"first_middle_last"` |
| `output_format` | `"cot_structured"` | `"cot_structured"`, `"waypoints_only"`, `"cot_free"` |
| `trajectory_encoding` | `"both"` | `"both"`, `"numeric"`, `"textual"` |
| `language` | `"en"` | `"en"`, `"es"`, `"zh"` (affects reasoning in few-shot) |
| `few_shot_examples` | `()` | list of train instance dicts |

### Few-shot with chain-of-thought

Pass train instances as `few_shot_examples`. Each one becomes a solved user/assistant pair using `trajectory.expert_like` and `reasoning` as the ground-truth answer.

```python
train_examples = list(iter_instances("train"))  # 3 scenarios

prompt = build_prompt(
    inst,
    few_shot_examples=train_examples,
    output_format="cot_structured",
)
# prompt.few_shot → list of FewShotExample(user_text, images, assistant_text)
# assistant_text is a JSON string with the 9 reasoning fields + future_waypoints
```

### Adapting to a specific API

`Prompt` is provider-agnostic — images are raw bytes. Each team member adds their own adapter. Example for **Anthropic**:

```python
import base64
import anthropic

def to_anthropic(prompt):
    def img_block(img):
        return {"type": "image", "source": {
            "type": "base64", "media_type": img.mime_type,
            "data": base64.b64encode(img.bytes).decode(),
        }}

    messages = []
    for ex in prompt.few_shot:
        messages.append({"role": "user",      "content": [img_block(i) for i in ex.images] + [{"type": "text", "text": ex.user_text}]})
        messages.append({"role": "assistant", "content": ex.assistant_text})

    messages.append({"role": "user", "content": [img_block(i) for i in prompt.images] + [{"type": "text", "text": prompt.user_text}]})
    return messages

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-opus-4-7",
    system=prompt.system,
    messages=to_anthropic(prompt),
    max_tokens=1024,
)
```

### Running tests

```bash
python -m pytest tests/ -v
```

Tests in `test_prompts.py` use a synthetic fixture and run without data. Tests in `test_dataset.py` hit real parquet files and are skipped automatically if `data/train/` is absent.

## Team

- Martín Santiago Soto
- Javier Borau
- Raul Fernandez Matellan
- Diego Caballero García-Alcaide
