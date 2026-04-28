# src/model.py

Esta hecho apra que coja el prompt en un string, se lo mande al modelo y devuelva 25 waypoints en una lista de Python con formato `(x, y)`


```python
from src.model import get_client

client = get_client("claude")
waypoints = client.predict(prompt)
# waypoints -> [(x0,y0), (x1,y1), ..., (x24,y24)]  # 25 tuples, 5 seconds at 5Hz
```





## Setup para claude chat y Ollama

### Claude 
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...   # from console.anthropic.com
```
```python
client = get_client("claude")                        # claude-opus-4-7 by default
client = get_client("claude", model="claude-sonnet-4-6")  # cheaper/faster
```

### OpenAI 
```bash
pip install openai
export OPENAI_API_KEY=sk-...          # from platform.openai.com
```
```python
client = get_client("openai")                # gpt-4o by default
client = get_client("openai", model="gpt-4o-mini")
```

### Ollama (local)
```bash
# 1. Install Ollama: https://ollama.com
# 2. Pull a model (once):
ollama pull gemma3:12b
# 3. No API key needed
pip install openai   # Ollama reuses the openai SDK
```
```python
client = get_client("ollama")                     # gemma3:12b by default
client = get_client("ollama", model="llama3.2")   # any pulled model
```






## Como integrarlo

Habia pensado que con los 4 módulos que digimos `model.py` estaria en el medio.

```
[1] Trajectory      [3] Prompt          [2] Model API       [4] Metric
    checker    -->      builder     -->      (this file)  -->   evaluation
    
    Reads past         Formats input          Calls LLM         Compares
    waypoints +        into a prompt          returns 25        predicted vs
    instruction        string                 waypoints         ground truth
```


Para pasar el prompt directamente desde [3] lo haces con esto:


```python
from src.model import get_client

client = get_client("ollama")   # or "claude", "openai"

prompt = build_prompt(past_waypoints, instruction, image_path)  # la funcion de la parte 3
predicted_waypoints = client.predict(prompt)
```

### Para sacar la salida al [4]

`predict()` devuelveuna lista de tuples — no necesitaz extra parsing

```python
predicted = client.predict(prompt)
# predicted = [(x0,y0), ..., (x24,y24)]

score = evaluate(predicted, ground_truth_waypoints)  # part 4's function
```

---

## Para cambiar el modelo sin tocar código

El cliente se crea una vez y se para, para cmabiar el provider:


```python
PROVIDER = "ollama"   # change to "claude" or "openai" to switch
MODEL    = "gemma3:12b"

client = get_client(PROVIDER, model=MODEL)
```

---

## Lo que debería devoler el modelo

El output del modelo se analiza solo, puede manejar:

- JSON list of pairs: `[[1.2, 3.4], [5.6, 7.8], ...]`
- JSON list of dicts: `[{"x": 1.2, "y": 3.4}, ...]`
- JSON object: `{"waypoints": [[...], ...]}`
- Numeros planos en texto y saca los primeros 50 numeros y los empareja

Diria que el prompt hecho por la parte 3 pregunte al modelo que devuelva "25 `[x, y]` pairs as JSON."

Si el modelo devuelve menos de 25 waypoints habrá un ValueError creo

