# KITScenes

Code for the [KITScenes LongTail Challenge](https://huggingface.co/spaces/KIT-MRT/KITScenes-LongTail-Challenge).

**Task:** Given a front-view image/video, past trajectory (21 waypoints at 5Hz), and a natural language driving instruction (e.g. "turn left", "overtake truck on the right"), predict the next 25 waypoints (5 seconds at 5Hz).

## Dataset

Request access at [KIT-MRT/KITScenes-LongTail](https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail), then:

```bash
pip install huggingface_hub
huggingface-cli login
python scripts/download_data.py --splits train  # ~245 MB
python scripts/download_data.py --splits test   # ~34 GB
```

## Team

- Martín Santiago Soto
- Javier Borau
- Raul Fernandez Matellan
- Diego Caballero García-Alcaide
