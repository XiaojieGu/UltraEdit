
<div align="center">
<h2><a href="https://arxiv.org/abs/2505.14679" style="color:#68edcb">UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models</a></h2>
        If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏
        
[![arXiv](https://img.shields.io/badge/arXiv-2505.14679-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2505.14679) 
</div>

## 🔥 News
* **`2025.05`** 🌟 We released the paper [UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models](https://arxiv.org/abs/2505.14679).



## 🚀 Setup

Create the environment and install dependencies:

```bash
conda create -n ultraedit python=3.10 -y
conda activate ultraedit
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 🧪 Run

Run the main experiment with:

```bash
sh run.sh
```

The `run.sh` script includes a sample command like:

```
python main.py dataset=zsre model=mistral-7b editor=ultraedit num_seq=200 \
    editor.cache_dir=cache \
    dataset.batch_size=10 \
    dataset.n_edits=100 \
    model.edit_modules="[model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]"

```


## 📦 Data Preparation

1️⃣ Download the files from [Google Drive](https://drive.google.com/drive/folders/1wsxG5Ybf6hT9QUlccvzTuJSfL_TFNyKQ?usp=sharing) and place them under `UltraEdit/data/raw`.

2️⃣ Download the [UltraEditBench dataset from Hugging Face](https://huggingface.co/datasets/XiaojieGu/UltraEditBench) and save it under `UltraEdit/data/raw/ultraeditbench` with the following script:

```python
from datasets import load_dataset
import os, json

dataset = load_dataset("XiaojieGu/UltraEditBench", split="train")
os.makedirs("data/raw/ultraeditbench", exist_ok=True)

with open("data/raw/ultraeditbench/UltraEditBench_2M.json", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
```





## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=XiaojieGu/UltraEdit&type=Date&width=600&height=300)](https://star-history.com/#XiaojieGu/UltraEdit&Date)


## 📑 Citation
If you find UltraEdit useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{Gu2025UltraEdit,
      title={UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models}, 
      author={Xiaojie Gu and Guangxu Chen and Jungang Li and Jia-Chen Gu and Xuming Hu and Kai Zhang},
      year={2025},
      eprint={2505.14679},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14679}, 
}
```
