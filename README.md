
<div align="center">
<h2><a href="https://arxiv.org/abs/2505.14679" style="color:#68edcb">UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models</a></h2>
        If our project helps you, please give us a star ⭐ on GitHub to support us. 😉😉
        
[![arXiv](https://img.shields.io/badge/arXiv-2505.14679-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2505.14679) 
</div>

## 🔥 News
* **`2025.05`** 🌟 We released our paper *UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models* — 📖 [UltraEdit on arXiv](https://arxiv.org/abs/2505.14679) | 🤗 [UltraEditBench on HuggingFace](https://huggingface.co/datasets/XiaojieGu/UltraEditBench).







## 📦 Data & Model Preparation

1️⃣ Download the files from [Google Drive](https://drive.google.com/drive/folders/1wsxG5Ybf6hT9QUlccvzTuJSfL_TFNyKQ?usp=sharing) and place them under `UltraEdit/data/raw`.

2️⃣ Download the [UltraEditBench](https://huggingface.co/datasets/XiaojieGu/UltraEditBench) and save it under `UltraEdit/data/raw/ultraeditbench`.

3️⃣ Specify the path to model weights by setting the `name_or_path` field in `UltraEdit/config/model/model.yaml`.

If you need to use locate-then-edit methods, we provide precomputed covariance matrices on Hugging Face for several models: [GPT-J 6B](https://huggingface.co/XiaojieGu/gpt-j-6b_CovarianceMatrix), [Qwen2.5-7B-Instruct](https://huggingface.co/XiaojieGu/Qwen2.5-7B-Instruct_CovarianceMatrix), [Mistral-7B-v0.3](https://huggingface.co/XiaojieGu/Mistral-7B-v0.3_CovarianceMatrix), [LLaMA-3-8B-Instruct](https://huggingface.co/XiaojieGu/Llama-3-8B-Instruct_CovarianceMatrix), and [LLaMA-2-7B-hf](https://huggingface.co/XiaojieGu/Llama-2-7b-hf_CovarianceMatrix). 

## 🚀 Setup

Create the environment and install dependencies:

```bash
conda create -n ultraedit python=3.10
conda activate ultraedit
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
💡 If you want to try editing a Mistral-7B model, even a **24GB consumer GPU** is enough — model editing for everyone!

## 🧪 Run

Run the main experiment with:

```bash
sh run.sh
```

The `run.sh` script includes a sample command like:

```
python main.py dataset=zsre model=mistral-7b editor=ultraedit num_seq=200 \ # Number of turns
    editor.cache_dir=cache \
    dataset.batch_size=10 \
    dataset.n_edits=100 \ # Number of edits per turn
    model.edit_modules="[model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]"
```
💡 Just try editing **20K samples** on Mistral-7B in **under 5 minutes** — ultra-efficient!



## 🙏 Acknowledgements

Our work builds upon several excellent model editing frameworks. We sincerely thank the authors of [EasyEdit](https://github.com/zjunlp/EasyEdit/tree/main), [MALMEN](https://github.com/ChenmienTan/malmen), [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit), and [RLEdit](https://github.com/zhrli324/RLEdit) for their valuable contributions to the field.



  


## 🌟 Star History

![Star History Chart](https://api.star-history.com/svg?repos=XiaojieGu/UltraEdit&type=Date&width=600&height=300&cache_bust=1)




## 📫 Contact

For any inquiries or possible collaboration, feel free to reach out at **peettherapynoys@gmail.com** — we’re open to connecting!


## 📑 Citation
If you find UltraEdit useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{gu2025ultraedit,
  title={UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models},
  author={Gu, Xiaojie and Chen, Guangxu and Li, Jungang and Gu, Jia-Chen and Hu, Xuming and Zhang, Kai},
  journal={arXiv preprint arXiv:2505.14679},
  year={2025}
}
```
