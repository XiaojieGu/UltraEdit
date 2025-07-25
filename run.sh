######################################  llama3  ###################################################

# python main.py dataset=zsre model=llama-3-instruct editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.11.mlp.gate_proj, model.layers.12.mlp.gate_proj, model.layers.13.mlp.gate_proj, model.layers.14.mlp.gate_proj, model.layers.15.mlp.gate_proj, model.layers.18.mlp.up_proj, model.layers.19.mlp.up_proj, model.layers.20.mlp.up_proj, model.layers.21.mlp.up_proj, model.layers.22.mlp.up_proj, model.layers.23.mlp.up_proj, model.layers.24.mlp.up_proj]" \

# python main.py dataset=fever model=llama-3-instruct editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.22.mlp.gate_proj, model.layers.23.mlp.gate_proj, model.layers.24.mlp.gate_proj, model.layers.25.mlp.gate_proj, model.layers.26.mlp.gate_proj, model.layers.27.mlp.gate_proj, model.layers.28.mlp.gate_proj, model.layers.29.mlp.gate_proj, model.layers.30.mlp.gate_proj, model.layers.22.mlp.up_proj, model.layers.23.mlp.up_proj, model.layers.24.mlp.up_proj, model.layers.25.mlp.up_proj, model.layers.26.mlp.up_proj, model.layers.27.mlp.up_proj, model.layers.28.mlp.up_proj, model.layers.29.mlp.up_proj, model.layers.30.mlp.up_proj]" \


# python main.py dataset=wikibigedit model=llama-3-instruct editor=ultraedit num_seq=170 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.11.mlp.gate_proj, model.layers.12.mlp.gate_proj, model.layers.13.mlp.gate_proj, model.layers.14.mlp.gate_proj, model.layers.15.mlp.gate_proj, model.layers.18.mlp.up_proj, model.layers.19.mlp.up_proj, model.layers.20.mlp.up_proj, model.layers.21.mlp.up_proj, model.layers.22.mlp.up_proj, model.layers.23.mlp.up_proj, model.layers.24.mlp.up_proj]" \

# python main.py dataset=ultraeditbench model=llama-3-instruct editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.11.mlp.gate_proj, model.layers.12.mlp.gate_proj, model.layers.13.mlp.gate_proj, model.layers.14.mlp.gate_proj, model.layers.15.mlp.gate_proj, model.layers.18.mlp.up_proj, model.layers.19.mlp.up_proj, model.layers.20.mlp.up_proj, model.layers.21.mlp.up_proj, model.layers.22.mlp.up_proj, model.layers.23.mlp.up_proj, model.layers.24.mlp.up_proj]" \


######################################  mistral-7b  ###################################################


python main.py dataset=zsre model=mistral-7b editor=ultraedit num_seq=200 \
    editor.cache_dir=cache \
    dataset.batch_size=10 \
    dataset.n_edits=100 \
    model.edit_modules="[model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \

# python main.py dataset=fever model=mistral-7b editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \


# python main.py dataset=wikibigedit model=mistral-7b editor=ultraedit num_seq=170 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \

# python main.py dataset=ultraeditbench model=mistral-7b editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.29.mlp.down_proj, model.layers.30.mlp.down_proj]" \


######################################  gpt-j  ###################################################


# python main.py dataset=zsre model=gpt-j editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[transformer.h.18.mlp.fc_out, transformer.h.19.mlp.fc_out, transformer.h.20.mlp.fc_out, transformer.h.21.mlp.fc_out, transformer.h.22.mlp.fc_out, transformer.h.23.mlp.fc_out, transformer.h.24.mlp.fc_out, transformer.h.25.mlp.fc_out, transformer.h.26.mlp.fc_out]" \

# python main.py dataset=fever model=gpt-j editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[transformer.h.25.mlp.fc_out, transformer.h.26.mlp.fc_out]" \

# python main.py dataset=wikibigedit model=gpt-j editor=ultraedit num_seq=170 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[transformer.h.19.mlp.fc_out, transformer.h.20.mlp.fc_out, transformer.h.21.mlp.fc_out, transformer.h.22.mlp.fc_out, transformer.h.23.mlp.fc_out, transformer.h.24.mlp.fc_out, transformer.h.25.mlp.fc_out, transformer.h.26.mlp.fc_out]" \

# python main.py dataset=ultraeditbench model=gpt-j editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[transformer.h.18.mlp.fc_out, transformer.h.19.mlp.fc_out, transformer.h.20.mlp.fc_out, transformer.h.21.mlp.fc_out, transformer.h.22.mlp.fc_out, transformer.h.23.mlp.fc_out, transformer.h.24.mlp.fc_out, transformer.h.25.mlp.fc_out, transformer.h.26.mlp.fc_out]" \


######################################  qwen2.5-7b  ###################################################


# python main.py dataset=zsre model=qwen2.5-7b editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.18.mlp.up_proj, model.layers.19.mlp.up_proj, model.layers.20.mlp.up_proj, model.layers.21.mlp.up_proj, model.layers.22.mlp.up_proj, model.layers.23.mlp.up_proj, model.layers.24.mlp.up_proj, model.layers.25.mlp.up_proj, model.layers.26.mlp.up_proj, model.layers.18.mlp.gate_proj, model.layers.19.mlp.gate_proj, model.layers.20.mlp.gate_proj, model.layers.21.mlp.gate_proj, model.layers.22.mlp.gate_proj, model.layers.23.mlp.gate_proj, model.layers.24.mlp.gate_proj, model.layers.25.mlp.gate_proj, model.layers.26.mlp.gate_proj]" \

# python main.py dataset=fever model=qwen2.5-7b editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.18.mlp.up_proj, model.layers.19.mlp.up_proj, model.layers.20.mlp.up_proj, model.layers.21.mlp.up_proj, model.layers.22.mlp.up_proj, model.layers.23.mlp.up_proj, model.layers.24.mlp.up_proj, model.layers.25.mlp.up_proj, model.layers.26.mlp.up_proj, model.layers.18.mlp.gate_proj, model.layers.19.mlp.gate_proj, model.layers.20.mlp.gate_proj, model.layers.21.mlp.gate_proj, model.layers.22.mlp.gate_proj, model.layers.23.mlp.gate_proj, model.layers.24.mlp.gate_proj, model.layers.25.mlp.gate_proj, model.layers.26.mlp.gate_proj]" \

# python main.py dataset=wikibigedit model=qwen2.5-7b editor=ultraedit num_seq=17 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.19.mlp.up_proj, model.layers.20.mlp.up_proj, model.layers.21.mlp.up_proj, model.layers.22.mlp.up_proj, model.layers.23.mlp.up_proj, model.layers.24.mlp.up_proj, model.layers.25.mlp.up_proj, model.layers.26.mlp.up_proj, model.layers.19.mlp.gate_proj, model.layers.20.mlp.gate_proj, model.layers.21.mlp.gate_proj, model.layers.22.mlp.gate_proj, model.layers.23.mlp.gate_proj, model.layers.24.mlp.gate_proj, model.layers.25.mlp.gate_proj, model.layers.26.mlp.gate_proj]" \

# python main.py dataset=ultraeditbench model=qwen2.5-7b editor=ultraedit num_seq=200 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.18.mlp.up_proj, model.layers.19.mlp.up_proj, model.layers.20.mlp.up_proj, model.layers.21.mlp.up_proj, model.layers.22.mlp.up_proj, model.layers.23.mlp.up_proj, model.layers.24.mlp.up_proj, model.layers.25.mlp.up_proj, model.layers.26.mlp.up_proj, model.layers.18.mlp.gate_proj, model.layers.19.mlp.gate_proj, model.layers.20.mlp.gate_proj, model.layers.21.mlp.gate_proj, model.layers.22.mlp.gate_proj, model.layers.23.mlp.gate_proj, model.layers.24.mlp.gate_proj, model.layers.25.mlp.gate_proj, model.layers.26.mlp.gate_proj]" \


######################################  gemma-3-27b  ###################################################

# python main.py dataset=wikibigedit model=gemma-3-27b editor=ultraedit num_seq=5000 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     dataset.valid_path=./data/raw/wikibigedit/wikibigedit.json \
#     dataset.eval_mhop=False \
#     model.edit_modules="[language_model.model.layers.52.mlp.gate_proj, language_model.model.layers.53.mlp.gate_proj, language_model.model.layers.54.mlp.gate_proj, language_model.model.layers.55.mlp.gate_proj, language_model.model.layers.56.mlp.gate_proj, language_model.model.layers.57.mlp.gate_proj, language_model.model.layers.58.mlp.gate_proj, language_model.model.layers.59.mlp.gate_proj, language_model.model.layers.60.mlp.gate_proj, language_model.model.layers.52.mlp.up_proj, language_model.model.layers.53.mlp.up_proj, language_model.model.layers.54.mlp.up_proj, language_model.model.layers.55.mlp.up_proj, language_model.model.layers.56.mlp.up_proj, language_model.model.layers.57.mlp.up_proj, language_model.model.layers.58.mlp.up_proj, language_model.model.layers.59.mlp.up_proj, language_model.model.layers.60.mlp.up_proj]" \

# python main.py dataset=ultraeditbench model=gemma-3-27b editor=ultraedit num_seq=10000 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[language_model.model.layers.52.mlp.gate_proj, language_model.model.layers.53.mlp.gate_proj, language_model.model.layers.54.mlp.gate_proj, language_model.model.layers.55.mlp.gate_proj, language_model.model.layers.56.mlp.gate_proj, language_model.model.layers.57.mlp.gate_proj, language_model.model.layers.58.mlp.gate_proj, language_model.model.layers.59.mlp.gate_proj, language_model.model.layers.60.mlp.gate_proj, language_model.model.layers.52.mlp.up_proj, language_model.model.layers.53.mlp.up_proj, language_model.model.layers.54.mlp.up_proj, language_model.model.layers.55.mlp.up_proj, language_model.model.layers.56.mlp.up_proj, language_model.model.layers.57.mlp.up_proj, language_model.model.layers.58.mlp.up_proj, language_model.model.layers.59.mlp.up_proj, language_model.model.layers.60.mlp.up_proj]" \


######################################  phi-4  ###################################################

# python main.py dataset=wikibigedit model=phi-4 editor=ultraedit num_seq=5000 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     dataset.valid_path=./data/raw/wikibigedit/wikibigedit.json \
#     dataset.eval_mhop=False \
#     model.edit_modules="[model.layers.30.mlp.down_proj, model.layers.31.mlp.down_proj, model.layers.32.mlp.down_proj, model.layers.33.mlp.down_proj, model.layers.34.mlp.down_proj, model.layers.35.mlp.down_proj, model.layers.36.mlp.down_proj, model.layers.37.mlp.down_proj, model.layers.38.mlp.down_proj]" \

# python main.py dataset=ultraeditbench model=phi-4 editor=ultraedit num_seq=10000 \
#     editor.cache_dir=cache \
#     dataset.batch_size=10 \
#     dataset.n_edits=100 \
#     model.edit_modules="[model.layers.30.mlp.down_proj, model.layers.31.mlp.down_proj, model.layers.32.mlp.down_proj, model.layers.33.mlp.down_proj, model.layers.34.mlp.down_proj, model.layers.35.mlp.down_proj, model.layers.36.mlp.down_proj, model.layers.37.mlp.down_proj, model.layers.38.mlp.down_proj]" \





