"""Zero-shot prefix reranking with Qwen3-1.7B (vLLM, greedy, thinking off).

Usage: python run_prefix_llm.py PROMPTS.jsonl OUT_DIR
Writes raw_outputs.jsonl (one row per user, raw model text, untouched) and run_meta.json.
The whole prompt set is generated twice to check that greedy outputs repeat.
"""
import json, os, platform, subprocess, sys, time

t_start = time.time()
import torch, transformers, vllm
from vllm import LLM, SamplingParams

MODEL_DIR = '/home/mstarchenko/xai_x/models/qwen3-1.7b'
SETTINGS = {'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 512, 'seed': 0,
            'max_model_len': 8192, 'dtype': 'bfloat16', 'gpu_memory_utilization': 0.30,
            'enable_thinking': False, 'chat_template': 'model default (tokenizer_config.json)'}

prompts = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

t_load = time.time()
llm = LLM(model=MODEL_DIR, dtype=SETTINGS['dtype'], max_model_len=SETTINGS['max_model_len'],
          gpu_memory_utilization=SETTINGS['gpu_memory_utilization'], seed=SETTINGS['seed'])
load_s = time.time() - t_load
tok = llm.get_tokenizer()
texts = [tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
         for p in prompts]
sp = SamplingParams(temperature=SETTINGS['temperature'], top_p=SETTINGS['top_p'],
                    max_tokens=SETTINGS['max_tokens'], seed=SETTINGS['seed'])

runs = []
for rep in range(2):
    t = time.time()
    outs = llm.generate(texts, sp)
    runs.append((time.time() - t, outs))

gen_s, outs = runs[0]
with open(os.path.join(out_dir, 'raw_outputs.jsonl'), 'w', encoding='utf-8') as f:
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        f.write(json.dumps({'user_id': p['user_id'], 'protocol_hash': p['protocol_hash'],
                            'raw_text': c.text, 'finish_reason': c.finish_reason,
                            'prompt_tokens': len(o.prompt_token_ids),
                            'output_tokens': len(c.token_ids)}, ensure_ascii=False) + '\n')
changed = sum(a.outputs[0].text != b.outputs[0].text for a, b in zip(runs[0][1], runs[1][1]))
manifest = json.load(open(os.path.join(MODEL_DIR, 'download_manifest.json')))
smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
meta = {'model': 'Qwen/Qwen3-1.7B', 'revision': manifest['revision'],
        'resolved_sha': manifest['resolved_sha'], 'settings': SETTINGS,
        'versions': {'vllm': vllm.__version__, 'torch': torch.__version__,
                     'transformers': transformers.__version__, 'python': platform.python_version()},
        'hardware': {'nvidia_smi': smi, 'device': torch.cuda.get_device_name(0),
                     'note': 'A100-SXM4-80GB in MIG mode, one 3g.40gb slice, 30% of its memory'},
        'users': len(prompts),
        'prompt_tokens_total': sum(len(o.prompt_token_ids) for o in outs),
        'output_tokens_total': sum(len(o.outputs[0].token_ids) for o in outs),
        'model_load_seconds': round(load_s, 1),
        'generation_seconds_run1': round(runs[0][0], 2),
        'generation_seconds_run2': round(runs[1][0], 2),
        'outputs_changed_between_two_greedy_runs': changed,
        'total_script_seconds': round(time.time() - t_start, 1)}
json.dump(meta, open(os.path.join(out_dir, 'run_meta.json'), 'w'), indent=1)
print(json.dumps(meta, indent=1))
