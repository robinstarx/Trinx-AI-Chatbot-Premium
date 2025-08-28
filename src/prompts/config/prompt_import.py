import yaml

# ── PROMPTS ────
path = "src/prompts/"
with open(path + "router_prompt.yaml", "r", encoding="utf-8") as f:
    router_prompt = yaml.safe_load(f)

with open(path + "fetch_price_node_prompt.yaml", "r", encoding="utf-8") as f:
    fetch_price_prompt = yaml.safe_load(f)

with open(path + "trinity_info_node.yaml", "r", encoding="utf-8") as f:
    trinity_info_prompt = yaml.safe_load(f)

with open(path + "answer_node_prompt.yaml", "r", encoding="utf-8") as f:
    answer_prompt = yaml.safe_load(f)