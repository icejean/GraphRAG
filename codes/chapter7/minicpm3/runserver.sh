# conda activate pytorch
cd /home/demo/minicpm3
vllm serve openbmb/MiniCPM3-4B \
     --api-key your_key \
     --port 6000 \
     --dtype half \
     --max_model_len 8192 \
     --trust-remote-code \
     --enable-auto-tool-choice \
     --tool-parser-plugin minicpm_tool_parser.py \
     --tool-call-parser minicpm \
     --chat-template minicpm_chat_template_with_tool.jinja \
     --gpu-memory-utilization 0.85

