# 设置科学上网
import os
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
from openai.resources import api_key
os.environ['GRAPHRAG_API_KEY']=api_key.key

from pathlib import Path
from graphrag.cli.query import run_local_search

answer =  run_local_search(
    config_filepath=None,
    data_dir=None,
    root_dir=Path("/home/ubuntu/dataset/test"),
    community_level=2,
    response_type="Multiple Paragraphs",
    streaming=False,
    query="《悟空传》中孙悟空跟女妖之间有什么故事？"
)

print(answer[0])
