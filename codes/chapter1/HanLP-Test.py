# 1、加载HanLP
import hanlp
import os, time

# 2、单任务模型， 分词。
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
toks = tok(['商品和服务。', '阿婆主来到北京立方庭参观自然语义科技公司。'])
print(toks)
