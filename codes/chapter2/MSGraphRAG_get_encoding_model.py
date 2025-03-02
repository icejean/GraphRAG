import tiktoken_ext.openai_public
import inspect
import hashlib

print(dir(tiktoken_ext.openai_public))
# The encoder we want is o200k_base, we see this as a possible function
# print(inspect.getsource(tiktoken_ext.openai_public.gpt2))
# The URL should be in the 'load_tiktoken_bpe function call'
print(inspect.getsource(tiktoken_ext.openai_public.o200k_base))
blobpath = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
print(cache_key)
# fb374d419588a4632f3f557e76b4b70aebbca790
