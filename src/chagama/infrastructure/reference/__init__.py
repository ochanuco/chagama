"""Reading a reference picture into tags: the Anima Tagger and the d1 verifier.

Both run on the worker's GPU inside ComfyUI's Python and import torch only when
a model is loaded, so the client environment can import this package.
"""
