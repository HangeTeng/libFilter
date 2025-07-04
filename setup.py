# setup.py

import setuptools

# setup() 将会从 pyproject.toml 文件中自动读取大部分元数据。
# 这个文件主要用于兼容一些仍然需要 setup.py 的旧工具。
# 对于现代的 Python 打包流程，这个文件甚至可以是空的，
# 但保留它可以增加兼容性。

setuptools.setup()