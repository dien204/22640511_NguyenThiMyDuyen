# Google Colab flow

```python
!pip install pandas numpy scikit-learn joblib pyyaml streamlit fastapi uvicorn pydantic
```

```python
import os
project = "myduyen_TH2"
for p in ["data/raw", "models", "src", "config", "docs"]:
    os.makedirs(f"{project}/{p}", exist_ok=True)
```

Công thức tạo file trên Colab:

```python
file_content = """
nội dung file giống VS Code
"""

with open(f"{project}/ten_file", "w", encoding="utf-8") as f:
    f.write(file_content)
```

Chạy thử:

```python
%cd /content/myduyen_TH2
!python src/train.py
!cat metrics.txt
```

Nén và tải về:

```python
%cd /content
import shutil
shutil.make_archive(project, "zip", project)

from google.colab import files
files.download(f"{project}.zip")
```
