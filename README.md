
# Copyright

## Algorithm 설명

원본 이미지들과 리셀러들의 이미지를 비교하여 리셀러의 이미지가 도용인지 아닌지를 판단합니다.

전체적인 알고리즘 구조는 다음과 같습니다. 

* `faiss`: 대규모 벡터 검색을 위해 개발된 라이브러리

1. 원본 이미지들의 embedding을 faiss에 저장하고, 
2. 리셀러들의 이미지를 embedding으로 추출하고 faiss내의 embedding과의 유사도를 측정하여 판단

## 레포 구조 

```
├── assets # 예제 이미지 저장소
│   ├── etc
│   ├── original
│   └── reseller
├── copyright # 패키지
│   ├── configs # 각각의 알고리즘을 실행하기 위한 파라미터 저장소
│   │   ├── default.yaml
│   │   └── loftr.yaml
│   └── src 
│       ├── embedder
│       │   ├── __init__.py
│       │   ├── embdder.py
│       │   ├── index.py
│       │   └── loftr.py
│       │   └── __init__.py
│       └── utils
│           ├── build_index.py
│           ├── run_loftr.py
│           ├── scan.py
│           └── version.py
├── Dockerfile
├── README.md
├── requirements.txt
├── setup.py
└── test.py
```


#### `configs/default.yaml`

```yaml
device: "cuda"   
original_image_dir: /workspace/assets/original
reseller_image_dir: /workspace/assets/reseller
output_dir: /workspace/outputs

patch:
  use: true
  width: 1024
  height: 1024

embedding:
  # name should be one of ["dinov2_vitg14", "dinov2_vitl14", "dinov2_vitb14", "dinov2_vits14", 
  #                        "clip_vitl14"]
  name: "dinov2_vitg14" 
  input_size: 518
  output_dir: embedding

faiss:
  index_file: /workspace/outputs/embedding/originals.faiss
  catalog_file: /workspace/outputs/embedding/originals.jsonl
  normalize: true
  faiss_nlist: 1
  faiss_nprobe: 1
  topk: 1
  output_dir: scan

thresholds:
  copied: 0.4
```

* `device`: `cpu`와 `cuda` 중에 선택할 수 있지만, 성능측면에서 필수적으로 `cuda` 필요
* `original_image_dir`: 원본 이미지 저장소
* `reseller_image_dir`: 리셀러 이미지 저장소
* `output_dir`: 알고리즘 실행 후 결과물을 저장할 장소

* `patch`: 비교할 이미지의 크기가 크므로 이를 window방식으로 cropping하여 검사할지를 결정할 파라미터
    * `use`: `true` or `false` 
    * `width`: patch의 width
    * `height`: patch의 height

* `embedding`: `faiss`에서 사용할 임베딩을 추출할 알고리즘에 대한 파라미터
    * `name`: one of ["dinov2_vitg14", "dinov2_vitl14", "dinov2_vitb14", "dinov2_vits14"]
    * `input_size`: 518 # dinov2의 경우 518로 사용하길 권장
    * `output_dir`: embedding # 임베딩 알고리즘에 대한 결과물 저장소 폴더 이름

* `faiss`: 앞선 임베딩 결과를 이용해서 비교하기 위한 알고리즘 파라미터
    * `index_file`: `embedding` 알고리즘을 돌린 결과물의 `originals.faiss` 위치
    * `catalog_file`: `embedding` 알고리즘을 돌린 결과물의 `originals.jsonl` 위치
    * `normalize``: true # 고정하길 권장
    * `faiss_nlist`: 1 # 고정하길 권장
    * `faiss_nprobe`: 1 # 고정하길 권장
    * `topk`: `faiss`를 실행하면 유사도의 ranking 순으로 비슷한 이미지에 대한 인덱스가 추출되며, 몇 순위까지 추출할지 결정
    * `output_dir`: `scan` 알고리즘을 실행한 결과물 저장소 폴더 이름

* thresholds: 
*   copied: 유사도에 대한 기준, 이 수치를 넘어야 `copied`로서 결정되며, 그렇지 않으며, `uncertain`으로 표시

## How to run

#### run docker
```sh
git clone git@github.com:wonchul-kim/similarity_inspection.git
cd similarity_inspection
docker build -t copyright .
docker run -it --name copyright --ipc host --gpus all -v <current absolute path>:/workspace copyright bash
```

#### Install 
```sh
apt-get update 
apt-get install -y libgl1
apt-get install -y libglib2.0-0

cd /workspace
pip inestall -e .
pip install -r requiremets.txt
```

#### run algorithm

1. Build index

```sh
python3 copyright/build_index.py
```

* `configs/default.yaml`을 토대로 알고리즘이 실행되므로 파라미터 수정은 `configs/default.yaml`에서 진행
* `src/embedder/embdder.py`는 원본 이미지로부터 embedding을 생성하는 class를 제공
* embedding을 추출하고, 해당 embedding에 대한 이미지와 매칭을 위한 index도 함꼐 추출
* 결과물로서 `originals.faiss`와 `originals.jsonl`이 생성되고, 이를 `faiss`에서 사용함
    * `originals.faiss`: 임베딩 벡터가 저장되어 있음
    * `originals.jsonl`: 원본 이미지와 embedding을 매칭하기 위한 index가 적혀 있음

2. Scan

```sh
python3 copyright/scan.py
```

* `originals.faiss`와 `originals.jsonl`를 로드함
* 리셀러의 이미지를 로드하고, `faiss`를 사용해서 임베딩 벡터에 리셀러의 임베딩과 비슷한 이미지가 있는 확인
* 결과물로서 `scan.json`이 생성되고, 각 이미지마다 또는 각 패치마다의 유사도 수치가 추출되어 있음