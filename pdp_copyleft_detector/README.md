# PDP Copyleft Detector (Deep Learning)

정확도(재현율) 최우선으로 설계된 **PDP(상품 상세 페이지) 이미지 도용 탐지** 레포입니다.  
원본(PDP_*.jpg)과 리셀러 페이지(*.png) 이미지가 있을 때, 딥러닝 기반으로 **유사도/지역 정합/문자 의미**를 결합해
도용 여부를 고신뢰로 판별합니다.

## 핵심 아이디어
1. **Global Visual Embedding (DINOv2/CLIP)**: 강건한 전역 임베딩으로 편집/크롭/압축에도 견고한 유사도 계산
2. **Local Region Matching (LoFTR)**: 로고, 제품컷 등 지역 특징 정합률로 미세 편집에도 강건
3. **OCR + Sentence Embedding**: 제품명/성분/카피 등을 딥러닝 OCR→문장 임베딩하여 의미 유사도 측정
4. **Learned Fusion (MLP)**: 위 3개 신호를 소형 MLP로 학습 결합 → 최종 도용 점수 산출
5. **Reranking + ANN**: 대규모 탐색을 위해 FAISS로 Top-K 후보를 빨리 찾고, 정교한 재평가로 오탐 최소화

## 폴더 구조
```
src/copyleft_detector/   # 라이브러리 모듈
scripts/                 # 학습/인덱스/스캔 스크립트
data/originals           # 원본 PDP 이미지 (예시)
data/resellers           # 리셀러 이미지 (예시)
artifacts/               # 모델/인덱스/캐시
results/                 # 결과 JSON/리포트
configs/                 # 설정(YAML)
```

## 빠른 시작
```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 원본 임베딩 인덱스 구축
python scripts/build_index.py --originals data/originals --out artifacts

# 3) 리셀러 스캔 (후보 탐색 → 재평가 → 결과 저장)
python scripts/scan_dir.py --resellers data/resellers     --index artifacts/originals.faiss --catalog artifacts/originals.jsonl     --out results/scan.jsonl --save-viz
```

## 학습 (선택)
라벨링된 (원본,리셀러,라벨) 페어 CSV가 있다면:
```bash
python scripts/train_pair_classifier.py --pairs_csv data/pairs.csv --epochs 5     --save_path artifacts/fusion_mlp.pt
```

## 결과
- `results/scan.jsonl`: 이미지별 상위 매칭, 최종 도용 점수, 세부 근거(전역 유사도, 로컬 정합률, OCR 의미 유사도)
- `results/viz/*`: LoFTR 매칭 시각화 이미지(옵션)

## 주의
- 초기 Threshold는 `scripts/calibrate_thresholds.py`로 검증셋에 맞춰 보정하세요.
- LoFTR/TrOCR/CLIP 등 모델은 최초 실행 시 가중치를 자동 다운로드할 수 있습니다.
