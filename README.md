# 케미체크119 Data Pipeline

AIHub·KOSHA·ICIS·PRTR 등 공개·승인 데이터를 재현 가능하게 수집하고, 출처와 품질을 검증해 버전이 고정된 manifest를 만드는 저장소입니다.

## 책임

- 데이터 다운로드·수집 스크립트
- 원본 출처·라이선스·기준일 기록
- 정규화·중복 제거·품질검증
- train/dev/test 분할 manifest와 파일 해시
- `speech-service`와 `analysis-engine`이 소비할 데이터 계약

## 저장하지 않는 것

- AIHub 원본 음성 및 개인정보
- KOSHA 등 원문 전체 덤프
- 모델 가중치와 전처리 산출물
- 현재 재고로 오해할 수 있는 미검증 시설 데이터

대용량 원본은 승인된 외부 저장소에 보관하고 Git에는 수집 방법, manifest, 해시, 스키마만 남깁니다. 작은 테스트 fixture는 라이선스·출처·개인정보 없음·SHA-256을 적은 동반 메타데이터를 통과한 경우에만 허용합니다.

Manifest는 용도(`training`, `development`, `evaluation`, `fixture`, `reference`)를 명시합니다. 평가용 manifest는 울산 Resolver 419건과 전국 Parser 442건의 식별자·건수를 서로 다르게 검증하며, 평가 split을 튜닝에 사용했다고 표시하면 CI가 거부합니다.

## AIHub 119 화재 음성 하위셋

전체 87GB를 내려받지 않고 광주 화재 파티션만 독립 기준선으로 사용합니다. 원본은 승인된 비공개 저장소에만 두고, 아래 명령은 음성·라벨 쌍, 중복, 필수 필드, 오디오 형식, split 누수를 검사한 뒤 통계와 SHA-256만 manifest에 기록합니다.

```bash
PYTHONPATH=src python -m chemicheck119_data.aihub119 \
  --training-audio /secure/TS_광주_화재.zip \
  --training-labels /secure/TL_광주_화재.zip \
  --validation-audio /secure/VS_광주_화재.zip \
  --validation-labels /secure/VL_광주_화재.zip \
  --artifact-prefix gs://PRIVATE_BUCKET/raw/aihub/71768/gwangju-fire \
  --output-dir data/manifests
```

이 평가는 `speech_aihub119_gwangju_fire_validation_77`이며 Resolver 울산 419건, Parser 전국 442건 평가와 별개입니다. 신고접수 음성이므로 실제 소방 무전 성능을 증명하지 않습니다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| 저장소·정책 CI 골격 | 구현 완료 |
| AIHub 신고음성 manifest 생성기 | 구현 완료 |
| 광주 화재 Training 659건·Validation 77건 검사 | 로컬 실행 완료 |
| STT train/dev/test 분할 | AIHub 제공 Training/Validation 사용; 별도 test 미구성 |
| 실제 현장 무전 데이터 | 확보 불가·검증되지 않음 |

## 기본 검증

```bash
python -m pip install --requirement requirements-dev.txt
python scripts/check_repository_policy.py
```
