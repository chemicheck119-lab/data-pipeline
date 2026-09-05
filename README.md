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
  --collected-at 2026-09-05T07:16:31Z \
  --output-dir data/manifests
```

Manifest는 append-only이므로 동일한 출력 경로가 있으면 명령이 실패합니다. 새 원본 snapshot은 새 버전의 고유 경로로 기록해야 합니다. 첫 snapshot에는 비교 기준이 없으므로 source drift를 `passed`가 아닌 `not_applicable`로 기록합니다.

후속 snapshot에서는 이전 manifest 두 개가 있는 디렉터리를 `--baseline-dir`로 지정합니다. 아카이브 SHA-256이 하나라도 바뀌면 새 manifest를 발행하지 않고 실패하며, 모두 같을 때만 source drift를 `passed`로 기록합니다. 고정 snapshot은 Training 659건과 Validation 77건을 모두 충족해야 합니다.

이 데이터는 [AI 허브 개방 데이터 이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do)의 2026-09-05 조회본을 적용했습니다. AI 모델 학습 목적, NIA 사업결과 표기, 승인 없는 제3자 열람·재배포 금지, 국외 반출 시 별도 합의 조건을 manifest에 기록했습니다. 게시된 정책 자체에는 버전 식별자가 없어 조회일도 함께 남겼습니다.

이번 AIHub ZIP에서는 라벨의 `audioPath`가 실제 압축 파일 안의 WAV 이름과 일치하지 않습니다. Training 659건과 Validation 77건 모두 라벨 JSON과 WAV의 **압축 멤버 파일명 stem**이 1:1로 일치해 그 규칙으로 연결했으며, provider path 불일치 건수를 manifest에 남깁니다. 이는 제공 원본의 한계이며 별도의 공식 키가 확인되기 전까지 “완전 검증된 연결”로 표현하지 않습니다.

이 평가는 `speech_aihub119_gwangju_fire_validation_77`이며 Resolver 울산 419건, Parser 전국 442건 평가와 별개입니다. 신고접수 음성이므로 실제 소방 무전 성능을 증명하지 않습니다.

### 교차지역 Validation manifest

서울·인천처럼 학습 partition을 내려받지 않고 Validation만 외부평가에 사용하는 경우에는
별도 명령으로 지역별 manifest를 만듭니다. 실제 레코드 수를 검사한 뒤 평가 ID에 고정하고,
학습 데이터가 없으므로 train/evaluation 간 source·event overlap은 `passed`가 아니라
`not_evaluated`로 기록합니다.

```bash
PYTHONPATH=src python -m chemicheck119_data.aihub119_evaluation \
  --validation-audio /secure/VS_서울_화재.zip \
  --validation-labels /secure/VL_서울_화재.zip \
  --artifact-prefix gs://PRIVATE_BUCKET/raw/aihub/71768/seoul-fire \
  --dataset-id aihub_71768_seoul_fire \
  --dataset-version dataset-71768-downloaded-2026-09-05 \
  --collected-at 2026-09-05T00:00:00Z \
  --output data/manifests/aihub-71768-seoul-fire-validation.json
```

결과를 보기 전에 지역명·모델 설정·우선용어 목록을 고정합니다. 원본과 전사문은 Git에
저장하지 않고 archive 해시·통계·검증 결과만 manifest에 남깁니다.

## 모의 통신 왜곡 파생 데이터

교차지역 원본 기준선을 먼저 측정한 뒤, 같은 Validation 레코드에 `radio-sim-v1` 변형을
적용할 수 있습니다. 전체 데이터를 18배 복제하지 않도록 우선용어 포함·미포함 레코드를
각각 최대 20건씩 고정 해시로 선택합니다. 이 표본은 우선용어 강건성과 false insertion을
함께 보기 위한 의도적 층화 표본이며 모집단 비율을 대표하지 않습니다.

```bash
PYTHONPATH=src python -m chemicheck119_data.radio_simulation \
  --audio-archive /secure/VS_서울_화재.zip \
  --label-archive /secure/VL_서울_화재.zip \
  --source-manifest /secure/aihub-71768-seoul-fire-validation.json \
  --priority-terms ../speech-service/config/domain_hotwords.txt \
  --output-dir /secure/derived/seoul-radio-sim-v1 \
  --artifact-prefix gs://PRIVATE_BUCKET/derived/aihub/71768/seoul-fire/radio-sim-v1 \
  --positive-records 20 --negative-records 20 --seed 119
```

고정 프로필은 깨끗한 대조군, 8kHz·300–3400Hz 대역 제한, 수학적 8-bit μ-law,
사이렌·차량·바람 절차적 잡음의 SNR 20/10/0dB, 송신 시작·종료 300ms 잘림,
-12dBFS 하드 클리핑, -18dB 음량 저하, 3×120ms 끊김, 복합 스트레스 조건을 만듭니다.
원본 ID·원본/산출물 SHA-256·변환 파라미터·레코드별 seed는
`provenance.private.jsonl`에 기록합니다. 이 파일과 음성·라벨 ZIP은 비공개 저장소에만
둡니다. Git에는 필요한 경우 개인정보가 없는 파생 manifest와 집계 결과만 추가합니다.
송신 시작·종료를 자른 조건은 사라진 음성을 deletion으로 평가하기 위해 원래 참조문을
유지하므로, 원래 발화 타임스탬프와 파생 음성의 시간 정렬은 `not_applicable`로 기록합니다.

사이렌·차량·바람은 실제 현장 녹음이 아닌 절차적 신호이며, μ-law도 특정 무전기 코덱의
bit-exact 구현이 아닙니다. 결과는 반드시 **모의 통신 왜곡 평가**로 부르고 현장 무전
검증이라고 표현하지 않습니다. 자세한 사전 조건은
[모의 통신 왜곡 평가 계획](docs/모의_통신_왜곡_평가_계획.md)에 기록했습니다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| 저장소·정책 CI 골격 | 구현 완료 |
| AIHub 신고음성 manifest 생성기 | 구현 완료 |
| 광주 화재 Training 659건·Validation 77건 검사 | 로컬 실행 완료 |
| 원본 보존형 `radio-sim-v1` 생성·provenance 하네스 | 부분 구현 또는 개발용 데모 |
| 서울·인천 실제 파생 데이터 생성·STT 평가 | 설계 완료·실행 전 |
| STT train/dev/test 분할 | AIHub 제공 Training/Validation 사용; 별도 test 미구성 |
| 실제 현장 무전 데이터 | 확보 불가·검증되지 않음 |

## 기본 검증

```bash
python -m pip install --requirement requirements-dev.txt
python scripts/check_repository_policy.py
```
