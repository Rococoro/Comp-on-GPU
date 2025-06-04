#!/usr/bin/env bash
#
# supersonic-channel-download.sh
#
# NASA Turbulence Modeling Resource “Supersonic Isothermal-Wall Channel Flow” 데이터 중
# Mach=1.5 (M1p5) 케이스의 time‐series 파일(보통 .bin 형식으로 제공됨)을 한꺼번에 받아오기 위한 스크립트입니다.
#
# 실제 페이지(https://turbmodels.larc.nasa.gov/Other_DNS_Data/supersonic-channel.html)에서
# 파일 목록을 확인해 보면, 파일명이 예를 들어 다음과 같은 형태로 나열되어 있습니다:
#   supersonic_M1p5_Re3000_t000.bin
#   supersonic_M1p5_Re3000_t001.bin
#   supersonic_M1p5_Re3000_t002.bin
#   …
#
# 스크립트 사용법:
#   1) 실행 권한 부여: chmod +x supersonic-channel-download.sh
#   2) 스크립트 실행: ./supersonic-channel-download.sh
#
# 만약 실제 파일명이 .h5인 경우:
#   • EXT="h5" 로 바꾸면 됩니다.
# 만약 Mach=3.0 데이터만 받고 싶다면:
#   • FILTER_KEYWORD="M3p0" 으로 바꿔 주시면 됩니다.

### 1) 기본 설정: URL, 키워드, 숫자 형식, 인덱스 범위 등
BASE_PAGE_DIR="https://turbmodels.larc.nasa.gov/Other_DNS_Data"
# → 실제 다운로드 URL은 BASE_PAGE_DIR/<파일명> 형태입니다.

PREFIX="supersonic_M1p5_Re3000_t"   # Mach=1.5, Re=3000 기준
EXT="bin"                           # 실제 페이지에 올라온 확장자(.bin 또는 .h5)에 맞춰 수정
NUM_DIGITS="%03g"                   # 인덱스 부분을 3자리(000,001,…) 형식으로 생성

# 다운로드할 시간 스텝 인덱스 범위 (예: t000 ~ t299 → 총 300개)
START_IDX=0
END_IDX=299

# 데이터를 저장할 로컬 디렉터리
OUTPUT_DIR="supersonic_M1p5_data"
mkdir -p "${OUTPUT_DIR}"


### 2) 실제 다운로드 루프
for (( i=START_IDX; i<=END_IDX; i++ )); do
  idx=$(printf "${NUM_DIGITS}" "${i}")            # "000", "001", … 생성
  filename="${PREFIX}${idx}.${EXT}"               # ex) supersonic_M1p5_Re3000_t000.bin
  full_url="${BASE_PAGE_DIR}/${filename}"

  # HTTP 상태 코드 먼저 확인 (HEAD 요청)
  http_code=$(curl -s -o /dev/null -w "%{http_code}" -I "${full_url}")
  if [[ "${http_code}" == "200" ]]; then
    # 파일 존재 → 실제 다운로드
    echo "[OK]   Downloading ${filename} …"
    wget -q --show-progress -O "${OUTPUT_DIR}/${filename}" "${full_url}"
  else
    # 파일이 없거나(404) 권한 문제(403 등)
    echo "[SKIP] ${filename} (HTTP ${http_code})"
  fi
done

echo
echo "=== 다운로드 완료 (혹은 SKIP) ==="
echo "총 다운로드된 파일 수: $(ls -1 "${OUTPUT_DIR}"/*.${EXT} 2>/dev/null | wc -l)"
