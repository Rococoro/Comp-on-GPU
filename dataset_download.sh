#!/usr/bin/env bash
#
# supersonic_channel_batch_download.sh
#
# NASA TMR “Supersonic Isothermal-Wall Channel Flow” 페이지에서
# Mach=1.5 케이스(supersonic_M1p5_Re3000_t*.bin) 파일만 전부 내려받는 스크립트입니다.
#
# 사용 전 확인할 사항:
#  1) 반드시 curl, grep, sed, wget, awk, tr 명령이 설치되어 있어야 합니다.
#  2) 네트워크 상에서 NASA TMR 서버(turbmodels.larc.nasa.gov)에 접근이 가능한 상태여야 합니다.
#  3) 아래 BASE_PAGE_URL 변수를 통해 “Supersonic Isothermal-Wall Channel Flow” 페이지에 접근합니다.
#     스크립트를 실행하기 전에 반드시 브라우저로 한 번 들어가서 
#     페이지 상단에 있는 “Data for M=1.5 case” 링크가 정상 동작하는지 확인해 주세요.
#
# 사용법:
#   chmod +x supersonic_channel_batch_download.sh
#   ./supersonic_channel_batch_download.sh
# 
# 만약 Mach=3.0 데이터(“supersonic_M3p0_…_tXXX.bin”)를 받고 싶다면
#   FILTER_KEYWORD="M3p0"
# 로 바꿔 주시면 됩니다.

#-------------------------------
# 1) 환경변수 설정
#-------------------------------
# “Supersonic Isothermal-Wall Channel Flow” 페이지 주소
BASE_PAGE_URL="https://turbmodels.larc.nasa.gov/Other_DNS_Data/supersonic-channel.html"

# NASA TMR 페이지에서 실제 파일(.bin)이 올라와 있는 디렉토리(예상)
# “Supersonic Isothermal-Wall Channel Flow” 페이지 하단의 링크가 보통 상대경로로 되어 있으므로,
# 브라우저에서 링크를 눌러보면 나오는 URL(디렉터리)까지 지정해 줍니다.
# 실제 링크를 클릭했을 때 예를 들어 다음과 같은 주소가 나와야 합니다:
#   https://turbmodels.larc.nasa.gov/Other_DNS_Data/supersonic_M1p5_Re3000_t000.bin 
# 만약 다른 디렉터리에 올라가 있다면(예: data/…/supersonic_…) 그 경로로 변경해 주세요.
BASE_DATA_DIR="https://turbmodels.larc.nasa.gov/Other_DNS_Data"

# 파일 이름 접두사(PREFIX)와 확장자(EXT)
#   Mach=1.5 케이스: supersonic_M1p5_Re3000_tXXX.bin
#   Mach=3.0 케이스: supersonic_M3p0_Re3000_tXXX.bin
FILTER_KEYWORD="M1p5"    # “M1p5” 또는 “M3p0” 등 원하는 Mach 케이스에 맞춰 바꿔주세요.
PREFIX="supersonic_${FILTER_KEYWORD}_Re3000_t"
EXT="bin"                # 실제 페이지에 올라온 자료가 .bin이면 bin, .h5라면 h5로 바꾸세요.

# 숫자 인덱스를 몇 자리로 표기했는지 (페이지에서 000, 001, 002 … 처럼 3자리인지 확인)
#   예: "%03g" → 000, 001, … /  "%02g" → 00, 01, …  /  "%04g" → 0000, 0001, …
NUM_FORMAT="%03g"

#-------------------------------
# 2) “Supersonic Isothermal-Wall Channel Flow” 페이지 내에 .bin 링크가 있는지 확인
#-------------------------------
echo "1) Supersonic 채널 페이지를 검사하여 .${EXT} 링크 목록을 가져옵니다…"
echo "   페이지 URL: ${BASE_PAGE_URL}"

# (A) 페이지에서 HTML 소스를 가져와, “.${EXT}” 인 부분만 grep
# (B) sed를 통해 href="…bin" 형태에서 순수 파일명만 추출
all_links=$(curl -s "${BASE_PAGE_URL}" \
  | grep -Eo "href=\"[^\"]+\.${EXT}\"" \
  | sed -e 's/^href="//' -e 's/"$//')

if [[ -z "${all_links}" ]]; then
  echo "Error: 페이지에서 .${EXT} 링크를 전혀 찾을 수 없습니다. URL을 확인해 주세요."
  exit 1
fi

echo "   총 찾은 .${EXT} 링크 수: $(echo \"${all_links}\" | wc -l)"
echo

#-------------------------------
# 3) 실제 다운로드할 파일만 골라서 wget 수행
#-------------------------------
mkdir -p supersonic_${FILTER_KEYWORD}_data
echo "2) Mach=${FILTER_KEYWORD} 케이스 파일만 골라서 supersonic_${FILTER_KEYWORD}_data/ 폴더에 다운로드합니다…"
echo

count=0
while IFS= read -r relpath; do
  # relpath 예시: "supersonic_M1p5_Re3000_t000.bin" 또는 "data/supersonic_M1p5_Re3000_t000.bin"
  # 일단 파일 이름만 뽑아내자
  filename=$(basename "${relpath}")

  # 우리가 원하는 Mach 케이스만 필터링
  if [[ "${filename}" != *"${FILTER_KEYWORD}"*".${EXT}" ]]; then
    echo "   [SKIP] ${filename} (Mach 키워드 불일치)"
    continue
  fi

  # 절대 URL을 구성: 
  #   - relpath이 이미 “http…”로 시작하면 그대로 사용
  #   - 아니라면 BASE_DATA_DIR을 붙여 준다
  if [[ "${relpath}" =~ ^https?:// ]]; then
    full_url="${relpath}"
  else
    full_url="${BASE_DATA_DIR}/${relpath}"
  fi

  # 실제 다운로드
  if [[ ! -f "supersonic_${FILTER_KEYWORD}_data/${filename}" ]]; then
    echo "   [DL] ${filename} from ${full_url}"
    wget -q --show-progress -O "supersonic_${FILTER_KEYWORD}_data/${filename}" "${full_url}"
    if [[ $? -ne 0 ]]; then
      echo "        → 다운로드 실패 (HTTP 에러 등)."
      # 실패 비율이 너무 높다면, 스크립트를 중단하길 원할 수 있습니다. 
      # 여기서는 “계속 진행”으로 두어, 나머지 파일도 시도합니다.
    else
      ((count++))
    fi
  else
    echo "   [SKIP] 이미 존재함: ${filename}"
  fi

done < <(echo "${all_links}")

echo
echo "=== 다운로드 완료 ==="
echo "총 다운로드된 파일 수: ${count}"
echo "다운로드된 파일 목록은 supersonic_${FILTER_KEYWORD}_data/ 폴더를 확인하세요."
