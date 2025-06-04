#!/usr/bin/env bash
#
#  supersonic-channel-download.sh
#
#  설명: NASA TMR “Supersonic Isothermal-Wall Channel Flow” 페이지에서
#       .h5 확장자를 갖는 모든 링크를 자동으로 추출한 뒤,
#       그중 “supersonic-channel” 데이터만을 다운로드합니다.
#
#  사용법:
#    1) 실행 권한 부여: chmod +x supersonic-channel-download.sh
#    2) 스크립트 실행:  ./supersonic-channel-download.sh
#
#  주의: 이 스크립트는 curl, grep, sed, wget이 설치되어 있어야 작동합니다.


### 1) 기본 URL 및 저장 디렉터리 설정
BASE_PAGE_URL="https://turbmodels.larc.nasa.gov/Other_DNS_Data/supersonic-channel.html"
BASE_DIR="supersonic_channel_data"
mkdir -p "${BASE_DIR}"


### 2) 페이지에서 모든 .h5 링크 추출
#    - curl로 HTML 소스를 가져오고,
#    - grep/ sed로 href="…​.h5" 형태의 부분만 골라낸 뒤,
#    - sed로 순수 URL(path만) 형태로 정리
all_h5_links=$(curl -s "${BASE_PAGE_URL}" \
  | grep -Eo 'href="[^"]+\.h5"' \
  | sed -e 's/^href="//' -e 's/"$//')


### 3) “supersonic-channel” 데이터만 필터링
#    - page 내 링크가 상대경로인 경우가 많으므로, 
#      상대경로(예: "data/supersonic_M1p5_Re3000_t000.h5")라면 
#      BASE_URL을 앞에 붙여야 합니다.
#
#    - 링크에 “channel” 혹은 “M1p5” 같은 키워드를 이용해 
#      Mach=1.5 또는 Mach=3.0 케이스를 선택할 수 있습니다.
#
#    예를 들어, Mach=1.5 데이터만 받고 싶다면 “M1p5” 키워드로 필터링:
FILTER_KEYWORD="M1p5"   # Mach=1.5 데이터만 받으려면 "M1p5", Mach=3.0이면 "M3p0" 등으로 수정
# (필요하다면 FILTER_KEYWORD를 "M3p0"으로 바꾸면 Mach=3.0 데이터만 받습니다.)

# 전체 링크 중 FILTER_KEYWORD가 포함된 것만 뽑아서, 
# http://turbmodels.larc.nasa.gov/Other_DNS_Data/ 경로를 붙인 후 다운로드
for rel_path in ${all_h5_links}; do
  if [[ "${rel_path}" == *"${FILTER_KEYWORD}"*".h5" ]]; then
    # 상대경로가 ../Other_DNS_Data/... 이거나 data/... 일 수 있으므로
    # “/Other_DNS_Data” 이하로 경로를 찾아 절대 URL로 조합
    #  ① rel_path에 "http"가 포함되어 있으면 이미 절대경로(아예 링크가 https://로 시작)인 경우
    if [[ "${rel_path}" =~ ^https?:// ]]; then
      full_url="${rel_path}"
    else
      #  ② 상대경로라면 BASE_URL의 디렉터리 경로까지 똑같이 붙여준다.
      #     예: rel_path="supersonic_M1p5_Re3000_t000.h5"
      #         그러면 full_url="https://turbmodels.larc.nasa.gov/Other_DNS_Data/supersonic_M1p5_Re3000_t000.h5"
      page_dir=$(dirname "${BASE_PAGE_URL}")
      full_url="${page_dir}/${rel_path}"
    fi

    # 파일명만 추출 (맨 끝의 슬래시 이후 부분)
    filename=$(basename "${rel_path}")

    # 이미 파일이 존재하지 않을 때만 다운로드
    if [[ ! -f "${BASE_DIR}/${filename}" ]]; then
      echo "[DL] ${filename} from ${full_url}"
      wget -q --show-progress -O "${BASE_DIR}/${filename}" "${full_url}"
    else
      echo "[SKIP] Already exists: ${filename}"
    fi
  fi
done

echo
echo "=== 다운로드 완료 ==="
echo "다운로드된 파일 목록 (총 $(ls -1 ${BASE_DIR}/*.h5 2>/dev/null | wc -l)개):"
ls -1 "${BASE_DIR}"/*.h5 2>/dev/null
