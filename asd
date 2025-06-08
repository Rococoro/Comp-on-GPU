# A. 남아 있는 12-9 / 575 패키지 목록 확인
dpkg -l | grep -E 'cuda-.*12-9|libnvidia.*575|nvidia-.*575' | awk '{print $2}'

# B. 한 번에 제거 (에러 나도 계속)
sudo dpkg --purge --force-all $(dpkg -l | \
   grep -E 'cuda-.*12-9|libnvidia.*575|nvidia-.*575' | awk '{print $2}') || true




sudo update-alternatives --remove-all nvidia || true
sudo update-alternatives --remove-all cuda || true

# repo pin 파일/레포 파일 제거
sudo rm -f /etc/apt/preferences.d/cuda-repository-pin-*
sudo rm -f /etc/apt/sources.list.d/*cuda*.list


sudo apt update
sudo apt --fix-broken install        # 이번엔 오류 없이 끝나야 합니다.
sudo apt autoremove --purge          # 고아 패키지 정리


sudo apt install -y cmake cmake-data librhash0
