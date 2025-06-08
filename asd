sudo dpkg --remove --force-remove-reinstreq cuda-runtime-12-9 \
     libnvidia-compute-575 libnvidia-decode-575 \
     nvidia-compute-utils-575 nvidia-driver-575-open \
     libnvidia-extra-575 libnvidia-gl-575 2>/dev/null || true
