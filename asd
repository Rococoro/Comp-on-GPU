sudo apt --fix-broken install
Reading package lists... Done
Building dependency tree       
Reading state information... Done
Correcting dependencies... Done
The following packages were automatically installed and are no longer required:
  cuda-cccl-12-9 cuda-command-line-tools-12-9 cuda-compiler-12-9 cuda-crt-12-9
  cuda-cudart-12-9 cuda-cudart-dev-12-9 cuda-cuobjdump-12-9 cuda-cupti-12-9
  cuda-cupti-dev-12-9 cuda-cuxxfilt-12-9 cuda-documentation-12-9
  cuda-driver-dev-12-9 cuda-gdb-12-9 cuda-libraries-12-9
  cuda-libraries-dev-12-9 cuda-nsight-compute-12-9 cuda-nsight-systems-12-9
  cuda-nvcc-12-9 cuda-nvdisasm-12-9 cuda-nvml-dev-12-9 cuda-nvprune-12-9
  cuda-nvrtc-12-9 cuda-nvrtc-dev-12-9 cuda-nvtx-12-9 cuda-nvvm-12-9
  cuda-profiler-api-12-9 cuda-sanitizer-12-9 cuda-toolkit-12-9
  cuda-toolkit-12-9-config-common cuda-toolkit-12-config-common
  cuda-tools-12-9 cuda-visual-tools-12-9 dkms gdal-data gds-tools-12-9 libaec0
  libarmadillo9 libarpack2 libavcodec-dev libavformat-dev libavresample-dev
  libavresample4 libavutil-dev libcfitsio8 libcharls2 libcublas-12-9
  libcublas-dev-12-9 libcufft-12-9 libcufft-dev-12-9 libcufile-12-9
  libcufile-dev-12-9 libcurand-12-9 libcurand-dev-12-9 libcusolver-12-9
  libcusolver-dev-12-9 libcusparse-12-9 libcusparse-dev-12-9 libdap25
  libdapclient6v5 libdc1394-22-dev libepsilon1 libevent-core-2.1-7
  libevent-pthreads-2.1-7 libexif-dev libfreexl1 libfyba0 libgdal26
  libgdcm-dev libgdcm3.0 libgeos-3.8.0 libgeos-c1v5 libgeotiff5 libgl2ps1.4
  libgphoto2-dev libhdf4-0-alt libhdf5-103 libhdf5-openmpi-103
  libhwloc-plugins libhwloc15 libilmbase-dev libjbig-dev libjpeg-dev
  libjpeg-turbo8-dev libjpeg8-dev libjsoncpp1 libkmlbase1 libkmldom1
  libkmlengine1 liblept5 liblzma-dev libminizip1 libnetcdf-c++4 libnetcdf15
  libnpp-12-9 libnpp-dev-12-9 libnvfatbin-12-9 libnvfatbin-dev-12-9
  libnvidia-cfg1-575 libnvidia-common-575 libnvidia-fbc1-575
  libnvidia-gpucomp-575 libnvjitlink-12-9 libnvjitlink-dev-12-9 libnvjpeg-12-9
  libnvjpeg-dev-12-9 libodbc1 libogdi4.1 libopencv-calib3d4.2
  libopencv-contrib4.2 libopencv-dnn4.2 libopencv-features2d4.2
  libopencv-flann4.2 libopencv-highgui4.2 libopencv-imgcodecs4.2
  libopencv-imgproc4.2 libopencv-ml4.2 libopencv-objdetect4.2
  libopencv-photo4.2 libopencv-shape4.2 libopencv-stitching4.2
  libopencv-superres4.2 libopencv-video4.2 libopencv-videoio4.2
  libopencv-videostab4.2 libopencv-viz4.2 libopencv4.2-java libopencv4.2-jni
  libopenexr-dev libopenmpi3 libpmix2 libpng-dev libpq5 libproj15 libqhull7
  libraw1394-dev libsocket++1 libspatialite7 libsuperlu5 libswresample-dev
  libswscale-dev libsz2 libtbb-dev libtesseract4 libtiff-dev libtiffxx5
  liburiparser1 libvtk6.3 libxcb-cursor0 libxerces-c3.2 libxnvctrl0
  linux-headers-5.4.0-216 linux-headers-5.4.0-216-generic
  linux-headers-generic nsight-compute-2025.2.1 nsight-systems-2025.1.3
  nvidia-dkms-575-open nvidia-firmware-575-575.57.08 nvidia-kernel-common-575
  nvidia-kernel-source-575-open nvidia-modprobe nvidia-prime nvidia-settings
  odbcinst odbcinst1debian2 proj-data screen-resolution-extra
  xserver-xorg-video-nvidia-575
Use 'sudo apt autoremove' to remove them.
The following packages will be REMOVED:
  cuda cuda-12-9 cuda-runtime-12-9 libnvidia-decode-575 libnvidia-encode-575
  nvidia-compute-utils-575 nvidia-driver-575-open nvidia-open
0 upgraded, 0 newly installed, 8 to remove and 331 not upgraded.
81 not fully installed or removed.
After this operation, 20.4 MB disk space will be freed.
Do you want to continue? [Y/n] Y
(Reading database ... 195949 files and directories currently installed.)
Removing cuda (12.9.1-1) ...
Removing cuda-12-9 (12.9.1-1) ...
Removing cuda-runtime-12-9 (12.9.1-1) ...
Removing nvidia-open (575.57.08-0ubuntu1) ...
Removing nvidia-driver-575-open (575.57.08-0ubuntu1) ...
Removing libnvidia-encode-575:arm64 (575.57.08-0ubuntu1) ...
Removing libnvidia-decode-575:arm64 (575.57.08-0ubuntu1) ...
Removing nvidia-compute-utils-575 (575.57.08-0ubuntu1) ...
Setting up linux-headers-5.4.0-216 (5.4.0-216.236) ...
Setting up gds-tools-12-9 (1.14.1.1-1) ...
Setting up nsight-compute-2025.2.1 (2025.2.1.3-1) ...
Setting up cuda-cuobjdump-12-9 (12.9.82-1) ...
Setting up cuda-nvrtc-12-9 (12.9.86-1) ...
Setting up cuda-sanitizer-12-9 (12.9.79-1) ...
Setting up cuda-nvvm-12-9 (12.9.86-1) ...
Setting up nvidia-kernel-source-575-open (575.57.08-0ubuntu1) ...
Setting up linux-headers-5.4.0-216-generic (5.4.0-216.236) ...
Setting up cuda-cupti-12-9 (12.9.79-1) ...
Setting up nvidia-prime (0.8.16~0.20.04.2) ...
Setting up cuda-nvml-dev-12-9 (12.9.79-1) ...
Setting up nvidia-firmware-575-575.57.08 (575.57.08-0ubuntu1) ...
Setting up cuda-nvprune-12-9 (12.9.82-1) ...
Setting up libnvidia-gpucomp-575:arm64 (575.57.08-0ubuntu1) ...
Setting up cuda-nvrtc-dev-12-9 (12.9.86-1) ...
Setting up linux-headers-generic (5.4.0.216.208) ...
Setting up dkms (2.8.1-5ubuntu2) ...
Setting up nvidia-modprobe (575.57.08-0ubuntu1) ...
Setting up cuda-toolkit-12-9-config-common (12.9.79-1) ...
Setting alternatives
update-alternatives: error: alternative path /usr/local/cuda-12.9 doesn't exist
dpkg: error processing package cuda-toolkit-12-9-config-common (--configure):
 installed cuda-toolkit-12-9-config-common package post-installation script subp
rocess returned error exit status 2
Setting up cuda-driver-dev-12-9 (12.9.79-1) ...
Setting up cuda-documentation-12-9 (12.9.88-1) ...
Setting up libxcb-cursor0:arm64 (0.1.1-4ubuntu1) ...
Setting up cuda-nvdisasm-12-9 (12.9.88-1) ...
Setting up libnvidia-fbc1-575:arm64 (575.57.08-0ubuntu1) ...
dpkg: dependency problems prevent configuration of libnvfatbin-12-9:
 libnvfatbin-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package libnvfatbin-12-9 (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of libnvjpeg-12-9:
 libnvjpeg-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package libnvjpeg-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because the error message indicates its a followup erro
r from a previous failure.
                          No apport report written because the error message ind
icates its a followup error from a previous failure.
                                                    Setting up cuda-profiler-api
-12-9 (12.9.79-1) ...
Setting up nvidia-kernel-common-575 (575.57.08-0ubuntu1) ...
update-initramfs: deferring update (trigger activated)
update-initramfs: deferring update (trigger activated)
dpkg: dependency problems prevent configuration of libcusolver-12-9:
 libcusolver-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package libcusolver-12-9 (--configure):
 dependency problems - leaving unconfigured
Setting up cuda-nsight-compute-12-9 (12.9.1-1) ...
No apport report written because MaxReports is reached already
                                                              dpkg: dependency p
roblems prevent configuration of libcufile-12-9:
 libcufile-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package libcufile-12-9 (--configure):
 dependency problems - leaving unconfigured
Setting up screen-resolution-extra (0.18build1) ...
No apport report written because MaxReports is reached already
                                                              Setting up cuda-cu
xxfilt-12-9 (12.9.82-1) ...
dpkg: dependency problems prevent configuration of libnvfatbin-dev-12-9:
 libnvfatbin-dev-12-9 depends on libnvfatbin-12-9 (>= 12.9.82); however:
  Package libnvfatbin-12-9 is not configured yet.

dpkg: error processing package libnvfatbin-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              Setting up nvidia-
settings (575.57.08-0ubuntu1) ...
Setting up cuda-cccl-12-9 (12.9.27-1) ...
dpkg: dependency problems prevent configuration of libcusolver-dev-12-9:
 libcusolver-dev-12-9 depends on libcusolver-12-9 (>= 11.7.5.82); however:
  Package libcusolver-12-9 is not configured yet.

dpkg: error processing package libcusolver-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              Setting up libnvid
ia-common-575 (575.57.08-0ubuntu1) ...
dpkg: dependency problems prevent configuration of cuda-cudart-12-9:
 cuda-cudart-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package cuda-cudart-12-9 (--configure):
 dependency problems - leaving unconfigured
Setting up cuda-cupti-dev-12-9 (12.9.79-1) ...
No apport report written because MaxReports is reached already
                                                              Setting up cuda-nv
tx-12-9 (12.9.79-1) ...
dpkg: dependency problems prevent configuration of cuda-cudart-dev-12-9:
 cuda-cudart-dev-12-9 depends on cuda-cudart-12-9 (>= 12.9.79); however:
  Package cuda-cudart-12-9 is not configured yet.

dpkg: error processing package cuda-cudart-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
Setting up libnvidia-cfg1-575:arm64 (575.57.08-0ubuntu1) ...
No apport report written because MaxReports is reached already
                                                              Setting up cuda-to
olkit-12-config-common (12.9.79-1) ...
dpkg: dependency problems prevent configuration of cuda-crt-12-9:
 cuda-crt-12-9 depends on cuda-cudart-dev-12-9; however:
  Package cuda-cudart-dev-12-9 is not configured yet.

dpkg: error processing package cuda-crt-12-9 (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of libcufft-12-9:
 libcufft-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package libcufft-12-9 (--configure):
 dependency problems - leaving unconfigured
Setting up nvidia-dkms-575-open (575.57.08-0ubuntu1) ...
No apport report written because MaxReports is reached already
                                                              No apport report w
ritten because MaxReports is reached already
                                            WARNING: unsupported arch: arm64
update-initramfs: deferring update (trigger activated)
INFO:Enable nvidia
DEBUG:Parsing /usr/share/ubuntu-drivers-common/quirks/put_your_quirks_here
DEBUG:Parsing /usr/share/ubuntu-drivers-common/quirks/dell_latitude
DEBUG:Parsing /usr/share/ubuntu-drivers-common/quirks/lenovo_thinkpad
Loading new nvidia-575.57.08 DKMS files...
It is likely that 5.10.192-tegra belongs to a chroot's host
Building for 5.10.192-tegra and 5.4.0-216-generic
Building for architecture arm64
Building initial module for 5.10.192-tegra
Done.

nvidia.ko:
Running module version sanity check.
 - Original module
 - Installation
   - Installing to /lib/modules/5.10.192-tegra/updates/dkms/

nvidia-modeset.ko:
Running module version sanity check.
 - Original module
 - Installation
   - Installing to /lib/modules/5.10.192-tegra/updates/dkms/

nvidia-drm.ko:
Running module version sanity check.
 - Original module
 - Installation
   - Installing to /lib/modules/5.10.192-tegra/updates/dkms/

nvidia-uvm.ko:
Running module version sanity check.
 - Original module
 - Installation
   - Installing to /lib/modules/5.10.192-tegra/updates/dkms/

nvidia-peermem.ko:
Running module version sanity check.
 - Original module
 - Installation
   - Installing to /lib/modules/5.10.192-tegra/updates/dkms/

depmod....

DKMS: install completed.
Building initial module for 5.4.0-216-generic
Done.

nvidia.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/5.4.0-216-generic/updates/dkms/

nvidia-modeset.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/5.4.0-216-generic/updates/dkms/

nvidia-drm.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/5.4.0-216-generic/updates/dkms/

nvidia-uvm.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/5.4.0-216-generic/updates/dkms/

nvidia-peermem.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/5.4.0-216-generic/updates/dkms/

depmod...

DKMS: install completed.
dpkg: dependency problems prevent configuration of cuda-nvcc-12-9:
 cuda-nvcc-12-9 depends on cuda-cudart-dev-12-9; however:
  Package cuda-cudart-dev-12-9 is not configured yet.
 cuda-nvcc-12-9 depends on cuda-crt-12-9 (= 12.9.86-1); however:
  Package cuda-crt-12-9 is not configured yet.

dpkg: error processing package cuda-nvcc-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              No apport report w
ritten because MaxReports is reached already
                                            dpkg: dependency problems prevent co
nfiguration of libcublas-12-9:
 libcublas-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package libcublas-12-9 (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of libcusparse-12-9:
 libcusparse-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package libcusparse-12-9 (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of libnvjpeg-dev-12-9:
 libnvjpeg-dev-12-9 depends on libnvjpeg-12-9 (>= 12.4.0.76); however:
  Package libnvjpeg-12-9 is not configured yet.

dpkg: error processing package libnvjpeg-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              No apport report w
ritten because MaxReports is reached already
                                            dpkg: dependency problems prevent co
nfiguration of libcufile-dev-12-9:
 libcufile-dev-12-9 depends on libcufile-12-9 (>= 1.14.1.1); however:
  Package libcufile-12-9 is not configured yet.

dpkg: error processing package libcufile-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              Setting up nsight-
systems-2025.1.3 (2025.1.3.140-251335620677v0) ...
update-alternatives: using /opt/nvidia/nsight-systems/2025.1.3/target-linux-sbsa
-armv8/nsys to provide /usr/local/bin/nsys (nsys) in manual mode
update-alternatives: using /opt/nvidia/nsight-systems/2025.1.3/host-linux-armv8/
nsys-ui to provide /usr/local/bin/nsys-ui (nsys-ui) in manual mode
dpkg: dependency problems prevent configuration of libcufft-dev-12-9:
 libcufft-dev-12-9 depends on libcufft-12-9 (>= 11.4.1.4); however:
  Package libcufft-12-9 is not configured yet.

dpkg: error processing package libcufft-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of libnpp-12-9:
 libnpp-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package libnpp-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              No apport report w
ritten because MaxReports is reached already
                                            dpkg: dependency problems prevent co
nfiguration of libnpp-dev-12-9:
 libnpp-dev-12-9 depends on libnpp-12-9 (>= 12.4.1.87); however:
  Package libnpp-12-9 is not configured yet.

dpkg: error processing package libnpp-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of libnvjitlink-12-9:
 libnvjitlink-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package libnvjitlink-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              No apport report w
ritten because MaxReports is reached already
                                            Setting up xserver-xorg-video-nvidia
-575 (575.57.08-0ubuntu1) ...
dpkg: dependency problems prevent configuration of libcurand-12-9:
 libcurand-12-9 depends on cuda-toolkit-12-9-config-common; however:
  Package cuda-toolkit-12-9-config-common is not configured yet.

dpkg: error processing package libcurand-12-9 (--configure):
 dependency problems - leaving unconfigured
Setting up cuda-gdb-12-9 (12.9.79-1) ...
No apport report written because MaxReports is reached already
                                                              dpkg: dependency p
roblems prevent configuration of cuda-compiler-12-9:
 cuda-compiler-12-9 depends on cuda-nvcc-12-9 (>= 12.9.86); however:
  Package cuda-nvcc-12-9 is not configured yet.

dpkg: error processing package cuda-compiler-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              dpkg: dependency p
roblems prevent configuration of cuda-libraries-dev-12-9:
 cuda-libraries-dev-12-9 depends on cuda-cudart-dev-12-9 (>= 12.9.79); however:
  Package cuda-cudart-dev-12-9 is not configured yet.
 cuda-libraries-dev-12-9 depends on libcufft-dev-12-9 (>= 11.4.1.4); however:
  Package libcufft-dev-12-9 is not configured yet.
 cuda-libraries-dev-12-9 depends on libcufile-dev-12-9 (>= 1.14.1.1); however:
  Package libcufile-dev-12-9 is not configured yet.
 cuda-libraries-dev-12-9 depends on libcusolver-dev-12-9 (>= 11.7.5.82); however
:
  Package libcusolver-dev-12-9 is not configured yet.
 cuda-libraries-dev-12-9 depends on libnpp-dev-12-9 (>= 12.4.1.87); however:
  Package libnpp-dev-12-9 is not configured yet.
 cuda-libraries-dev-12-9 depends on libnvfatbin-dev-12-9 (>= 12.9.82); however:
  Package libnvfatbin-dev-12-9 is not configured yet.
 cuda-libraries-dev-12-9 depends on libnvjpeg-dev-12-9 (>= 12.4.0.76); however:
  Package libnvjpeg-dev-12-9 is not configured yet.

dpkg: error processing package cuda-libraries-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              dpkg: dependency p
roblems prevent configuration of cuda-libraries-12-9:
 cuda-libraries-12-9 depends on cuda-cudart-12-9 (>= 12.9.79); however:
  Package cuda-cudart-12-9 is not configured yet.
 cuda-libraries-12-9 depends on libcublas-12-9 (>= 12.9.1.4); however:
  Package libcublas-12-9 is not configured yet.
 cuda-libraries-12-9 depends on libcufft-12-9 (>= 11.4.1.4); however:
  Package libcufft-12-9 is not configured yet.
 cuda-libraries-12-9 depends on libcufile-12-9 (>= 1.14.1.1); however:
  Package libcufile-12-9 is not configured yet.
 cuda-libraries-12-9 depends on libcurand-12-9 (>= 10.3.10.19); however:
  Package libcurand-12-9 is not configured yet.
 cuda-libraries-12-9 depends on libcusolver-12-9 (>= 11.7.5.82); however:
  Package libcusolver-12-9 is not configured yet.
 cuda-libraries-12-9 depends on libcusparse-12-9 (>= 12.5.10.65); however:
  Package libcusparse-12-9 is not configured yet.
 cuda-libraries-12-9 depends on libnpp-12-9 (>= 12.4.1.87); however:
  Package libnpp-12-9 is not configured yet.
 cuda-libraries-12-9 depends on libnvjitlink-12-9 (>= 12.9.86); however:
  Package libnvjitlink-12-9 is not configured yet.
 cuda-libraries-12-9 depends on libnvfatbin-12-9 (>= 12.9.82); however:
  Package libnvfatbin-12-9 is not configured yet.
 cuda-libraries-12-9 depends on libnvjpeg-12-9 (>= 12.4.0.76); however:
  Package libnvjpeg-12-9 is not configured yet.

dpkg: error processing package cuda-libraries-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              dpkg: dependency p
roblems prevent configuration of libnvjitlink-dev-12-9:
 libnvjitlink-dev-12-9 depends on libnvjitlink-12-9 (>= 12.9.86); however:
  Package libnvjitlink-12-9 is not configured yet.

dpkg: error processing package libnvjitlink-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of libcusparse-dev-12-9:
 libcusparse-dev-12-9 depends on libcusparse-12-9 (>= 12.5.10.65); however:
  Package libcusparse-12-9 is not configured yet.

dpkg: error processing package libcusparse-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              No apport report w
ritten because MaxReports is reached already
                                            Setting up cuda-nsight-systems-12-9 
(12.9.1-1) ...
dpkg: dependency problems prevent configuration of libcurand-dev-12-9:
 libcurand-dev-12-9 depends on libcurand-12-9 (>= 10.3.10.19); however:
  Package libcurand-12-9 is not configured yet.

dpkg: error processing package libcurand-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of libcublas-dev-12-9:
 libcublas-dev-12-9 depends on libcublas-12-9 (>= 12.9.1.4); however:
  Package libcublas-12-9 is not configured yet.

dpkg: error processing package libcublas-dev-12-9 (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of cuda-visual-tools-12-9:No app
ort report written because MaxReports is reached already
                                                        No apport report written
 because MaxReports is reached already

 cuda-visual-tools-12-9 depends on cuda-libraries-dev-12-9 (>= 12.9.1); however:
  Package cuda-libraries-dev-12-9 is not configured yet.

dpkg: error processing package cuda-visual-tools-12-9 (--configure):
 dependency problems - leaving unconfigured
dpkg: dependency problems prevent configuration of cuda-toolkit-12-9:
 cuda-toolkit-12-9 depends on cuda-compiler-12-9 (>= 12.9.1); however:
  Package cuda-compiler-12-9 is not configured yet.
 cuda-toolkit-12-9 depends on cuda-libraries-12-9 (>= 12.9.1); however:
  Package cuda-libraries-12-9 is not configured yet.
No apport report written because MaxReports is reached already
                                                               cuda-toolkit-12-9
 depends on cuda-libraries-dev-12-9 (>= 12.9.1); however:
  Package cuda-libraries-dev-12-9 is not configured yet.

dpkg: error processing package cuda-toolkit-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              Setting up cuda-co
mmand-line-tools-12-9 (12.9.1-1) ...
dpkg: dependency problems prevent configuration of cuda-tools-12-9:
 cuda-tools-12-9 depends on cuda-visual-tools-12-9 (>= 12.9.1); however:
  Package cuda-visual-tools-12-9 is not configured yet.

dpkg: error processing package cuda-tools-12-9 (--configure):
 dependency problems - leaving unconfigured
No apport report written because MaxReports is reached already
                                                              Processing trigger
s for mime-support (3.64ubuntu1) ...
Processing triggers for gnome-menus (3.36.0-1ubuntu1) ...
Processing triggers for libc-bin (2.31-0ubuntu9.14) ...
Processing triggers for man-db (2.9.1-1) ...
Processing triggers for dbus (1.12.16-2ubuntu2.3) ...
Processing triggers for desktop-file-utils (0.24-1ubuntu3) ...
Processing triggers for initramfs-tools (0.136ubuntu6.7) ...
Errors were encountered while processing:
 cuda-toolkit-12-9-config-common
 libnvfatbin-12-9
 libnvjpeg-12-9
 libcusolver-12-9
 libcufile-12-9
 libnvfatbin-dev-12-9
 libcusolver-dev-12-9
 cuda-cudart-12-9
 cuda-cudart-dev-12-9
 cuda-crt-12-9
 libcufft-12-9
 cuda-nvcc-12-9
 libcublas-12-9
 libcusparse-12-9
 libnvjpeg-dev-12-9
 libcufile-dev-12-9
 libcufft-dev-12-9
 libnpp-12-9
 libnpp-dev-12-9
 libnvjitlink-12-9
 libcurand-12-9
 cuda-compiler-12-9
 cuda-libraries-dev-12-9
 cuda-libraries-12-9
 libnvjitlink-dev-12-9
 libcusparse-dev-12-9
 libcurand-dev-12-9
 libcublas-dev-12-9
 cuda-visual-tools-12-9
 cuda-toolkit-12-9
 cuda-tools-12-9
E: Sub-process /usr/bin/dpkg returned an error code (1)
