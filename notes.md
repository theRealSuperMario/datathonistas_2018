## Densepose and Detectron
* Caffe and pytorch are installed in the `py2caffe` envrionment on the dlvm-team08 vm.
* The fb detectron zoo is also setup there
* For densepose a dockercontainer `densepose` is avaialbe. One has to mount the vm-local densepose/DensePoseData folder into the docker container: (`sudo nvidia-docker run -v ~/densepose/DensePoseData:/denseposedata -it densepose:wd bash
`) (And potentially symlink the folder into the right location from within the container -> [`ln -s /denseposedata DensePposeData`])

# Potential crap data
* Kid 0043 with 5.8 cm heigt
