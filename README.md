build instructions:
  1. bash scripts/configure.sh 
  2. run/profile hip: bash scripts/hip_build_and_run.sh / bash scripts/hip_build_and_profile.sh
  3. run/profile cuda: bash scripts/hip_build_and_run.sh / bash scripts/hip_build_and_profile.sh


tests
  1. mem_transfers.cpp, for comparing speed of device to host memcpy of small amount of bytes 
  2. pass_writes.cpp

 last test is related to integrator_shade_shadow ([intern/cycles/kernel/integrator/shade_shadow.h](https://github.com/blender/blender/blob/582ea0eb406bfac3fbeb9b1e312cffea6751c2a1/intern/cycles/kernel/integrator/shade_shadow.h#L153)). it is 3 times slower for hip. I see that it happens because of film_write_direct_light (https://github.com/blender/blender/blob/main/intern/cycles/kernel/film/light_passes.h#L461) 
this function just writes values to passes with atomicAdd

