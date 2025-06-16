# Elementwise

## 环境
Hardware: H100 PCIe
Docker image: nvcr.io/nvidia/pytorch:23.05-py3
CUDA: 12.2
torch: 2.0.0

## 测试脚本

```bash
# 当前默认编译Hooper架构, 按需调增，如需精度校验, 可加上命令行参数
# 后续完善脚本
bash run.sh
```

H100输出
```
--------------------------------------------------
                S=1, K=1024
out_f32_torch: perf only, time:0.00341058ms, 
     out_f32: test pass, time:0.00238609ms, 
   out_f32x4: test pass, time:0.00235605ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.00336981ms, 
     out_f16: test pass, time:0.00238943ms, 
   out_f16x2: test pass, time:0.00234962ms, 
   out_f16x8: test pass, time:0.00236797ms, 
out_f16x8pack: test pass, time:0.00233698ms, 
--------------------------------------------------
--------------------------------------------------
                S=1, K=2048
out_f32_torch: perf only, time:0.00341225ms, 
     out_f32: test pass, time:0.00245380ms, 
   out_f32x4: test pass, time:0.00242758ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.00337815ms, 
     out_f16: test pass, time:0.00245714ms, 
   out_f16x2: test pass, time:0.00238204ms, 
   out_f16x8: test pass, time:0.00248861ms, 
out_f16x8pack: test pass, time:0.00233555ms, 
--------------------------------------------------
--------------------------------------------------
                S=1, K=4096
out_f32_torch: perf only, time:0.00334454ms, 
     out_f32: test pass, time:0.00245619ms, 
   out_f32x4: test pass, time:0.00266910ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.00334501ms, 
     out_f16: test pass, time:0.00245571ms, 
   out_f16x2: test pass, time:0.00245929ms, 
   out_f16x8: test pass, time:0.00270653ms, 
out_f16x8pack: test pass, time:0.00241470ms, 
--------------------------------------------------
--------------------------------------------------
                S=1024, K=1024
out_f32_torch: perf only, time:0.00389838ms, 
     out_f32: test pass, time:0.00510073ms, 
   out_f32x4: test pass, time:0.00402999ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.00357676ms, 
     out_f16: test pass, time:0.00507569ms, 
   out_f16x2: test pass, time:0.00353575ms, 
   out_f16x8: test pass, time:0.00391459ms, 
out_f16x8pack: test pass, time:0.00335193ms, 
--------------------------------------------------
--------------------------------------------------
                S=1024, K=2048
out_f32_torch: perf only, time:0.00562215ms, 
     out_f32: test pass, time:0.00785375ms, 
   out_f32x4: test pass, time:0.00580502ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.00492954ms, 
     out_f16: test pass, time:0.00760341ms, 
   out_f16x2: test pass, time:0.00511193ms, 
   out_f16x8: test pass, time:0.00524664ms, 
out_f16x8pack: test pass, time:0.00414419ms, 
--------------------------------------------------
--------------------------------------------------
                S=1024, K=4096
out_f32_torch: perf only, time:0.02949786ms, 
     out_f32: test pass, time:0.03317857ms, 
   out_f32x4: test pass, time:0.02997804ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.00762296ms, 
     out_f16: test pass, time:0.01314354ms, 
   out_f16x2: test pass, time:0.01307821ms, 
   out_f16x8: test pass, time:0.00989723ms, 
out_f16x8pack: test pass, time:0.00586104ms, 
--------------------------------------------------
--------------------------------------------------
                S=2048, K=1024
out_f32_torch: perf only, time:0.00559092ms, 
     out_f32: test pass, time:0.00833440ms, 
   out_f32x4: test pass, time:0.00584459ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.00494170ms, 
     out_f16: test pass, time:0.00771236ms, 
   out_f16x2: test pass, time:0.00475264ms, 
   out_f16x8: test pass, time:0.00505304ms, 
out_f16x8pack: test pass, time:0.00412250ms, 
--------------------------------------------------
--------------------------------------------------
                S=2048, K=2048
out_f32_torch: perf only, time:0.03000808ms, 
     out_f32: test pass, time:0.03323030ms, 
   out_f32x4: test pass, time:0.03017473ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.00772047ms, 
     out_f16: test pass, time:0.01344585ms, 
   out_f16x2: test pass, time:0.00859475ms, 
   out_f16x8: test pass, time:0.00884628ms, 
out_f16x8pack: test pass, time:0.00606966ms, 
--------------------------------------------------
--------------------------------------------------
                S=2048, K=4096
out_f32_torch: perf only, time:0.05841351ms, 
     out_f32: test pass, time:0.06206322ms, 
   out_f32x4: test pass, time:0.05817080ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.03015709ms, 
     out_f16: test pass, time:0.03816795ms, 
   out_f16x2: test pass, time:0.03311801ms, 
   out_f16x8: test pass, time:0.03023982ms, 
out_f16x8pack: test pass, time:0.03011894ms, 
--------------------------------------------------
--------------------------------------------------
                S=4096, K=1024
out_f32_torch: perf only, time:0.02950215ms, 
     out_f32: test pass, time:0.03306103ms, 
   out_f32x4: test pass, time:0.02963662ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.00760317ms, 
     out_f16: test pass, time:0.01353121ms, 
   out_f16x2: test pass, time:0.00755358ms, 
   out_f16x8: test pass, time:0.00729251ms, 
out_f16x8pack: test pass, time:0.00583339ms, 
--------------------------------------------------
--------------------------------------------------
                S=4096, K=2048
out_f32_torch: perf only, time:0.05841112ms, 
     out_f32: test pass, time:0.06201553ms, 
   out_f32x4: test pass, time:0.05850029ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.03020716ms, 
     out_f16: test pass, time:0.03771353ms, 
   out_f16x2: test pass, time:0.03360677ms, 
   out_f16x8: test pass, time:0.03022838ms, 
out_f16x8pack: test pass, time:0.03011036ms, 
--------------------------------------------------
--------------------------------------------------
                S=4096, K=4096
out_f32_torch: perf only, time:0.11065626ms, 
     out_f32: test pass, time:0.11884356ms, 
   out_f32x4: test pass, time:0.11057448ms, 
--------------------------------------------------
out_f16_torch: perf only, time:0.05813980ms, 
     out_f16: test pass, time:0.07307720ms, 
   out_f16x2: test pass, time:0.06161022ms, 
   out_f16x8: test pass, time:0.05811405ms, 
out_f16x8pack: test pass, time:0.05860877ms, 
--------------------------------------------------
```