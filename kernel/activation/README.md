# Elementwise

## 环境
Hardware: 
  H100 PCIe  
  A100 SXM4-80G  
Docker image: nvcr.io/nvidia/pytorch:23.05-py3  
CUDA: 12.2  
torch: 2.0.0  

## 测试脚本

```bash
# 当前默认编译Hooper架构, 按需调增，如需精度校验, 可加上命令行参数
# 后续完善脚本 
bash run.sh
```

A100输出
```
--------------------------------------------------
                    S=1, K=1024
      out_f32_th: perf only, time:0.02369118ms, 
         out_f32: test pass, time:0.00231934ms, 
       out_f32x4: test pass, time:0.00241446ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.02353835ms, 
         out_f16: test pass, time:0.00239468ms, 
       out_f16x2: test pass, time:0.00255537ms, 
       out_f16x8: test pass, time:0.00271749ms, 
   out_f16x8pack: test pass, time:0.00270915ms, 
--------------------------------------------------
--------------------------------------------------
                    S=1, K=2048
      out_f32_th: perf only, time:0.02414918ms, 
         out_f32: test pass, time:0.00245762ms, 
       out_f32x4: test pass, time:0.00238371ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.02350354ms, 
         out_f16: test pass, time:0.00247622ms, 
       out_f16x2: test pass, time:0.00253820ms, 
       out_f16x8: test pass, time:0.00280976ms, 
   out_f16x8pack: test pass, time:0.00274181ms, 
--------------------------------------------------
--------------------------------------------------
                    S=1, K=4096
      out_f32_th: perf only, time:0.02383089ms, 
         out_f32: test pass, time:0.00245357ms, 
       out_f32x4: test pass, time:0.00254202ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.02363825ms, 
         out_f16: test pass, time:0.00247288ms, 
       out_f16x2: test pass, time:0.00254965ms, 
       out_f16x8: test pass, time:0.00297117ms, 
   out_f16x8pack: test pass, time:0.00287151ms, 
--------------------------------------------------
--------------------------------------------------
                    S=1024, K=1024
      out_f32_th: perf only, time:0.02334857ms, 
         out_f32: test pass, time:0.00483632ms, 
       out_f32x4: test pass, time:0.00381374ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.02341890ms, 
         out_f16: test pass, time:0.00520635ms, 
       out_f16x2: test pass, time:0.00408673ms, 
       out_f16x8: test pass, time:0.00401044ms, 
   out_f16x8pack: test pass, time:0.00393009ms, 
--------------------------------------------------
--------------------------------------------------
                    S=1024, K=2048
      out_f32_th: perf only, time:0.03148246ms, 
         out_f32: test pass, time:0.00762510ms, 
       out_f32x4: test pass, time:0.00512433ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.02451229ms, 
         out_f16: test pass, time:0.00791097ms, 
       out_f16x2: test pass, time:0.00634217ms, 
       out_f16x8: test pass, time:0.00511646ms, 
   out_f16x8pack: test pass, time:0.00495529ms, 
--------------------------------------------------
--------------------------------------------------
                    S=1024, K=4096
      out_f32_th: perf only, time:0.07406688ms, 
         out_f32: test pass, time:0.01499152ms, 
       out_f32x4: test pass, time:0.00969267ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.04676485ms, 
         out_f16: test pass, time:0.01318312ms, 
       out_f16x2: test pass, time:0.01317978ms, 
       out_f16x8: test pass, time:0.00757933ms, 
   out_f16x8pack: test pass, time:0.00702381ms, 
--------------------------------------------------
--------------------------------------------------
                    S=2048, K=1024
      out_f32_th: perf only, time:0.03145838ms, 
         out_f32: test pass, time:0.00744820ms, 
       out_f32x4: test pass, time:0.00511146ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.02445197ms, 
         out_f16: test pass, time:0.00800967ms, 
       out_f16x2: test pass, time:0.00569057ms, 
       out_f16x8: test pass, time:0.00504351ms, 
   out_f16x8pack: test pass, time:0.00490880ms, 
--------------------------------------------------
--------------------------------------------------
                    S=2048, K=2048
      out_f32_th: perf only, time:0.07453775ms, 
         out_f32: test pass, time:0.01448512ms, 
       out_f32x4: test pass, time:0.00955558ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.04160500ms, 
         out_f16: test pass, time:0.01317167ms, 
       out_f16x2: test pass, time:0.01047540ms, 
       out_f16x8: test pass, time:0.00724435ms, 
   out_f16x8pack: test pass, time:0.00692987ms, 
--------------------------------------------------
--------------------------------------------------
                    S=2048, K=4096
      out_f32_th: perf only, time:0.20974398ms, 
         out_f32: test pass, time:0.04404306ms, 
       out_f32x4: test pass, time:0.04154253ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.08604622ms, 
         out_f16: test pass, time:0.02637744ms, 
       out_f16x2: test pass, time:0.02425718ms, 
       out_f16x8: test pass, time:0.01286912ms, 
   out_f16x8pack: test pass, time:0.01213431ms, 
--------------------------------------------------
--------------------------------------------------
                    S=4096, K=1024
      out_f32_th: perf only, time:0.07605100ms, 
         out_f32: test pass, time:0.01662540ms, 
       out_f32x4: test pass, time:0.00939918ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.04159880ms, 
         out_f16: test pass, time:0.01381445ms, 
       out_f16x2: test pass, time:0.00888658ms, 
       out_f16x8: test pass, time:0.00718212ms, 
   out_f16x8pack: test pass, time:0.00688124ms, 
--------------------------------------------------
--------------------------------------------------
                    S=4096, K=2048
      out_f32_th: perf only, time:0.20976567ms, 
         out_f32: test pass, time:0.04403281ms, 
       out_f32x4: test pass, time:0.04150534ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.08607745ms, 
         out_f16: test pass, time:0.02697206ms, 
       out_f16x2: test pass, time:0.02062273ms, 
       out_f16x8: test pass, time:0.01260853ms, 
   out_f16x8pack: test pass, time:0.01194906ms, 
--------------------------------------------------
--------------------------------------------------
                    S=4096, K=4096
      out_f32_th: perf only, time:0.40136003ms, 
         out_f32: test pass, time:0.08452272ms, 
       out_f32x4: test pass, time:0.07821798ms, 
--------------------------------------------------
      out_f16_th: perf only, time:0.22115564ms, 
         out_f16: test pass, time:0.06870222ms, 
       out_f16x2: test pass, time:0.04603434ms, 
       out_f16x8: test pass, time:0.04201102ms, 
   out_f16x8pack: test pass, time:0.04206157ms, 
--------------------------------------------------
```