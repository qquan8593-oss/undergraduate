package org.example.mambatest;
import android.app.Activity;
import android.os.Bundle;
import android.widget.*;
import java.io.*;
import java.nio.*;
import java.util.*;
import ai.onnxruntime.*;

public class MainActivity extends Activity {
  TextView status;
  @Override public void onCreate(Bundle b) {
    super.onCreate(b);
    LinearLayout layout=new LinearLayout(this); layout.setOrientation(1); layout.setPadding(30,60,30,30);
    status=new TextView(this); status.setText("Mamba-YOLO FP32 部署测试\n固定输入 256×256，CPU 推理。\n测试合成输入，不评价识别准确率。");
    Button run=new Button(this); run.setText("运行模型测试");
    layout.addView(status); layout.addView(run); setContentView(layout);
    run.setOnClickListener(v->{run.setEnabled(false);status.setText("正在加载并运行，请稍候……");new Thread(()->{
      String message;
      try {
        File model=new File(getFilesDir(),"model.onnx");
        try(InputStream in=getAssets().open("model.onnx");OutputStream out=new FileOutputStream(model)) {byte[] buf=new byte[65536];int n;while((n=in.read(buf))!=-1)out.write(buf,0,n);}
        OrtEnvironment env=OrtEnvironment.getEnvironment();
        try(OrtSession.SessionOptions options=new OrtSession.SessionOptions()) {
          options.setIntraOpNumThreads(4);
          try(OrtSession session=env.createSession(model.getAbsolutePath(),options)) {
            float[] input=new float[3*256*256];Arrays.fill(input,0.5f);
            try(OnnxTensor tensor=OnnxTensor.createTensor(env,FloatBuffer.wrap(input),new long[]{1,3,256,256})) {
              long start=System.nanoTime();
              try(OrtSession.Result result=session.run(Collections.singletonMap("images",tensor))) {
                double ms=(System.nanoTime()-start)/1e6;
                FloatBuffer values=((OnnxTensor)result.get(0)).getFloatBuffer();int invalid=0;
                while(values.hasRemaining()){float f=values.get();if(Float.isNaN(f)||Float.isInfinite(f))invalid++;}
                message="推理完成\n首次推理耗时："+String.format(Locale.US,"%.1f",ms)+" ms\n无效数值："+invalid+"\n仅验证部署，不代表检测精度或稳定帧率。";
              }
            }
          }
        }
      } catch(Throwable e) {message="测试失败："+e.toString();}
      final String text=message;runOnUiThread(()->{status.setText(text);run.setEnabled(true);});
    }).start();});
  }
}
