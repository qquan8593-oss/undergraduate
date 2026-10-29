"""Build a small ARM64 test APK using an existing Android SDK/JDK."""
from pathlib import Path
import os, subprocess, zipfile, urllib.request, shutil
ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'build'
BUILD.mkdir(exist_ok=True)
SDK=Path('D:/SDK'); JAVA=Path('D:/Android Studio/jbr/bin')
BT=SDK/'build-tools/36.0.0'; PLATFORM=SDK/'platforms/android-37.0/android.jar'
env=os.environ.copy();env['JAVA_HOME']=str(JAVA.parent)
def run(*args):
    subprocess.run([str(a) for a in args],check=True,env=env)
aar=BUILD/'onnxruntime.aar'
if not aar.exists():
    urllib.request.urlretrieve('https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.18.0/onnxruntime-android-1.18.0.aar',aar)
with zipfile.ZipFile(aar) as z:
    z.extract('classes.jar',BUILD)
    z.extract('jni/arm64-v8a/libonnxruntime.so',BUILD)
    z.extract('jni/arm64-v8a/libonnxruntime4j_jni.so',BUILD)
classes=BUILD/'classes';classes.mkdir(exist_ok=True)
run(JAVA/'javac.exe','-encoding','UTF-8','-source','8','-target','8','-classpath',str(PLATFORM)+';'+str(BUILD/'classes.jar'),'-d',classes,ROOT/'MainActivity.java')
dex=BUILD/'dex';dex.mkdir(exist_ok=True)
run(JAVA/'java.exe','-cp',BT/'lib/d8.jar','com.android.tools.r8.D8','--min-api','26','--lib',PLATFORM,'--output',dex,BUILD/'classes.jar',*classes.rglob('*.class'))
unsigned=BUILD/'unsigned.apk'
run(BT/'aapt2.exe','link','-o',unsigned,'--manifest',ROOT/'AndroidManifest.xml','-I',PLATFORM)
with zipfile.ZipFile(unsigned,'a',compression=zipfile.ZIP_DEFLATED) as z:
    z.write(dex/'classes.dex','classes.dex')
    z.write(ROOT.parent.parent/'exports/android/model.onnx','assets/model.onnx')
    for f in (BUILD/'jni/arm64-v8a').glob('*.so'):z.write(f,'lib/arm64-v8a/'+f.name)
aligned=BUILD/'aligned.apk'
run(BT/'zipalign.exe','-f','4',unsigned,aligned)
key=BUILD/'test.keystore'
if not key.exists():
    run(JAVA/'keytool.exe','-genkeypair','-keystore',key,'-storepass','android','-keypass','android','-alias','test','-dname','CN=Mamba Test','-keyalg','RSA','-validity','10000')
apk=ROOT/'mamba-test.apk'
run(JAVA/'java.exe','-jar',BT/'lib/apksigner.jar','sign','--ks',key,'--ks-pass','pass:android','--out',apk,aligned)
run(JAVA/'java.exe','-jar',BT/'lib/apksigner.jar','verify',apk)
print('APK:',apk)
