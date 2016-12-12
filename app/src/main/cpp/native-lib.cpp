#include <jni.h>
#include <string>
#include "test/test.h"
extern "C"
jstring
Java_asus_fade_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    Test test;
    std::string hello = test.test();
    return env->NewStringUTF(hello.c_str());
}
