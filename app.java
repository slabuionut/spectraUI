package com.pas.webcam;

import android.content.Context;
import android.os.Build;
import android.provider.Settings;
import android.util.Log;
import androidx.multidex.MultiDexApplication;
import com.google.firebase.crashlytics.FirebaseCrashlytics;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.lang.Thread;
import p067e.p157e.p167g.C2219p;
import p067e.p157e.p167g.p171l0.C2032e;
import p067e.p157e.p167g.p171l0.C2033f;
import p067e.p157e.p167g.p171l0.C2034g;
import p067e.p157e.p167g.p171l0.C2037i;
import p067e.p157e.p167g.p171l0.C2040l;
import p067e.p157e.p167g.p173n0.C2192p;
import p193j.p194a.p195a.p198b.C2429e;

public class App extends MultiDexApplication {

   
    public static Context f1196b;

   
    public static ByteArrayOutputStream f1197c;

   
    public class C0218a implements Thread.UncaughtExceptionHandler {

       
        public final Thread.UncaughtExceptionHandler f1198a;

        public C0218a(App app, Thread.UncaughtExceptionHandler uncaughtExceptionHandler) {
            this.f1198a = uncaughtExceptionHandler;
        }

        public void uncaughtException(Thread thread, Throwable th) {
            if (thread.getName().startsWith("AdWorker")) {
                Log.w("ADMOB", "AdWorker thread thrown an exception.", th);
                return;
            }
            Thread.UncaughtExceptionHandler uncaughtExceptionHandler = this.f1198a;
            if (uncaughtExceptionHandler != null) {
                uncaughtExceptionHandler.uncaughtException(thread, th);
                return;
            }
            throw new RuntimeException("No default uncaught exception handler.", th);
        }
    }

    public void onCreate() {
        C2192p.C2194b bVar = C2192p.C2194b.StoppedSuccessfully;
        try {
            Class.forName("android.os.AsyncTask");
        } catch (Throwable unused) {
        }
        super.onCreate();
        f1196b = getApplicationContext();
        C2219p.f6465a = f1196b;
        if (C2192p.m4235i(C2192p.C2194b.CrashlyticsEnabled)) {
            FirebaseCrashlytics.getInstance().setCrashlyticsCollectionEnabled(true);
            String r = C2192p.m4244r(C2192p.C2200h.CrashUserId);
            if (!"".equals(r)) {
                FirebaseCrashlytics.getInstance().setUserId(r);
            }
        } else {
            FirebaseCrashlytics.getInstance().setCrashlyticsCollectionEnabled(false);
        }
        C2040l.f5737i = getResources();
        if (!"".equals(C2192p.m4244r(C2192p.C2200h.SmtpLogin)) && !C2192p.m4235i(bVar)) {
            C2192p.m4248v(bVar, true);
            f1197c = new ByteArrayOutputStream();
            String string = Settings.System.getString(f1196b.getContentResolver(), "android_id");
            try {
                ByteArrayOutputStream byteArrayOutputStream = f1197c;
                byteArrayOutputStream.write(("ID: " + string + "\n").getBytes());
                ByteArrayOutputStream byteArrayOutputStream2 = f1197c;
                byteArrayOutputStream2.write(("Model: " + C2192p.m4234h() + "\n").getBytes());
                ByteArrayOutputStream byteArrayOutputStream3 = f1197c;
                byteArrayOutputStream3.write(("API level:" + Build.VERSION.SDK_INT + "\n").getBytes());
                f1197c.write("App version:768\n".getBytes());
                ByteArrayOutputStream byteArrayOutputStream4 = f1197c;
                byteArrayOutputStream4.write(("Build:" + Build.VERSION.CODENAME + " " + Build.VERSION.INCREMENTAL + " " + Build.VERSION.RELEASE + "\n").getBytes());
                Process exec = Runtime.getRuntime().exec(new String[]{"logcat", "-v", "time", "-d"});
                C2429e.m4860c(exec.getInputStream(), f1197c);
                exec.destroy();
                f1197c.write("\nDevice uptime is: ".getBytes());
                C2429e.m4860c(Runtime.getRuntime().exec(new String[]{"cat", "/proc/uptime"}).getInputStream(), f1197c);
            } catch (Exception e) {
                if (f1197c == null) {
                    f1197c = new ByteArrayOutputStream();
                }
                try {
                    f1197c.write(e.toString().getBytes());
                } catch (IOException e2) {
                    e2.printStackTrace();
                }
            }
        }
        Interop.registerEndpoint(new C2034g(this));
        Interop.registerEndpoint(new C2037i());
        Interop.registerEndpoint(new C2032e());
        Interop.registerEndpoint(new C2033f());
        File cacheDir = C2219p.m4280a().getCacheDir();
        cacheDir.mkdirs();
        File file = new File(cacheDir, "lastError");
        Interop.f1220b = file;
        byte[] bytes = file.getAbsolutePath().getBytes();
        Interop.setErrorFile(bytes, bytes.length);
        Thread.setDefaultUncaughtExceptionHandler(new C0218a(this, Thread.getDefaultUncaughtExceptionHandler()));
        Log.v("App", "AplicatieCamera version 1");
    }
}
