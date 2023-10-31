package com.pas.webcam;

import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.preference.ListPreference;
import android.preference.PreferenceCategory;
import android.preference.PreferenceScreen;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.EditText;
import android.widget.Toast;
import com.pas.uied.UiEditor;
import com.pas.webcam.configpages.CloudStreamingConfiguration;
import com.pas.webcam.configpages.IPWPreferenceBase;
import com.pas.webcam.configpages.MotionDetection;
import com.pas.webcam.configpages.OnvifConfiguration;
import com.pas.webcam.configpages.OverlayConfiguration;
import com.pas.webcam.configpages.PermissionsConfiguration;
import com.pas.webcam.configpages.PowerConfiguration;
import com.pas.webcam.configpages.SensorConfiguration;
import com.pas.webcam.pro.R;
import com.pas.webcam.utils.MyDialogPreference;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Date;
import java.util.Random;
import p008d.p009a.p010k.C0375p;
import p008d.p026e.p027d.C0547a;
import p008d.p026e.p028e.C0567a;
import p067e.p068a.p069a.p070a.C0802a;
import p067e.p157e.p159b.C1845a;
import p067e.p157e.p159b.C1852f;
import p067e.p157e.p159b.C1855g;
import p067e.p157e.p159b.C1856h;
import p067e.p157e.p160c.C1858b;
import p067e.p157e.p167g.C1937f;
import p067e.p157e.p167g.C1940g;
import p067e.p157e.p167g.C1943h;
import p067e.p157e.p167g.C1945i;
import p067e.p157e.p167g.C1954j;
import p067e.p157e.p167g.C1965k;
import p067e.p157e.p167g.C2021l;
import p067e.p157e.p167g.C2045m;
import p067e.p157e.p167g.C2051n;
import p067e.p157e.p167g.C2218o;
import p067e.p157e.p167g.C2220q;
import p067e.p157e.p167g.p169j0.C1962g;
import p067e.p157e.p167g.p171l0.C2034g;
import p067e.p157e.p167g.p173n0.C2082b0;
import p067e.p157e.p167g.p173n0.C2096e0;
import p067e.p157e.p167g.p173n0.C2105h;
import p067e.p157e.p167g.p173n0.C2176l0;
import p067e.p157e.p167g.p173n0.C2186n;
import p067e.p157e.p167g.p173n0.C2192p;
import p067e.p157e.p167g.p173n0.C2216z;
import p193j.p194a.p195a.p198b.C2427c;

public class Configuration extends IPWPreferenceBase implements C0547a.C0549b {

   
    public final C1855g<Integer> f1199b = C1856h.m3585c();

   
    public final C1855g<Integer> f1200c = C1856h.m3584b();

   
    public C1852f f1201d;

   
    public ListPreference f1202e;

   
    public PreferenceScreen f1203f = null;

   
    public PreferenceScreen f1204g = null;

   
    public PreferenceCategory f1205h = null;

   
    public PreferenceScreen f1206i = null;

   
    public C2082b0 f1207j = new C2082b0(C2082b0.f5873f, 96);

   
    public String f1208k = "";

   
    public class C0219a implements Runnable {

       
        public final Context f1209b;

       
        public class C0220a implements Runnable {

           
            public class C0221a implements Runnable {
                public C0221a() {
                }

                public void run() {
                    new C2105h(C0219a.this.f1209b).mo8520j();
                    Configuration.this.startActivityForResult(new Intent().setAction("android.intent.action.MAIN").setClass(C0219a.this.f1209b, Rolling.class), -1);
                }
            }

            public C0220a() {
            }

            public void run() {
                Configuration configuration = Configuration.this;
                configuration.f1207j.mo8492a(configuration, new C0221a(), (Runnable) null);
            }
        }

        public C0219a(Context context) {
            this.f1209b = context;
        }

        public void run() {
            Context context = this.f1209b;
            if (Build.VERSION.SDK_INT >= 29) {
                try {
                    C2427c.m4852c(C0567a.m1463g(context, "tmp_videos")[0]);
                } catch (RuntimeException e) {
                    C0375p.m757E0(e);
                }
            }
            Context context2 = this.f1209b;
            boolean i = C2192p.m4235i(C2192p.C2194b.HttpsForceRegen);
            C0220a aVar = new C0220a();
            File dir = context2.getDir("Cert", 0);
            File file = new File(dir, "private.pem");
            File file2 = new File(dir, "public.cer");
            if (!file.exists() || !file2.exists() || i) {
                C2186n nVar = new C2186n(context2, R.string.generating_https_cert);
                nVar.f6169f = i;
                nVar.f6170g = aVar;
                nVar.f6172i = file;
                nVar.f6171h = file2;
                nVar.execute(new Void[0]);
                return;
            }
            Log.i("CertGen", "Reusing existing cert");
            aVar.run();
        }
    }

   
    public class C0222b implements DialogInterface.OnClickListener {
        public C0222b(Configuration configuration) {
        }

        public void onClick(DialogInterface dialogInterface, int i) {
            App.f1197c = null;
        }
    }

   
    public class C0223c implements DialogInterface.OnClickListener {

       
        public final Context f1213b;

        public C0223c(Context context) {
            this.f1213b = context;
        }

        public void onClick(DialogInterface dialogInterface, int i) {
            Configuration.this.mo5083l(this.f1213b);
        }
    }

   
    public class C0224d implements DialogInterface.OnClickListener {

       
        public final EditText f1215b;

       
        public class C0225a implements C2176l0.C2182f {
            public C0225a(C0224d dVar) {
            }

           
            public void mo5095a(boolean z) {
            }
        }

        public C0224d(Configuration configuration, EditText editText) {
            this.f1215b = editText;
        }

        public void onClick(DialogInterface dialogInterface, int i) {
            C2176l0.m4186b(this.f1215b.getText().toString(), C2176l0.C2181e.USER_INPUT, new C0225a(this));
        }
    }

   
    public class C0226e extends AsyncTask<C1858b, Void, Exception> {

       
        public final Context f1216a;

       
        public ProgressDialog f1217b;

        public C0226e(Context context) {
            this.f1216a = context;
        }

        public Object doInBackground(Object[] objArr) {
            try {
                ((C1858b[]) objArr)[0].mo8211d();
                App.f1197c = null;
                return null;
            } catch (Exception e) {
                Log.e("IPWebcam", "Cannot send email", e);
                return e;
            }
        }

        public void onPostExecute(Object obj) {
            Exception exc = (Exception) obj;
            super.onPostExecute(exc);
            this.f1217b.dismiss();
            if (exc != null) {
                AlertDialog.Builder builder = new AlertDialog.Builder(this.f1216a);
                builder.setMessage(Configuration.this.getString(R.string.failed_to_send_report_for_reason) + "\n\n" + exc.toString()).setPositiveButton(R.string.yes, new C2051n(this)).setNegativeButton(R.string.no, new C2045m(this)).show();
            }
        }

        public void onPreExecute() {
            super.onPreExecute();
            ProgressDialog progressDialog = new ProgressDialog(this.f1216a);
            this.f1217b = progressDialog;
            progressDialog.setTitle(R.string.processing_request);
            this.f1217b.setCancelable(false);
            this.f1217b.setIndeterminate(true);
            this.f1217b.show();
        }
    }

   
    public static class C0227f extends C1845a<Void, Void> {

       
        public Runnable f1219f;

        public C0227f(Context context, Runnable runnable) {
            super(context, R.string.init_thumb_migration);
            this.f1219f = runnable;
        }

        public Object doInBackground(Object[] objArr) {
            Void[] voidArr = (Void[]) objArr;
            C2034g gVar = (C2034g) Interop.getEndpoint(C1962g.class);
            if (gVar == null) {
                return null;
            }
            C2216z.m4277c(this.f5347c, gVar, new C2218o(this));
            return null;
        }

        public void onPostExecute(Object obj) {
            super.onPostExecute((Void) obj);
            this.f1219f.run();
        }
    }

   
    public void mo5083l(Context context) {
        byte[] byteArray = App.f1197c.toByteArray();
        StringBuilder d = C0802a.m2027d("Bugreport ");
        d.append(this.f1208k);
        d.append(" from ");
        d.append(new Date().toString());
        String sb = d.toString();
        C1858b bVar = new C1858b(C2192p.m4244r(C2192p.C2200h.SmtpServer), C2192p.m4241o(C2192p.C2198f.SmtpPort), C2192p.m4241o(C2192p.C2198f.SmtpEncryption));
        bVar.f5390e = sb;
        bVar.f5389d = sb;
        bVar.f5392g = C2192p.m4244r(C2192p.C2200h.SmtpTo);
        bVar.f5391f = C2192p.m4244r(C2192p.C2200h.SmtpFrom);
        bVar.f5393h = C2192p.m4244r(C2192p.C2200h.SmtpLogin);
        bVar.f5394i = C2192p.m4244r(C2192p.C2200h.SmtpPassword);
        bVar.f5396k.add(new C1858b.C1859a(bVar, new ByteArrayInputStream(byteArray), "logcat.txt"));
        new C0226e(context).execute(new C1858b[]{bVar});
    }

   
    public void mo5084m() {
        C2192p.m4248v(C2192p.C2194b.VideoFallbackToInternal, false);
        C0219a aVar = new C0219a(this);
        if (!C2096e0.m3931p(this)) {
            new C0227f(this, aVar).execute(new Void[0]);
        } else {
            aVar.run();
        }
    }

    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        C0375p.m755D0("Opened app");
        C2176l0.m4195k(this, false, R.string.app_name);
        if (Interop.apiCheck() != 6645643) {
            Toast.makeText(this, "API version incorrect", 1).show();
            finish();
            return;
        }
        C2192p.C2198f fVar = C2192p.C2198f.StartCount;
        this.f1201d = C1852f.m3561c(this, new Object[]{Integer.valueOf(R.string.audio_mode_enabled), Integer.valueOf(C2220q.C2221a.Enabled.f6475b), Integer.valueOf(R.string.audio_mode_disabled), Integer.valueOf(C2220q.C2221a.Disabled.f6475b), Integer.valueOf(R.string.audio_mode_audio_only), Integer.valueOf(C2220q.C2221a.AudioOnly.f6475b)}, new C1855g[]{this.f1199b, this.f1200c});
        PreferenceScreen createPreferenceScreen = getPreferenceManager().createPreferenceScreen(this);
        PreferenceCategory preferenceCategory = new PreferenceCategory(this);
        preferenceCategory.setTitle(R.string.plugins);
        createPreferenceScreen.addPreference(preferenceCategory);
        preferenceCategory.addPreference(mo5014f(R.string.plugins, R.string.scripts_desc, new C1937f(this, this)));
        PreferenceCategory preferenceCategory2 = new PreferenceCategory(this);
        preferenceCategory2.setTitle(R.string.ip_webcam_settings);
        createPreferenceScreen.addPreference(preferenceCategory2);
        preferenceCategory2.addPreference(mo5014f(R.string.video_preferences, R.string.video_settings_desc, new C1940g(this)));
        preferenceCategory2.addPreference(mo5013e(R.string.effects_title, R.string.effects_desc, new Intent().setAction("android.intent.action.MAIN").setClass(this, OverlayConfiguration.class), -1));
        preferenceCategory2.addPreference(mo5013e(R.string.power_mgmt, R.string.power_mgmt_desc, new Intent().setAction("android.intent.action.MAIN").setClass(this, PowerConfiguration.class), -1));
        preferenceCategory2.addPreference(mo5013e(R.string.motion_sound_detection, R.string.motion_detection_desc, new Intent().setAction("android.intent.action.MAIN").setClass(this, MotionDetection.class), -1));
        preferenceCategory2.addPreference(mo5013e(R.string.sensors, R.string.sensors_desc, new Intent().setAction("android.intent.action.MAIN").setClass(this, SensorConfiguration.class), -1));
        preferenceCategory2.addPreference(mo5013e(R.string.user_interface, R.string.customize_ui, new Intent().setAction("android.intent.action.MAIN").setClass(this, UiEditor.class), -1));
        PreferenceCategory preferenceCategory3 = new PreferenceCategory(this);
        preferenceCategory3.setTitle(R.string.connection_settings);
        createPreferenceScreen.addPreference(preferenceCategory3);
        PreferenceScreen f = mo5014f(R.string.local_broadcasting, -1, new C1943h(this, this));
        this.f1206i = f;
        preferenceCategory3.addPreference(f);
        PreferenceScreen e = mo5013e(R.string.onvif, R.string.onvif_desc, new Intent().setAction("android.intent.action.MAIN").setClass(this, OnvifConfiguration.class), -1);
        preferenceCategory3.addPreference(e);
        PreferenceScreen e2 = mo5013e(R.string.cloud_streaming, R.string.cloud_streaming_desc, new Intent().setAction("android.intent.action.MAIN").setClass(this, CloudStreamingConfiguration.class), -1);
        this.f1203f = e2;
        preferenceCategory3.addPreference(e2);
        Class<PreferenceScreen> cls = PreferenceScreen.class;
        try {
            Method method = cls.getMethod("setIcon", new Class[]{Integer.TYPE});
            if (this.f1203f != null) {
                method.invoke(this.f1203f, new Object[]{Integer.valueOf(R.drawable.icon_globe)});
            }
            method.invoke(this.f1206i, new Object[]{Integer.valueOf(R.drawable.ic_signal_wifi_statusbar_3_bar_white_26x24dp)});
            method.invoke(e, new Object[]{Integer.valueOf(R.drawable.onvif)});
        } catch (IllegalAccessException | NoSuchMethodException | InvocationTargetException unused) {
        }
        PreferenceCategory preferenceCategory4 = new PreferenceCategory(this);
        this.f1205h = preferenceCategory4;
        preferenceCategory4.setTitle(R.string.misc);
        createPreferenceScreen.addPreference(this.f1205h);
        String str = "android.intent.action.MAIN";
        ListPreference i = mo5017i(R.string.audiopref, -1, Integer.valueOf(C2192p.m4241o(C2192p.C2198f.AudioMode)), -1, (T[]) null, this.f1201d.mo8197m(this.f1199b), new C1945i(this));
        this.f1202e = i;
        this.f1205h.addPreference(i);
        C2192p.m4223A(fVar, C2192p.m4241o(fVar) + 1);
        C1855g<String> d = C1856h.m3586d();
        C1855g<String> d2 = C1856h.m3586d();
        C1852f c = C1852f.m3561c(this, new Object[]{getString(R.string.skype_faq_title), getString(R.string.skype_faq), getString(R.string.fps_faq_title), getString(R.string.fps_faq), getString(R.string.accessing_title), getString(R.string.accessing), getString(R.string.known_issues_title), getString(R.string.known_issues), getString(R.string.faq_another_title), getString(R.string.faq_another), getString(R.string.acknowledgements_title), getString(R.string.acknowledgements)}, new C1855g[]{d, d2});
        MyDialogPreference myDialogPreference = new MyDialogPreference(this, (AttributeSet) null, new AlertDialog.Builder(this).setTitle(R.string.faq).setItems((CharSequence[]) c.mo8193i(d), new C1954j(this, c, d2)).create());
        myDialogPreference.setTitle(R.string.faq);
        myDialogPreference.setSummary(R.string.including_one_for_impatient_skype_users);
        this.f1205h.addPreference(myDialogPreference);
        this.f1205h.addPreference(mo5014f(R.string.create_start_shortcut, R.string.start_from_homescreen, new C1965k(this, this)));
        PreferenceCategory preferenceCategory5 = new PreferenceCategory(this);
        preferenceCategory5.setTitle(R.string.service_control);
        PreferenceScreen preferenceScreen = createPreferenceScreen;
        preferenceScreen.addPreference(preferenceCategory5);
        if (PermissionsConfiguration.m675l(this)) {
            preferenceCategory5.addPreference(mo5013e(R.string.optional_permissions, R.string.optional_permissions_desc, new Intent().setAction(str).setClass(this, PermissionsConfiguration.class), -1));
        }
        preferenceCategory5.addPreference(mo5014f(R.string.start_server, R.string.begin_serving_video_stream, new C2021l(this)));
        mo5251k(preferenceScreen);
        if (App.f1197c != null) {
            this.f1208k = Integer.toHexString(new Random().nextInt());
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setMessage(getString(R.string.ipwebcam_crashed_send_report) + " " + this.f1208k).setPositiveButton(R.string.yes, new C0223c(this)).setNegativeButton(R.string.no, new C0222b(this)).show();
        }
    }

    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_options, menu);
        return true;
    }

    public boolean onOptionsItemSelected(MenuItem menuItem) {
        if (menuItem.getItemId() == R.id.cheats_menu) {
            EditText editText = new EditText(this);
            new AlertDialog.Builder(this).setMessage(R.string.enter_a_cheat).setView(editText).setPositiveButton(R.string.ok, new C0224d(this, editText)).setNegativeButton(R.string.cancel, (DialogInterface.OnClickListener) null).show();
            return true;
        } else if (menuItem.getItemId() == R.id.start_server_menu) {
            mo5084m();
            return true;
        } else {
            finish();
            return true;
        }
    }

    public void onRequestPermissionsResult(int i, String[] strArr, int[] iArr) {
        super.onRequestPermissionsResult(i, strArr, iArr);
        this.f1207j.onRequestPermissionsResult(i, strArr, iArr);
    }

    public void onResume() {
        String str;
        C2192p.C2194b bVar = C2192p.C2194b.IvideonEnabled;
        super.onResume();
        CharSequence[] k = this.f1201d.mo8195k(this.f1199b);
        if (!C2192p.m4235i(bVar)) {
            this.f1202e.setEntries(k);
        } else {
            this.f1202e.setEntries(new CharSequence[]{k[0], k[1]});
        }
        int o = C2192p.m4241o(C2192p.C2198f.AudioMode);
        this.f1202e.setValueIndex(o);
        this.f1202e.setSummary(this.f1201d.mo8196l(o, this.f1199b));
        PreferenceScreen preferenceScreen = this.f1203f;
        if (preferenceScreen != null) {
            StringBuilder sb = new StringBuilder();
            if (C2192p.m4235i(bVar)) {
                str = getString(R.string.connect_using_ivideon_desc_enabled);
            } else {
                str = getString(R.string.connect_using_ivideon_desc_disabled);
            }
            sb.append(str);
            sb.append(" ");
            sb.append(getString(R.string.connect_using_ivideon_desc));
            preferenceScreen.setSummary(sb.toString());
        }
        if (!(this.f1204g == null || this.f1205h == null || C2192p.m4235i(C2192p.C2194b.ShowBeware))) {
            this.f1205h.removePreference(this.f1204g);
            this.f1204g = null;
        }
        if (this.f1206i != null) {
            this.f1206i.setSummary(getString(R.string.local_broadcasting_desc).replace("$PORT", Integer.toString(C2192p.m4241o(C2192p.C2198f.Port))).replace("$LPW", getString("".equals(C2192p.m4244r(C2192p.C2200h.Login)) ? R.string.lpw_notset : R.string.lpw_set)));
        }
        if (Interop.f1220b.exists()) {
            try {
                String h = C2427c.m4857h(Interop.f1220b);
                Interop.f1220b.delete();
                if (!h.equals("")) {
                    new AlertDialog.Builder(this).setTitle(R.string.warn).setMessage(getString(R.string.native_error_happened).replace("$ERR", h)).setPositiveButton(R.string.ok, (DialogInterface.OnClickListener) null).create().show();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
