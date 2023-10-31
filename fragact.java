package androidx.fragment.app;

import android.content.Context;
import android.content.Intent;
import android.content.IntentSender;
import android.content.res.Configuration;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import androidx.core.app.ComponentActivity;
import java.io.FileDescriptor;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.List;
import org.mozilla.javascript.Parser;
import p008d.p023c.C0528d;
import p008d.p023c.C0538i;
import p008d.p026e.p027d.C0547a;
import p008d.p050j.p051a.C0692d;
import p008d.p050j.p051a.C0693e;
import p008d.p050j.p051a.C0694f;
import p008d.p050j.p051a.C0695g;
import p008d.p050j.p051a.C0712k;
import p008d.p055l.C0739f;
import p008d.p055l.C0743h;
import p008d.p055l.C0753p;
import p008d.p055l.C0754q;
import p008d.p056m.p057a.C0755a;

public class FragmentActivity extends ComponentActivity implements C0754q, C0547a.C0549b, C0547a.C0551d {

   
    public final Handler f818c = new C0124a();

   
    public final C0692d f819d = new C0692d(new C0125b());

   
    public C0753p f820e;

   
    public boolean f821f;

   
    public boolean f822g;

   
    public boolean f823h = true;

   
    public boolean f824i;

   
    public boolean f825j;

   
    public boolean f826k;

   
    public int f827l;

   
    public C0538i<String> f828m;

   
    public class C0124a extends Handler {
        public C0124a() {
        }

        public void handleMessage(Message message) {
            if (message.what != 2) {
                super.handleMessage(message);
                return;
            }
            FragmentActivity.this.mo1284l();
            FragmentActivity.this.f819d.mo6739a();
        }
    }

   
    public class C0125b extends C0693e<FragmentActivity> {
        public C0125b() {
            super(FragmentActivity.this);
        }

       
        public View mo1277b(int i) {
            return FragmentActivity.this.findViewById(i);
        }

       
        public boolean mo1278c() {
            Window window = FragmentActivity.this.getWindow();
            return (window == null || window.peekDecorView() == null) ? false : true;
        }

       
        public void mo1305d(Fragment fragment, Intent intent, int i, Bundle bundle) {
            FragmentActivity fragmentActivity = FragmentActivity.this;
            fragmentActivity.f826k = true;
            if (i == -1) {
                try {
                    C0547a.m1440m(fragmentActivity, intent, -1, bundle);
                } catch (Throwable th) {
                    fragmentActivity.f826k = false;
                    throw th;
                }
            } else {
                FragmentActivity.m403h(i);
                C0547a.m1440m(fragmentActivity, intent, ((fragmentActivity.mo1281g(fragment) + 1) << 16) + (i & Parser.CLEAR_TI_MASK), bundle);
            }
            fragmentActivity.f826k = false;
        }
    }

   
    public static final class C0126c {

       
        public C0753p f831a;

       
        public C0712k f832b;
    }

   
    public static void m403h(int i) {
        if ((i & -65536) != 0) {
            throw new IllegalArgumentException("Can only use lower 16 bits for requestCode");
        }
    }

   
    public static boolean m404j(C0694f fVar, C0739f.C0741b bVar) {
        List<Fragment> list;
        C0695g gVar = (C0695g) fVar;
        if (gVar.f2790e.isEmpty()) {
            list = Collections.emptyList();
        } else {
            synchronized (gVar.f2790e) {
                list = (List) gVar.f2790e.clone();
            }
        }
        boolean z = false;
        for (Fragment fragment : list) {
            if (fragment != null) {
                if (((C0743h) fragment.getLifecycle()).f2922b.compareTo(C0739f.C0741b.STARTED) >= 0) {
                    fragment.mLifecycleRegistry.mo6869d(bVar);
                    z = true;
                }
                C0694f peekChildFragmentManager = fragment.peekChildFragmentManager();
                if (peekChildFragmentManager != null) {
                    z |= m404j(peekChildFragmentManager, bVar);
                }
            }
        }
        return z;
    }

   
    public final void mo1279a(int i) {
        if (!this.f824i && i != -1) {
            m403h(i);
        }
    }

    public void dump(String str, FileDescriptor fileDescriptor, PrintWriter printWriter, String[] strArr) {
        super.dump(str, fileDescriptor, printWriter, strArr);
        printWriter.print(str);
        printWriter.print("Local FragmentActivity ");
        printWriter.print(Integer.toHexString(System.identityHashCode(this)));
        printWriter.println(" State:");
        String str2 = str + "  ";
        printWriter.print(str2);
        printWriter.print("mCreated=");
        printWriter.print(this.f821f);
        printWriter.print(" mResumed=");
        printWriter.print(this.f822g);
        printWriter.print(" mStopped=");
        printWriter.print(this.f823h);
        if (getApplication() != null) {
            C0755a.m1947b(this).mo6887a(str2, fileDescriptor, printWriter, strArr);
        }
        this.f819d.f2775a.f2779d.mo6741a(str, fileDescriptor, printWriter, strArr);
    }

   
    public final int mo1281g(Fragment fragment) {
        if (this.f828m.mo6409i() < 65534) {
            while (true) {
                C0538i<String> iVar = this.f828m;
                int i = this.f827l;
                if (iVar.f2361b) {
                    iVar.mo6402c();
                }
                if (C0528d.m1386a(iVar.f2362c, iVar.f2364e, i) >= 0) {
                    this.f827l = (this.f827l + 1) % 65534;
                } else {
                    int i2 = this.f827l;
                    this.f828m.mo6407g(i2, fragment.mWho);
                    this.f827l = (this.f827l + 1) % 65534;
                    return i2;
                }
            }
        } else {
            throw new IllegalStateException("Too many pending Fragment activity results.");
        }
    }

    public C0739f getLifecycle() {
        return this.f646b;
    }

    public C0753p getViewModelStore() {
        if (getApplication() != null) {
            if (this.f820e == null) {
                C0126c cVar = (C0126c) getLastNonConfigurationInstance();
                if (cVar != null) {
                    this.f820e = cVar.f831a;
                }
                if (this.f820e == null) {
                    this.f820e = new C0753p();
                }
            }
            return this.f820e;
        }
        throw new IllegalStateException("Your activity is not yet attached to the Application instance. You can't request ViewModel before onCreate call.");
    }

   
    public C0694f mo1282i() {
        return this.f819d.f2775a.f2779d;
    }

   
    public void mo1283k() {
    }

   
    public void mo1284l() {
        this.f819d.f2775a.f2779d.mo6757M();
    }

    @Deprecated
   
    public void mo38m() {
        invalidateOptionsMenu();
    }

    public void onActivityResult(int i, int i2, Intent intent) {
        this.f819d.mo6740b();
        int i3 = i >> 16;
        if (i3 != 0) {
            int i4 = i3 - 1;
            String d = this.f828m.mo6404d(i4);
            this.f828m.mo6408h(i4);
            if (d == null) {
                Log.w("FragmentActivity", "Activity result delivered for unknown Fragment.");
                return;
            }
            Fragment W = this.f819d.f2775a.f2779d.mo6767W(d);
            if (W == null) {
                Log.w("FragmentActivity", "Activity result no fragment exists for who: " + d);
                return;
            }
            W.onActivityResult(i & Parser.CLEAR_TI_MASK, i2, intent);
            return;
        }
        C0547a.m1437j();
        super.onActivityResult(i, i2, intent);
    }

    public void onBackPressed() {
        C0695g gVar = this.f819d.f2775a.f2779d;
        boolean c = gVar.mo6743c();
        if (c && Build.VERSION.SDK_INT <= 25) {
            return;
        }
        if (c || !gVar.mo6744d()) {
            super.onBackPressed();
        }
    }

    public void onConfigurationChanged(Configuration configuration) {
        super.onConfigurationChanged(configuration);
        this.f819d.mo6740b();
        this.f819d.f2775a.f2779d.mo6789n(configuration);
    }

    public void onCreate(Bundle bundle) {
        C0753p pVar;
        C0693e<?> eVar = this.f819d.f2775a;
        C0695g gVar = eVar.f2779d;
        if (gVar.f2798m == null) {
            gVar.f2798m = eVar;
            gVar.f2799n = eVar;
            C0712k kVar = null;
            gVar.f2800o = null;
            super.onCreate(bundle);
            C0126c cVar = (C0126c) getLastNonConfigurationInstance();
            if (!(cVar == null || (pVar = cVar.f831a) == null || this.f820e != null)) {
                this.f820e = pVar;
            }
            if (bundle != null) {
                Parcelable parcelable = bundle.getParcelable("android:support:fragments");
                C0692d dVar = this.f819d;
                if (cVar != null) {
                    kVar = cVar.f832b;
                }
                dVar.f2775a.f2779d.mo6787l0(parcelable, kVar);
                if (bundle.containsKey("android:support:next_request_index")) {
                    this.f827l = bundle.getInt("android:support:next_request_index");
                    int[] intArray = bundle.getIntArray("android:support:request_indicies");
                    String[] stringArray = bundle.getStringArray("android:support:request_fragment_who");
                    if (intArray == null || stringArray == null || intArray.length != stringArray.length) {
                        Log.w("FragmentActivity", "Invalid requestCode mapping in savedInstanceState.");
                    } else {
                        this.f828m = new C0538i<>(intArray.length);
                        for (int i = 0; i < intArray.length; i++) {
                            this.f828m.mo6407g(intArray[i], stringArray[i]);
                        }
                    }
                }
            }
            if (this.f828m == null) {
                this.f828m = new C0538i<>(10);
                this.f827l = 0;
            }
            this.f819d.f2775a.f2779d.mo6795p();
            return;
        }
        throw new IllegalStateException("Already attached");
    }

    public boolean onCreatePanelMenu(int i, Menu menu) {
        if (i != 0) {
            return super.onCreatePanelMenu(i, menu);
        }
        boolean onCreatePanelMenu = super.onCreatePanelMenu(i, menu);
        C0692d dVar = this.f819d;
        return onCreatePanelMenu | dVar.f2775a.f2779d.mo6797q(menu, getMenuInflater());
    }

    public View onCreateView(View view, String str, Context context, AttributeSet attributeSet) {
        View onCreateView = this.f819d.f2775a.f2779d.onCreateView(view, str, context, attributeSet);
        return onCreateView == null ? super.onCreateView(view, str, context, attributeSet) : onCreateView;
    }

    public void onDestroy() {
        super.onDestroy();
        if (this.f820e != null && !isChangingConfigurations()) {
            this.f820e.mo6886a();
        }
        this.f819d.f2775a.f2779d.mo6799r();
    }

    public void onLowMemory() {
        super.onLowMemory();
        this.f819d.f2775a.f2779d.mo6800s();
    }

    public boolean onMenuItemSelected(int i, MenuItem menuItem) {
        if (super.onMenuItemSelected(i, menuItem)) {
            return true;
        }
        if (i == 0) {
            return this.f819d.f2775a.f2779d.mo6753I(menuItem);
        }
        if (i != 6) {
            return false;
        }
        return this.f819d.f2775a.f2779d.mo6791o(menuItem);
    }

    public void onMultiWindowModeChanged(boolean z) {
        this.f819d.f2775a.f2779d.mo6802t(z);
    }

    public void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        this.f819d.mo6740b();
    }

    public void onPanelClosed(int i, Menu menu) {
        if (i == 0) {
            this.f819d.f2775a.f2779d.mo6754J(menu);
        }
        super.onPanelClosed(i, menu);
    }

    public void onPause() {
        super.onPause();
        this.f822g = false;
        if (this.f818c.hasMessages(2)) {
            this.f818c.removeMessages(2);
            mo1284l();
        }
        this.f819d.f2775a.f2779d.mo6759O(3);
    }

    public void onPictureInPictureModeChanged(boolean z) {
        this.f819d.f2775a.f2779d.mo6755K(z);
    }

    public void onPostResume() {
        super.onPostResume();
        this.f818c.removeMessages(2);
        mo1284l();
        this.f819d.mo6739a();
    }

    public boolean onPreparePanel(int i, View view, Menu menu) {
        if (i != 0 || menu == null) {
            return super.onPreparePanel(i, view, menu);
        }
        return super.onPreparePanel(0, view, menu) | this.f819d.f2775a.f2779d.mo6756L(menu);
    }

    public void onRequestPermissionsResult(int i, String[] strArr, int[] iArr) {
        this.f819d.mo6740b();
        int i2 = (i >> 16) & Parser.CLEAR_TI_MASK;
        if (i2 != 0) {
            int i3 = i2 - 1;
            String d = this.f828m.mo6404d(i3);
            this.f828m.mo6408h(i3);
            if (d == null) {
                Log.w("FragmentActivity", "Activity result delivered for unknown Fragment.");
                return;
            }
            Fragment W = this.f819d.f2775a.f2779d.mo6767W(d);
            if (W == null) {
                Log.w("FragmentActivity", "Activity result no fragment exists for who: " + d);
                return;
            }
            W.onRequestPermissionsResult(i & Parser.CLEAR_TI_MASK, strArr, iArr);
        }
    }

    public void onResume() {
        super.onResume();
        this.f818c.sendEmptyMessage(2);
        this.f822g = true;
        this.f819d.mo6739a();
    }

    public final Object onRetainNonConfigurationInstance() {
        C0695g gVar = this.f819d.f2775a.f2779d;
        C0695g.m1815t0(gVar.f2785B);
        C0712k kVar = gVar.f2785B;
        if (kVar == null && this.f820e == null) {
            return null;
        }
        C0126c cVar = new C0126c();
        cVar.f831a = this.f820e;
        cVar.f832b = kVar;
        return cVar;
    }

    public void onSaveInstanceState(Bundle bundle) {
        super.onSaveInstanceState(bundle);
        do {
        } while (m404j(mo1282i(), C0739f.C0741b.CREATED));
        Parcelable n0 = this.f819d.f2775a.f2779d.mo6790n0();
        if (n0 != null) {
            bundle.putParcelable("android:support:fragments", n0);
        }
        if (this.f828m.mo6409i() > 0) {
            bundle.putInt("android:support:next_request_index", this.f827l);
            int[] iArr = new int[this.f828m.mo6409i()];
            String[] strArr = new String[this.f828m.mo6409i()];
            for (int i = 0; i < this.f828m.mo6409i(); i++) {
                iArr[i] = this.f828m.mo6406f(i);
                strArr[i] = this.f828m.mo6410j(i);
            }
            bundle.putIntArray("android:support:request_indicies", iArr);
            bundle.putStringArray("android:support:request_fragment_who", strArr);
        }
    }

    public void onStart() {
        super.onStart();
        this.f823h = false;
        if (!this.f821f) {
            this.f821f = true;
            this.f819d.f2775a.f2779d.mo6788m();
        }
        this.f819d.mo6740b();
        this.f819d.mo6739a();
        this.f819d.f2775a.f2779d.mo6758N();
    }

    public void onStateNotSaved() {
        this.f819d.mo6740b();
    }

    public void onStop() {
        super.onStop();
        this.f823h = true;
        do {
        } while (m404j(mo1282i(), C0739f.C0741b.CREATED));
        C0695g gVar = this.f819d.f2775a.f2779d;
        gVar.f2804s = true;
        gVar.mo6759O(2);
    }

    public void startActivityForResult(Intent intent, int i) {
        if (!this.f826k && i != -1) {
            m403h(i);
        }
        super.startActivityForResult(intent, i);
    }

    public void startIntentSenderForResult(IntentSender intentSender, int i, Intent intent, int i2, int i3, int i4) {
        if (!this.f825j && i != -1) {
            m403h(i);
        }
        super.startIntentSenderForResult(intentSender, i, intent, i2, i3, i4);
    }

    public View onCreateView(String str, Context context, AttributeSet attributeSet) {
        View onCreateView = this.f819d.f2775a.f2779d.onCreateView((View) null, str, context, attributeSet);
        return onCreateView == null ? super.onCreateView(str, context, attributeSet) : onCreateView;
    }

    public void startActivityForResult(Intent intent, int i, Bundle bundle) {
        if (!this.f826k && i != -1) {
            m403h(i);
        }
        super.startActivityForResult(intent, i, bundle);
    }

    public void startIntentSenderForResult(IntentSender intentSender, int i, Intent intent, int i2, int i3, int i4, Bundle bundle) {
        if (!this.f825j && i != -1) {
            m403h(i);
        }
        super.startIntentSenderForResult(intentSender, i, intent, i2, i3, i4, bundle);
    }
}
