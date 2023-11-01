package androidx.fragment.app;

import android.animation.Animator;
import android.app.Activity;
import android.content.ComponentCallbacks;
import android.content.Context;
import android.content.Intent;
import android.content.IntentSender;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.os.Bundle;
import android.os.Looper;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.util.SparseArray;
import android.view.ContextMenu;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.Animation;
import androidx.fragment.app.FragmentActivity;
import androidx.lifecycle.LiveData;
import java.io.FileDescriptor;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import org.mozilla.javascript.Parser;
import org.mozilla.javascript.Token;
import p008d.p009a.p010k.C0375p;
import p008d.p023c.C0537h;
import p008d.p026e.p027d.C0547a;
import p008d.p026e.p027d.C0565j;
import p008d.p050j.p051a.C0689a0;
import p008d.p050j.p051a.C0691c;
import p008d.p050j.p051a.C0693e;
import p008d.p050j.p051a.C0694f;
import p008d.p050j.p051a.C0695g;
import p008d.p050j.p051a.C0712k;
import p008d.p055l.C0739f;
import p008d.p055l.C0742g;
import p008d.p055l.C0743h;
import p008d.p055l.C0747k;
import p008d.p055l.C0753p;
import p008d.p055l.C0754q;
import p008d.p056m.p057a.C0755a;
import p008d.p056m.p057a.C0756b;
import p067e.p068a.p069a.p070a.C0802a;

public class Fragment implements ComponentCallbacks, View.OnCreateContextMenuListener, C0742g, C0754q {
    public static final int ACTIVITY_CREATED = 2;
    public static final int CREATED = 1;
    public static final int INITIALIZING = 0;
    public static final int RESUMED = 4;
    public static final int STARTED = 3;
    public static final Object USE_DEFAULT_TRANSITION = new Object();
    public static final C0537h<String, Class<?>> sClassMap = new C0537h<>();
    public boolean mAdded;
    public C0121d mAnimationInfo;
    public Bundle mArguments;
    public int mBackStackNesting;
    public boolean mCalled;
    public C0695g mChildFragmentManager;
    public C0712k mChildNonConfig;
    public ViewGroup mContainer;
    public int mContainerId;
    public boolean mDeferStart;
    public boolean mDetached;
    public int mFragmentId;
    public C0695g mFragmentManager;
    public boolean mFromLayout;
    public boolean mHasMenu;
    public boolean mHidden;
    public boolean mHiddenChanged;
    public C0693e mHost;
    public boolean mInLayout;
    public int mIndex = -1;
    public View mInnerView;
    public boolean mIsCreated;
    public boolean mIsNewlyAdded;
    public LayoutInflater mLayoutInflater;
    public C0743h mLifecycleRegistry = new C0743h(this);
    public boolean mMenuVisible = true;
    public Fragment mParentFragment;
    public boolean mPerformedCreateView;
    public float mPostponedAlpha;
    public boolean mRemoving;
    public boolean mRestored;
    public boolean mRetainInstance;
    public boolean mRetaining;
    public Bundle mSavedFragmentState;
    public Boolean mSavedUserVisibleHint;
    public SparseArray<Parcelable> mSavedViewState;
    public int mState = 0;
    public String mTag;
    public Fragment mTarget;
    public int mTargetIndex = -1;
    public int mTargetRequestCode;
    public boolean mUserVisibleHint = true;
    public View mView;
    public C0742g mViewLifecycleOwner;
    public C0747k<C0742g> mViewLifecycleOwnerLiveData = new C0747k<>();
    public C0743h mViewLifecycleRegistry;
    public C0753p mViewModelStore;
    public String mWho;

    public static class SavedState implements Parcelable {
        public static final Parcelable.Creator<SavedState> CREATOR = new C0117a();

       
        public final Bundle f795b;

       
        public static class C0117a implements Parcelable.ClassLoaderCreator<SavedState> {
            public Object createFromParcel(Parcel parcel) {
                return new SavedState(parcel, (ClassLoader) null);
            }

            public Object[] newArray(int i) {
                return new SavedState[i];
            }

            public Object createFromParcel(Parcel parcel, ClassLoader classLoader) {
                return new SavedState(parcel, classLoader);
            }
        }

        public SavedState(Parcel parcel, ClassLoader classLoader) {
            Bundle readBundle = parcel.readBundle();
            this.f795b = readBundle;
            if (classLoader != null && readBundle != null) {
                readBundle.setClassLoader(classLoader);
            }
        }

        public int describeContents() {
            return 0;
        }

        public void writeToParcel(Parcel parcel, int i) {
            parcel.writeBundle(this.f795b);
        }
    }

   
    public class C0118a implements Runnable {
        public C0118a() {
        }

        public void run() {
            Fragment.this.callStartTransitionListener();
        }
    }

   
    public class C0119b extends C0691c {
        public C0119b() {
        }

       
        public Fragment mo1276a(Context context, String str, Bundle bundle) {
            if (Fragment.this.mHost != null) {
                return Fragment.instantiate(context, str, bundle);
            }
            throw null;
        }

       
        public View mo1277b(int i) {
            View view = Fragment.this.mView;
            if (view != null) {
                return view.findViewById(i);
            }
            throw new IllegalStateException("Fragment does not have a view");
        }

       
        public boolean mo1278c() {
            return Fragment.this.mView != null;
        }
    }

   
    public class C0120c implements C0742g {
        public C0120c() {
        }

        public C0739f getLifecycle() {
            Fragment fragment = Fragment.this;
            if (fragment.mViewLifecycleRegistry == null) {
                fragment.mViewLifecycleRegistry = new C0743h(fragment.mViewLifecycleOwner);
            }
            return Fragment.this.mViewLifecycleRegistry;
        }
    }

   
    public static class C0121d {

       
        public View f799a;

       
        public Animator f800b;

       
        public int f801c;

       
        public int f802d;

       
        public int f803e;

       
        public int f804f;

       
        public Object f805g = null;

       
        public Object f806h;

       
        public Object f807i;

       
        public Object f808j;

       
        public Object f809k;

       
        public Object f810l;

       
        public Boolean f811m;

       
        public Boolean f812n;

       
        public C0565j f813o;

       
        public C0565j f814p;

       
        public boolean f815q;

       
        public C0123f f816r;

       
        public boolean f817s;

        public C0121d() {
            Object obj = Fragment.USE_DEFAULT_TRANSITION;
            this.f806h = obj;
            this.f807i = null;
            this.f808j = obj;
            this.f809k = null;
            this.f810l = obj;
            this.f813o = null;
            this.f814p = null;
        }
    }

   
    public static class C0122e extends RuntimeException {
        public C0122e(String str, Exception exc) {
            super(str, exc);
        }
    }

   
    public interface C0123f {
    }

    private C0121d ensureAnimationInfo() {
        if (this.mAnimationInfo == null) {
            this.mAnimationInfo = new C0121d();
        }
        return this.mAnimationInfo;
    }

    public static Fragment instantiate(Context context, String str) {
        return instantiate(context, str, (Bundle) null);
    }

    public static boolean isSupportFragmentClass(Context context, String str) {
        try {
            Class<?> cls = sClassMap.get(str);
            if (cls == null) {
                cls = context.getClassLoader().loadClass(str);
                sClassMap.put(str, cls);
            }
            return Fragment.class.isAssignableFrom(cls);
        } catch (ClassNotFoundException unused) {
            return false;
        }
    }

    public void callStartTransitionListener() {
        C0121d dVar = this.mAnimationInfo;
        C0123f fVar = null;
        if (dVar != null) {
            dVar.f815q = false;
            C0123f fVar2 = dVar.f816r;
            dVar.f816r = null;
            fVar = fVar2;
        }
        if (fVar != null) {
            C0695g.C0707k kVar = (C0695g.C0707k) fVar;
            int i = kVar.f2832c - 1;
            kVar.f2832c = i;
            if (i == 0) {
                kVar.f2831b.f2749a.mo6798q0();
            }
        }
    }

    public void dump(String str, FileDescriptor fileDescriptor, PrintWriter printWriter, String[] strArr) {
        printWriter.print(str);
        printWriter.print("mFragmentId=#");
        printWriter.print(Integer.toHexString(this.mFragmentId));
        printWriter.print(" mContainerId=#");
        printWriter.print(Integer.toHexString(this.mContainerId));
        printWriter.print(" mTag=");
        printWriter.println(this.mTag);
        printWriter.print(str);
        printWriter.print("mState=");
        printWriter.print(this.mState);
        printWriter.print(" mIndex=");
        printWriter.print(this.mIndex);
        printWriter.print(" mWho=");
        printWriter.print(this.mWho);
        printWriter.print(" mBackStackNesting=");
        printWriter.println(this.mBackStackNesting);
        printWriter.print(str);
        printWriter.print("mAdded=");
        printWriter.print(this.mAdded);
        printWriter.print(" mRemoving=");
        printWriter.print(this.mRemoving);
        printWriter.print(" mFromLayout=");
        printWriter.print(this.mFromLayout);
        printWriter.print(" mInLayout=");
        printWriter.println(this.mInLayout);
        printWriter.print(str);
        printWriter.print("mHidden=");
        printWriter.print(this.mHidden);
        printWriter.print(" mDetached=");
        printWriter.print(this.mDetached);
        printWriter.print(" mMenuVisible=");
        printWriter.print(this.mMenuVisible);
        printWriter.print(" mHasMenu=");
        printWriter.println(this.mHasMenu);
        printWriter.print(str);
        printWriter.print("mRetainInstance=");
        printWriter.print(this.mRetainInstance);
        printWriter.print(" mRetaining=");
        printWriter.print(this.mRetaining);
        printWriter.print(" mUserVisibleHint=");
        printWriter.println(this.mUserVisibleHint);
        if (this.mFragmentManager != null) {
            printWriter.print(str);
            printWriter.print("mFragmentManager=");
            printWriter.println(this.mFragmentManager);
        }
        if (this.mHost != null) {
            printWriter.print(str);
            printWriter.print("mHost=");
            printWriter.println(this.mHost);
        }
        if (this.mParentFragment != null) {
            printWriter.print(str);
            printWriter.print("mParentFragment=");
            printWriter.println(this.mParentFragment);
        }
        if (this.mArguments != null) {
            printWriter.print(str);
            printWriter.print("mArguments=");
            printWriter.println(this.mArguments);
        }
        if (this.mSavedFragmentState != null) {
            printWriter.print(str);
            printWriter.print("mSavedFragmentState=");
            printWriter.println(this.mSavedFragmentState);
        }
        if (this.mSavedViewState != null) {
            printWriter.print(str);
            printWriter.print("mSavedViewState=");
            printWriter.println(this.mSavedViewState);
        }
        if (this.mTarget != null) {
            printWriter.print(str);
            printWriter.print("mTarget=");
            printWriter.print(this.mTarget);
            printWriter.print(" mTargetRequestCode=");
            printWriter.println(this.mTargetRequestCode);
        }
        if (getNextAnim() != 0) {
            printWriter.print(str);
            printWriter.print("mNextAnim=");
            printWriter.println(getNextAnim());
        }
        if (this.mContainer != null) {
            printWriter.print(str);
            printWriter.print("mContainer=");
            printWriter.println(this.mContainer);
        }
        if (this.mView != null) {
            printWriter.print(str);
            printWriter.print("mView=");
            printWriter.println(this.mView);
        }
        if (this.mInnerView != null) {
            printWriter.print(str);
            printWriter.print("mInnerView=");
            printWriter.println(this.mView);
        }
        if (getAnimatingAway() != null) {
            printWriter.print(str);
            printWriter.print("mAnimatingAway=");
            printWriter.println(getAnimatingAway());
            printWriter.print(str);
            printWriter.print("mStateAfterAnimating=");
            printWriter.println(getStateAfterAnimating());
        }
        if (getContext() != null) {
            C0755a.m1947b(this).mo6887a(str, fileDescriptor, printWriter, strArr);
        }
        if (this.mChildFragmentManager != null) {
            printWriter.print(str);
            printWriter.println("Child " + this.mChildFragmentManager + ":");
            this.mChildFragmentManager.mo6741a(C0802a.m2038o(str, "  "), fileDescriptor, printWriter, strArr);
        }
    }

    public final boolean equals(Object obj) {
        return super.equals(obj);
    }

    public Fragment findFragmentByWho(String str) {
        if (str.equals(this.mWho)) {
            return this;
        }
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            return gVar.mo6767W(str);
        }
        return null;
    }

    public final FragmentActivity getActivity() {
        C0693e eVar = this.mHost;
        if (eVar == null) {
            return null;
        }
        return (FragmentActivity) eVar.f2776a;
    }

    public boolean getAllowEnterTransitionOverlap() {
        Boolean bool;
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null || (bool = dVar.f812n) == null) {
            return true;
        }
        return bool.booleanValue();
    }

    public boolean getAllowReturnTransitionOverlap() {
        Boolean bool;
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null || (bool = dVar.f811m) == null) {
            return true;
        }
        return bool.booleanValue();
    }

    public View getAnimatingAway() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return null;
        }
        return dVar.f799a;
    }

    public Animator getAnimator() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return null;
        }
        return dVar.f800b;
    }

    public final Bundle getArguments() {
        return this.mArguments;
    }

    public final C0694f getChildFragmentManager() {
        if (this.mChildFragmentManager == null) {
            instantiateChildFragmentManager();
            int i = this.mState;
            if (i >= 4) {
                this.mChildFragmentManager.mo6757M();
            } else if (i >= 3) {
                this.mChildFragmentManager.mo6758N();
            } else if (i >= 2) {
                this.mChildFragmentManager.mo6788m();
            } else if (i >= 1) {
                this.mChildFragmentManager.mo6795p();
            }
        }
        return this.mChildFragmentManager;
    }

    public Context getContext() {
        C0693e eVar = this.mHost;
        if (eVar == null) {
            return null;
        }
        return eVar.f2777b;
    }

    public Object getEnterTransition() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return null;
        }
        return dVar.f805g;
    }

    public C0565j getEnterTransitionCallback() {
        if (this.mAnimationInfo == null) {
        }
        return null;
    }

    public Object getExitTransition() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return null;
        }
        return dVar.f807i;
    }

    public C0565j getExitTransitionCallback() {
        if (this.mAnimationInfo == null) {
        }
        return null;
    }

    public final C0694f getFragmentManager() {
        return this.mFragmentManager;
    }

    public final Object getHost() {
        C0693e eVar = this.mHost;
        if (eVar == null) {
            return null;
        }
        return FragmentActivity.this;
    }

    public final int getId() {
        return this.mFragmentId;
    }

    public final LayoutInflater getLayoutInflater() {
        LayoutInflater layoutInflater = this.mLayoutInflater;
        return layoutInflater == null ? performGetLayoutInflater((Bundle) null) : layoutInflater;
    }

    public C0739f getLifecycle() {
        return this.mLifecycleRegistry;
    }

    @Deprecated
    public C0755a getLoaderManager() {
        return C0755a.m1947b(this);
    }

    public int getNextAnim() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return 0;
        }
        return dVar.f802d;
    }

    public int getNextTransition() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return 0;
        }
        return dVar.f803e;
    }

    public int getNextTransitionStyle() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return 0;
        }
        return dVar.f804f;
    }

    public final Fragment getParentFragment() {
        return this.mParentFragment;
    }

    public Object getReenterTransition() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return null;
        }
        Object obj = dVar.f808j;
        return obj == USE_DEFAULT_TRANSITION ? getExitTransition() : obj;
    }

    public final Resources getResources() {
        return requireContext().getResources();
    }

    public final boolean getRetainInstance() {
        return this.mRetainInstance;
    }

    public Object getReturnTransition() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return null;
        }
        Object obj = dVar.f806h;
        return obj == USE_DEFAULT_TRANSITION ? getEnterTransition() : obj;
    }

    public Object getSharedElementEnterTransition() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return null;
        }
        return dVar.f809k;
    }

    public Object getSharedElementReturnTransition() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return null;
        }
        Object obj = dVar.f810l;
        return obj == USE_DEFAULT_TRANSITION ? getSharedElementEnterTransition() : obj;
    }

    public int getStateAfterAnimating() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return 0;
        }
        return dVar.f801c;
    }

    public final String getString(int i) {
        return getResources().getString(i);
    }

    public final String getTag() {
        return this.mTag;
    }

    public final Fragment getTargetFragment() {
        return this.mTarget;
    }

    public final int getTargetRequestCode() {
        return this.mTargetRequestCode;
    }

    public final CharSequence getText(int i) {
        return getResources().getText(i);
    }

    public boolean getUserVisibleHint() {
        return this.mUserVisibleHint;
    }

    public View getView() {
        return this.mView;
    }

    public C0742g getViewLifecycleOwner() {
        C0742g gVar = this.mViewLifecycleOwner;
        if (gVar != null) {
            return gVar;
        }
        throw new IllegalStateException("Can't access the Fragment View's LifecycleOwner when getView() is null i.e., before onCreateView() or after onDestroyView()");
    }

    public LiveData<C0742g> getViewLifecycleOwnerLiveData() {
        return this.mViewLifecycleOwnerLiveData;
    }

    public C0753p getViewModelStore() {
        if (getContext() != null) {
            if (this.mViewModelStore == null) {
                this.mViewModelStore = new C0753p();
            }
            return this.mViewModelStore;
        }
        throw new IllegalStateException("Can't access ViewModels from detached fragment");
    }

    public final boolean hasOptionsMenu() {
        return this.mHasMenu;
    }

    public final int hashCode() {
        return super.hashCode();
    }

    public void initState() {
        this.mIndex = -1;
        this.mWho = null;
        this.mAdded = false;
        this.mRemoving = false;
        this.mFromLayout = false;
        this.mInLayout = false;
        this.mRestored = false;
        this.mBackStackNesting = 0;
        this.mFragmentManager = null;
        this.mChildFragmentManager = null;
        this.mHost = null;
        this.mFragmentId = 0;
        this.mContainerId = 0;
        this.mTag = null;
        this.mHidden = false;
        this.mDetached = false;
        this.mRetaining = false;
    }

    public void instantiateChildFragmentManager() {
        if (this.mHost != null) {
            C0695g gVar = new C0695g();
            this.mChildFragmentManager = gVar;
            C0693e eVar = this.mHost;
            C0119b bVar = new C0119b();
            if (gVar.f2798m == null) {
                gVar.f2798m = eVar;
                gVar.f2799n = bVar;
                gVar.f2800o = this;
                return;
            }
            throw new IllegalStateException("Already attached");
        }
        throw new IllegalStateException("Fragment has not been attached yet.");
    }

    public final boolean isAdded() {
        return this.mHost != null && this.mAdded;
    }

    public final boolean isDetached() {
        return this.mDetached;
    }

    public final boolean isHidden() {
        return this.mHidden;
    }

    public boolean isHideReplaced() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return false;
        }
        return dVar.f817s;
    }

    public final boolean isInBackStack() {
        return this.mBackStackNesting > 0;
    }

    public final boolean isInLayout() {
        return this.mInLayout;
    }

    public final boolean isMenuVisible() {
        return this.mMenuVisible;
    }

    public boolean isPostponed() {
        C0121d dVar = this.mAnimationInfo;
        if (dVar == null) {
            return false;
        }
        return dVar.f815q;
    }

    public final boolean isRemoving() {
        return this.mRemoving;
    }

    public final boolean isResumed() {
        return this.mState >= 4;
    }

    public final boolean isStateSaved() {
        C0695g gVar = this.mFragmentManager;
        if (gVar == null) {
            return false;
        }
        return gVar.mo6743c();
    }

   
   
    public final boolean isVisible() {
       
        throw new UnsupportedOperationException("Method not decompiled: androidx.fragment.app.Fragment.isVisible():boolean");
    }

    public void noteStateNotSaved() {
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6777g0();
        }
    }

    public void onActivityCreated(Bundle bundle) {
        this.mCalled = true;
    }

    public void onActivityResult(int i, int i2, Intent intent) {
    }

    public void onAttach(Context context) {
        Activity activity;
        this.mCalled = true;
        C0693e eVar = this.mHost;
        if (eVar == null) {
            activity = null;
        } else {
            activity = eVar.f2776a;
        }
        if (activity != null) {
            this.mCalled = false;
            onAttach(activity);
        }
    }

    public void onAttachFragment(Fragment fragment) {
    }

    public void onConfigurationChanged(Configuration configuration) {
        this.mCalled = true;
    }

    public boolean onContextItemSelected(MenuItem menuItem) {
        return false;
    }

    public void onCreate(Bundle bundle) {
        boolean z = true;
        this.mCalled = true;
        restoreChildFragmentState(bundle);
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            if (gVar.f2797l < 1) {
                z = false;
            }
            if (!z) {
                this.mChildFragmentManager.mo6795p();
            }
        }
    }

    public Animation onCreateAnimation(int i, boolean z, int i2) {
        return null;
    }

    public Animator onCreateAnimator(int i, boolean z, int i2) {
        return null;
    }

    public void onCreateContextMenu(ContextMenu contextMenu, View view, ContextMenu.ContextMenuInfo contextMenuInfo) {
        getActivity().onCreateContextMenu(contextMenu, view, contextMenuInfo);
    }

    public void onCreateOptionsMenu(Menu menu, MenuInflater menuInflater) {
    }

    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        return null;
    }

    public void onDestroy() {
        boolean z = true;
        this.mCalled = true;
        FragmentActivity activity = getActivity();
        if (activity == null || !activity.isChangingConfigurations()) {
            z = false;
        }
        C0753p pVar = this.mViewModelStore;
        if (pVar != null && !z) {
            pVar.mo6886a();
        }
    }

    public void onDestroyOptionsMenu() {
    }

    public void onDestroyView() {
        this.mCalled = true;
    }

    public void onDetach() {
        this.mCalled = true;
    }

    public LayoutInflater onGetLayoutInflater(Bundle bundle) {
        return getLayoutInflater(bundle);
    }

    public void onHiddenChanged(boolean z) {
    }

    public void onInflate(Context context, AttributeSet attributeSet, Bundle bundle) {
        Activity activity;
        this.mCalled = true;
        C0693e eVar = this.mHost;
        if (eVar == null) {
            activity = null;
        } else {
            activity = eVar.f2776a;
        }
        if (activity != null) {
            this.mCalled = false;
            onInflate(activity, attributeSet, bundle);
        }
    }

    public void onLowMemory() {
        this.mCalled = true;
    }

    public void onMultiWindowModeChanged(boolean z) {
    }

    public boolean onOptionsItemSelected(MenuItem menuItem) {
        return false;
    }

    public void onOptionsMenuClosed(Menu menu) {
    }

    public void onPause() {
        this.mCalled = true;
    }

    public void onPictureInPictureModeChanged(boolean z) {
    }

    public void onPrepareOptionsMenu(Menu menu) {
    }

    public void onRequestPermissionsResult(int i, String[] strArr, int[] iArr) {
    }

    public void onResume() {
        this.mCalled = true;
    }

    public void onSaveInstanceState(Bundle bundle) {
    }

    public void onStart() {
        this.mCalled = true;
    }

    public void onStop() {
        this.mCalled = true;
    }

    public void onViewCreated(View view, Bundle bundle) {
    }

    public void onViewStateRestored(Bundle bundle) {
        this.mCalled = true;
    }

    public C0694f peekChildFragmentManager() {
        return this.mChildFragmentManager;
    }

    public void performActivityCreated(Bundle bundle) {
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6777g0();
        }
        this.mState = 2;
        this.mCalled = false;
        onActivityCreated(bundle);
        if (this.mCalled) {
            C0695g gVar2 = this.mChildFragmentManager;
            if (gVar2 != null) {
                gVar2.mo6788m();
                return;
            }
            return;
        }
        throw new C0689a0(C0802a.m2036m("Fragment ", this, " did not call through to super.onActivityCreated()"));
    }

    public void performConfigurationChanged(Configuration configuration) {
        onConfigurationChanged(configuration);
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6789n(configuration);
        }
    }

    public boolean performContextItemSelected(MenuItem menuItem) {
        if (this.mHidden) {
            return false;
        }
        if (onContextItemSelected(menuItem)) {
            return true;
        }
        C0695g gVar = this.mChildFragmentManager;
        if (gVar == null || !gVar.mo6791o(menuItem)) {
            return false;
        }
        return true;
    }

    public void performCreate(Bundle bundle) {
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6777g0();
        }
        this.mState = 1;
        this.mCalled = false;
        onCreate(bundle);
        this.mIsCreated = true;
        if (this.mCalled) {
            this.mLifecycleRegistry.mo6868b(C0739f.C0740a.ON_CREATE);
            return;
        }
        throw new C0689a0(C0802a.m2036m("Fragment ", this, " did not call through to super.onCreate()"));
    }

    public boolean performCreateOptionsMenu(Menu menu, MenuInflater menuInflater) {
        boolean z = false;
        if (this.mHidden) {
            return false;
        }
        if (this.mHasMenu && this.mMenuVisible) {
            onCreateOptionsMenu(menu, menuInflater);
            z = true;
        }
        C0695g gVar = this.mChildFragmentManager;
        return gVar != null ? z | gVar.mo6797q(menu, menuInflater) : z;
    }

    public void performCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6777g0();
        }
        this.mPerformedCreateView = true;
        this.mViewLifecycleOwner = new C0120c();
        this.mViewLifecycleRegistry = null;
        View onCreateView = onCreateView(layoutInflater, viewGroup, bundle);
        this.mView = onCreateView;
        if (onCreateView != null) {
            this.mViewLifecycleOwner.getLifecycle();
            this.mViewLifecycleOwnerLiveData.mo6873g(this.mViewLifecycleOwner);
        } else if (this.mViewLifecycleRegistry == null) {
            this.mViewLifecycleOwner = null;
        } else {
            throw new IllegalStateException("Called getViewLifecycleOwner() but onCreateView() returned null");
        }
    }

    public void performDestroy() {
        this.mLifecycleRegistry.mo6868b(C0739f.C0740a.ON_DESTROY);
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6799r();
        }
        this.mState = 0;
        this.mCalled = false;
        this.mIsCreated = false;
        onDestroy();
        if (this.mCalled) {
            this.mChildFragmentManager = null;
            return;
        }
        throw new C0689a0(C0802a.m2036m("Fragment ", this, " did not call through to super.onDestroy()"));
    }

    public void performDestroyView() {
        if (this.mView != null) {
            this.mViewLifecycleRegistry.mo6868b(C0739f.C0740a.ON_DESTROY);
        }
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6759O(1);
        }
        this.mState = 1;
        this.mCalled = false;
        onDestroyView();
        if (this.mCalled) {
            C0756b.C0759c cVar = ((C0756b) C0755a.m1947b(this)).f2933b;
            int i = cVar.f2937a.mo6409i();
            for (int i2 = 0; i2 < i; i2++) {
                C0742g gVar2 = cVar.f2937a.mo6410j(i2).f2934j;
            }
            this.mPerformedCreateView = false;
            return;
        }
        throw new C0689a0(C0802a.m2036m("Fragment ", this, " did not call through to super.onDestroyView()"));
    }

    public void performDetach() {
        this.mCalled = false;
        onDetach();
        this.mLayoutInflater = null;
        if (this.mCalled) {
            C0695g gVar = this.mChildFragmentManager;
            if (gVar == null) {
                return;
            }
            if (this.mRetaining) {
                gVar.mo6799r();
                this.mChildFragmentManager = null;
                return;
            }
            throw new IllegalStateException("Child FragmentManager of " + this + " was not " + " destroyed and this fragment is not retaining instance");
        }
        throw new C0689a0(C0802a.m2036m("Fragment ", this, " did not call through to super.onDetach()"));
    }

    public LayoutInflater performGetLayoutInflater(Bundle bundle) {
        LayoutInflater onGetLayoutInflater = onGetLayoutInflater(bundle);
        this.mLayoutInflater = onGetLayoutInflater;
        return onGetLayoutInflater;
    }

    public void performLowMemory() {
        onLowMemory();
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6800s();
        }
    }

    public void performMultiWindowModeChanged(boolean z) {
        onMultiWindowModeChanged(z);
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6802t(z);
        }
    }

    public boolean performOptionsItemSelected(MenuItem menuItem) {
        if (this.mHidden) {
            return false;
        }
        if (this.mHasMenu && this.mMenuVisible && onOptionsItemSelected(menuItem)) {
            return true;
        }
        C0695g gVar = this.mChildFragmentManager;
        if (gVar == null || !gVar.mo6753I(menuItem)) {
            return false;
        }
        return true;
    }

    public void performOptionsMenuClosed(Menu menu) {
        if (!this.mHidden) {
            if (this.mHasMenu && this.mMenuVisible) {
                onOptionsMenuClosed(menu);
            }
            C0695g gVar = this.mChildFragmentManager;
            if (gVar != null) {
                gVar.mo6754J(menu);
            }
        }
    }

    public void performPause() {
        if (this.mView != null) {
            this.mViewLifecycleRegistry.mo6868b(C0739f.C0740a.ON_PAUSE);
        }
        this.mLifecycleRegistry.mo6868b(C0739f.C0740a.ON_PAUSE);
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6759O(3);
        }
        this.mState = 3;
        this.mCalled = false;
        onPause();
        if (!this.mCalled) {
            throw new C0689a0(C0802a.m2036m("Fragment ", this, " did not call through to super.onPause()"));
        }
    }

    public void performPictureInPictureModeChanged(boolean z) {
        onPictureInPictureModeChanged(z);
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6755K(z);
        }
    }

    public boolean performPrepareOptionsMenu(Menu menu) {
        boolean z = false;
        if (this.mHidden) {
            return false;
        }
        if (this.mHasMenu && this.mMenuVisible) {
            onPrepareOptionsMenu(menu);
            z = true;
        }
        C0695g gVar = this.mChildFragmentManager;
        return gVar != null ? z | gVar.mo6756L(menu) : z;
    }

    public void performResume() {
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6777g0();
            this.mChildFragmentManager.mo6763S();
        }
        this.mState = 4;
        this.mCalled = false;
        onResume();
        if (this.mCalled) {
            C0695g gVar2 = this.mChildFragmentManager;
            if (gVar2 != null) {
                gVar2.mo6757M();
                this.mChildFragmentManager.mo6763S();
            }
            this.mLifecycleRegistry.mo6868b(C0739f.C0740a.ON_RESUME);
            if (this.mView != null) {
                this.mViewLifecycleRegistry.mo6868b(C0739f.C0740a.ON_RESUME);
                return;
            }
            return;
        }
        throw new C0689a0(C0802a.m2036m("Fragment ", this, " did not call through to super.onResume()"));
    }

    public void performSaveInstanceState(Bundle bundle) {
        Parcelable n0;
        onSaveInstanceState(bundle);
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null && (n0 = gVar.mo6790n0()) != null) {
            bundle.putParcelable("android:support:fragments", n0);
        }
    }

    public void performStart() {
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.mo6777g0();
            this.mChildFragmentManager.mo6763S();
        }
        this.mState = 3;
        this.mCalled = false;
        onStart();
        if (this.mCalled) {
            C0695g gVar2 = this.mChildFragmentManager;
            if (gVar2 != null) {
                gVar2.mo6758N();
            }
            this.mLifecycleRegistry.mo6868b(C0739f.C0740a.ON_START);
            if (this.mView != null) {
                this.mViewLifecycleRegistry.mo6868b(C0739f.C0740a.ON_START);
                return;
            }
            return;
        }
        throw new C0689a0(C0802a.m2036m("Fragment ", this, " did not call through to super.onStart()"));
    }

    public void performStop() {
        if (this.mView != null) {
            this.mViewLifecycleRegistry.mo6868b(C0739f.C0740a.ON_STOP);
        }
        this.mLifecycleRegistry.mo6868b(C0739f.C0740a.ON_STOP);
        C0695g gVar = this.mChildFragmentManager;
        if (gVar != null) {
            gVar.f2804s = true;
            gVar.mo6759O(2);
        }
        this.mState = 2;
        this.mCalled = false;
        onStop();
        if (!this.mCalled) {
            throw new C0689a0(C0802a.m2036m("Fragment ", this, " did not call through to super.onStop()"));
        }
    }

    public void postponeEnterTransition() {
        ensureAnimationInfo().f815q = true;
    }

    public void registerForContextMenu(View view) {
        view.setOnCreateContextMenuListener(this);
    }

   
    public final void requestPermissions(String[] strArr, int i) {
        C0693e eVar = this.mHost;
        if (eVar != null) {
            FragmentActivity fragmentActivity = FragmentActivity.this;
            if (fragmentActivity == null) {
                throw null;
            } else if (i == -1) {
                C0547a.m1438k(fragmentActivity, strArr, i);
            } else {
                FragmentActivity.m403h(i);
                try {
                    fragmentActivity.f824i = true;
                    C0547a.m1438k(fragmentActivity, strArr, ((fragmentActivity.mo1281g(this) + 1) << 16) + (i & Parser.CLEAR_TI_MASK));
                    fragmentActivity.f824i = false;
                } catch (Throwable th) {
                    fragmentActivity.f824i = false;
                    throw th;
                }
            }
        } else {
            throw new IllegalStateException(C0802a.m2036m("Fragment ", this, " not attached to Activity"));
        }
    }

    public final FragmentActivity requireActivity() {
        FragmentActivity activity = getActivity();
        if (activity != null) {
            return activity;
        }
        throw new IllegalStateException(C0802a.m2036m("Fragment ", this, " not attached to an activity."));
    }

    public final Context requireContext() {
        Context context = getContext();
        if (context != null) {
            return context;
        }
        throw new IllegalStateException(C0802a.m2036m("Fragment ", this, " not attached to a context."));
    }

    public final C0694f requireFragmentManager() {
        C0694f fragmentManager = getFragmentManager();
        if (fragmentManager != null) {
            return fragmentManager;
        }
        throw new IllegalStateException(C0802a.m2036m("Fragment ", this, " not associated with a fragment manager."));
    }

    public final Object requireHost() {
        Object host = getHost();
        if (host != null) {
            return host;
        }
        throw new IllegalStateException(C0802a.m2036m("Fragment ", this, " not attached to a host."));
    }

    public void restoreChildFragmentState(Bundle bundle) {
        Parcelable parcelable;
        if (bundle != null && (parcelable = bundle.getParcelable("android:support:fragments")) != null) {
            if (this.mChildFragmentManager == null) {
                instantiateChildFragmentManager();
            }
            this.mChildFragmentManager.mo6787l0(parcelable, this.mChildNonConfig);
            this.mChildNonConfig = null;
            this.mChildFragmentManager.mo6795p();
        }
    }

    public final void restoreViewState(Bundle bundle) {
        SparseArray<Parcelable> sparseArray = this.mSavedViewState;
        if (sparseArray != null) {
            this.mInnerView.restoreHierarchyState(sparseArray);
            this.mSavedViewState = null;
        }
        this.mCalled = false;
        onViewStateRestored(bundle);
        if (!this.mCalled) {
            throw new C0689a0(C0802a.m2036m("Fragment ", this, " did not call through to super.onViewStateRestored()"));
        } else if (this.mView != null) {
            this.mViewLifecycleRegistry.mo6868b(C0739f.C0740a.ON_CREATE);
        }
    }

    public void setAllowEnterTransitionOverlap(boolean z) {
        ensureAnimationInfo().f812n = Boolean.valueOf(z);
    }

    public void setAllowReturnTransitionOverlap(boolean z) {
        ensureAnimationInfo().f811m = Boolean.valueOf(z);
    }

    public void setAnimatingAway(View view) {
        ensureAnimationInfo().f799a = view;
    }

    public void setAnimator(Animator animator) {
        ensureAnimationInfo().f800b = animator;
    }

    public void setArguments(Bundle bundle) {
        if (this.mIndex < 0 || !isStateSaved()) {
            this.mArguments = bundle;
            return;
        }
        throw new IllegalStateException("Fragment already active and state has been saved");
    }

    public void setEnterSharedElementCallback(C0565j jVar) {
        ensureAnimationInfo();
    }

    public void setEnterTransition(Object obj) {
        ensureAnimationInfo().f805g = obj;
    }

    public void setExitSharedElementCallback(C0565j jVar) {
        ensureAnimationInfo();
    }

    public void setExitTransition(Object obj) {
        ensureAnimationInfo().f807i = obj;
    }

    public void setHasOptionsMenu(boolean z) {
        if (this.mHasMenu != z) {
            this.mHasMenu = z;
            if (isAdded() && !isHidden()) {
                FragmentActivity.this.mo38m();
            }
        }
    }

    public void setHideReplaced(boolean z) {
        ensureAnimationInfo().f817s = z;
    }

    public final void setIndex(int i, Fragment fragment) {
        this.mIndex = i;
        if (fragment != null) {
            this.mWho = fragment.mWho + ":" + this.mIndex;
            return;
        }
        StringBuilder d = C0802a.m2027d("android:fragment:");
        d.append(this.mIndex);
        this.mWho = d.toString();
    }

    public void setInitialSavedState(SavedState savedState) {
        Bundle bundle;
        if (this.mIndex < 0) {
            if (savedState == null || (bundle = savedState.f795b) == null) {
                bundle = null;
            }
            this.mSavedFragmentState = bundle;
            return;
        }
        throw new IllegalStateException("Fragment already active");
    }

    public void setMenuVisibility(boolean z) {
        if (this.mMenuVisible != z) {
            this.mMenuVisible = z;
            if (this.mHasMenu && isAdded() && !isHidden()) {
                FragmentActivity.this.mo38m();
            }
        }
    }

    public void setNextAnim(int i) {
        if (this.mAnimationInfo != null || i != 0) {
            ensureAnimationInfo().f802d = i;
        }
    }

    public void setNextTransition(int i, int i2) {
        if (this.mAnimationInfo != null || i != 0 || i2 != 0) {
            ensureAnimationInfo();
            C0121d dVar = this.mAnimationInfo;
            dVar.f803e = i;
            dVar.f804f = i2;
        }
    }

    public void setOnStartEnterTransitionListener(C0123f fVar) {
        ensureAnimationInfo();
        C0123f fVar2 = this.mAnimationInfo.f816r;
        if (fVar != fVar2) {
            if (fVar == null || fVar2 == null) {
                C0121d dVar = this.mAnimationInfo;
                if (dVar.f815q) {
                    dVar.f816r = fVar;
                }
                if (fVar != null) {
                    ((C0695g.C0707k) fVar).f2832c++;
                    return;
                }
                return;
            }
            throw new IllegalStateException("Trying to set a replacement startPostponedEnterTransition on " + this);
        }
    }

    public void setReenterTransition(Object obj) {
        ensureAnimationInfo().f808j = obj;
    }

    public void setRetainInstance(boolean z) {
        this.mRetainInstance = z;
    }

    public void setReturnTransition(Object obj) {
        ensureAnimationInfo().f806h = obj;
    }

    public void setSharedElementEnterTransition(Object obj) {
        ensureAnimationInfo().f809k = obj;
    }

    public void setSharedElementReturnTransition(Object obj) {
        ensureAnimationInfo().f810l = obj;
    }

    public void setStateAfterAnimating(int i) {
        ensureAnimationInfo().f801c = i;
    }

    public void setTargetFragment(Fragment fragment, int i) {
        C0694f fragmentManager = getFragmentManager();
        C0694f fragmentManager2 = fragment != null ? fragment.getFragmentManager() : null;
        if (fragmentManager == null || fragmentManager2 == null || fragmentManager == fragmentManager2) {
            Fragment fragment2 = fragment;
            while (fragment2 != null) {
                if (fragment2 != this) {
                    fragment2 = fragment2.getTargetFragment();
                } else {
                    throw new IllegalArgumentException("Setting " + fragment + " as the target of " + this + " would create a target cycle");
                }
            }
            this.mTarget = fragment;
            this.mTargetRequestCode = i;
            return;
        }
        throw new IllegalArgumentException(C0802a.m2036m("Fragment ", fragment, " must share the same FragmentManager to be set as a target fragment"));
    }

    public void setUserVisibleHint(boolean z) {
        if (!this.mUserVisibleHint && z && this.mState < 3 && this.mFragmentManager != null && isAdded() && this.mIsCreated) {
            this.mFragmentManager.mo6779h0(this);
        }
        this.mUserVisibleHint = z;
        this.mDeferStart = this.mState < 3 && !z;
        if (this.mSavedFragmentState != null) {
            this.mSavedUserVisibleHint = Boolean.valueOf(z);
        }
    }

    public boolean shouldShowRequestPermissionRationale(String str) {
        C0693e eVar = this.mHost;
        if (eVar != null) {
            return C0547a.m1439l(FragmentActivity.this, str);
        }
        return false;
    }

    public void startActivity(Intent intent) {
        startActivity(intent, (Bundle) null);
    }

    public void startActivityForResult(Intent intent, int i) {
        startActivityForResult(intent, i, (Bundle) null);
    }

    public void startIntentSenderForResult(IntentSender intentSender, int i, Intent intent, int i2, int i3, int i4, Bundle bundle) {
        int i5 = i;
        C0693e eVar = this.mHost;
        if (eVar != null) {
            FragmentActivity fragmentActivity = FragmentActivity.this;
            fragmentActivity.f825j = true;
            if (i5 == -1) {
                try {
                    C0547a.m1441n(fragmentActivity, intentSender, i, intent, i2, i3, i4, bundle);
                } catch (Throwable th) {
                    fragmentActivity.f825j = false;
                    throw th;
                }
            } else {
                FragmentActivity.m403h(i);
                C0547a.m1441n(fragmentActivity, intentSender, ((fragmentActivity.mo1281g(this) + 1) << 16) + (i5 & Parser.CLEAR_TI_MASK), intent, i2, i3, i4, bundle);
            }
            fragmentActivity.f825j = false;
            return;
        }
        throw new IllegalStateException(C0802a.m2036m("Fragment ", this, " not attached to Activity"));
    }

    public void startPostponedEnterTransition() {
        C0695g gVar = this.mFragmentManager;
        if (gVar == null || gVar.f2798m == null) {
            ensureAnimationInfo().f815q = false;
        } else if (Looper.myLooper() != this.mFragmentManager.f2798m.f2778c.getLooper()) {
            this.mFragmentManager.f2798m.f2778c.postAtFrontOfQueue(new C0118a());
        } else {
            callStartTransitionListener();
        }
    }

    public String toString() {
        StringBuilder sb = new StringBuilder(Token.EMPTY);
        C0375p.m818g(this, sb);
        if (this.mIndex >= 0) {
            sb.append(" #");
            sb.append(this.mIndex);
        }
        if (this.mFragmentId != 0) {
            sb.append(" id=0x");
            sb.append(Integer.toHexString(this.mFragmentId));
        }
        if (this.mTag != null) {
            sb.append(" ");
            sb.append(this.mTag);
        }
        sb.append('}');
        return sb.toString();
    }

    public void unregisterForContextMenu(View view) {
        view.setOnCreateContextMenuListener((View.OnCreateContextMenuListener) null);
    }

    public static Fragment instantiate(Context context, String str, Bundle bundle) {
        try {
            Class<?> cls = sClassMap.get(str);
            if (cls == null) {
                cls = context.getClassLoader().loadClass(str);
                sClassMap.put(str, cls);
            }
            Fragment fragment = (Fragment) cls.getConstructor(new Class[0]).newInstance(new Object[0]);
            if (bundle != null) {
                bundle.setClassLoader(fragment.getClass().getClassLoader());
                fragment.setArguments(bundle);
            }
            return fragment;
        } catch (ClassNotFoundException e) {
            throw new C0122e("Unable to instantiate fragment " + str + ": make sure class name exists, is public, and has an" + " empty constructor that is public", e);
        } catch (InstantiationException e2) {
            throw new C0122e("Unable to instantiate fragment " + str + ": make sure class name exists, is public, and has an" + " empty constructor that is public", e2);
        } catch (IllegalAccessException e3) {
            throw new C0122e("Unable to instantiate fragment " + str + ": make sure class name exists, is public, and has an" + " empty constructor that is public", e3);
        } catch (NoSuchMethodException e4) {
            throw new C0122e(C0802a.m2039p("Unable to instantiate fragment ", str, ": could not find Fragment constructor"), e4);
        } catch (InvocationTargetException e5) {
            throw new C0122e(C0802a.m2039p("Unable to instantiate fragment ", str, ": calling Fragment constructor caused an exception"), e5);
        }
    }

    public final String getString(int i, Object... objArr) {
        return getResources().getString(i, objArr);
    }

    public void startActivity(Intent intent, Bundle bundle) {
        C0693e eVar = this.mHost;
        if (eVar != null) {
            eVar.mo1305d(this, intent, -1, bundle);
            return;
        }
        throw new IllegalStateException(C0802a.m2036m("Fragment ", this, " not attached to Activity"));
    }

    public void startActivityForResult(Intent intent, int i, Bundle bundle) {
        C0693e eVar = this.mHost;
        if (eVar != null) {
            eVar.mo1305d(this, intent, i, bundle);
            return;
        }
        throw new IllegalStateException(C0802a.m2036m("Fragment ", this, " not attached to Activity"));
    }

    @Deprecated
    public LayoutInflater getLayoutInflater(Bundle bundle) {
        C0693e eVar = this.mHost;
        if (eVar != null) {
            FragmentActivity.C0125b bVar = (FragmentActivity.C0125b) eVar;
            LayoutInflater cloneInContext = FragmentActivity.this.getLayoutInflater().cloneInContext(FragmentActivity.this);
            getChildFragmentManager();
            C0695g gVar = this.mChildFragmentManager;
            if (gVar != null) {
                C0375p.m793W0(cloneInContext, gVar);
                return cloneInContext;
            }
            throw null;
        }
        throw new IllegalStateException("onGetLayoutInflater() cannot be executed until the Fragment is attached to the FragmentManager.");
    }

    @Deprecated
    public void onAttach(Activity activity) {
        this.mCalled = true;
    }

    @Deprecated
    public void onInflate(Activity activity, AttributeSet attributeSet, Bundle bundle) {
        this.mCalled = true;
    }
}
