import streamlit as st
from supabase import create_client

SUPABASE_KEY = "sb_publishable_D3NT90yBSQRbpEzCBJItYA_qTDPNPI5"
SUPABASE_URL = "https://qrpbsjxrccplaovamfux.supabase.co"


supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_user():
    try:
        user = supabase.auth.get_user()
        if user and user.user:
            return user.user
    except:
        return None
    return None

def get_display_name(user):
    if not user:
        return None
    meta = user.user_metadata or {}

    if meta.get("full_name"):
        return meta["full_name"]
    if meta.get("name"):
        return meta["name"]
    if meta.get("email"):
        return meta["email"]
    return "User"

def process_login_callback():
    params = st.query_params
    if "code" in params:
        code = params["code"] 
        try:
            supabase.auth.exchange_code_for_session({"auth_code": code})
            st.query_params.clear()
            return True
        except Exception as e:
            st.error(f"Lỗi khi đổi mã xác thực: {e}")
    return False

def login(provider):
    options = {
        "redirectTo": "http://localhost:8501", 
    }

    if provider == "facebook":
        options["scopes"] = "email public_profile" 
    elif provider == "google":
        options["scopes"] = "email profile"

    res = supabase.auth.sign_in_with_oauth({
        "provider": provider,
        "options": options
    })

    st.markdown(
        f'<meta http-equiv = "refresh" content ="0; url ={res.url}">',
        unsafe_allow_html= True
    )

def logout():
    supabase.auth.sign_out()
    st.session_state.clear()