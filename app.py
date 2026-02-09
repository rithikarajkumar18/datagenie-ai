import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.linear_model import LinearRegression
import numpy as np
import mysql.connector

st.set_page_config(page_title="DataGenie AI", layout="wide")

# ---------- DATABASE CONNECTION ----------
def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",      # replace with your host
        user="your_user",      # replace with your MySQL username
        password="your_password", # replace with your MySQL password
        database="datagenie"   # replace with your database
    )
    return conn

# ---------- USER FUNCTIONS ----------
def register_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        return True
    except mysql.connector.IntegrityError:
        return False
    finally:
        cursor.close()
        conn.close()

def check_login(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username=%s AND password=%s", (username, password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user  # None if not found

# ---------- FILE UPLOAD HISTORY ----------
def save_upload(user_id, filename):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO uploads (user_id, filename) VALUES (%s, %s)", (user_id, filename))
    conn.commit()
    cursor.close()
    conn.close()

def get_user_uploads(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, upload_time FROM uploads WHERE user_id=%s ORDER BY upload_time DESC", (user_id,))
    uploads = cursor.fetchall()
    cursor.close()
    conn.close()
    return uploads

# ---------- SESSION STATE ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ---------- LOGIN PAGE ----------
def login_page():
    st.markdown("""
        <h1 style='text-align: center;'>🤖 DataGenie</h1>
        <p style='text-align: center; color: gray;'>AI-Powered Decision Support Dashboard</p>
    """, unsafe_allow_html=True)

    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = check_login(username, password)
        if user:
            st.session_state.user_id = user[0]
            st.session_state.logged_in = True
            st.session_state.page = "app"
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.markdown("---")
    if st.button("Create new account"):
        st.session_state.page = "register"
        st.rerun()

# ---------- REGISTER PAGE ----------
def register_page():
    st.markdown("<h1 style='text-align: center;'>📝 Register for DataGenie</h1>", unsafe_allow_html=True)
    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")
    confirm_pass = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if not new_user or not new_pass:
            st.warning("Please fill all fields")
        elif new_pass != confirm_pass:
            st.error("Passwords do not match")
        elif register_user(new_user, new_pass) is False:
            st.error("Username already exists")
        else:
            st.success("Account created successfully! Please login.")

    st.markdown("---")
    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# ---------- MAIN APP ----------
def main_app():
    # ---------- HEADER ----------
    col_left, col_right = st.columns([9, 1])
    with col_left:
        st.markdown("""
            <h1 style='text-align: left;'>🤖 DataGenie</h1>
            <p style='text-align: left; color: gray;'>AI-Powered Decision Support Dashboard</p>
        """, unsafe_allow_html=True)
    with col_right:
        if st.button("Logout"):
            st.session_state.logged_in = False
