import streamlit as st
from multiapp import MultiApp
from apps import forestpractice # import your app modules here

app = MultiApp()

st.markdown("""
#  Inteligencia de Negocios - Grupo 6
""")

app.add_app("RandomForestClassifier", forestpractice.app)
# The main app
app.run()
