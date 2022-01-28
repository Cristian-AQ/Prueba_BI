import streamlit as st
from multiapp import MultiApp
from apps import forestpractice,home # import your app modules here

app = MultiApp()
app.add_app("Home", home.app)
app.add_app("RandomForestClassifier", forestpractice.app)
# The main app
app.run()
