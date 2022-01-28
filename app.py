import streamlit as st
from multiapp import MultiApp
from apps import forestpractice # import your app modules here
from apps import SVR

app = MultiApp()

app.add_app("RandomForestClassifier", forestpractice.app)
app.add_app("SVR", SVR.app)

# The main app
app.run()
