# streamlit.py
import streamlit as st
import os
import glob
from datetime import datetime
import time
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_autorefresh import st_autorefresh

#Configuration
CALL_LOGS_DIR = "../../call-logs"          #matches main.py
LIVE_TRANSCRIPT_FILE = "../../call-logs/live_transcript.txt"   #Optional live feed


st.set_page_config(
    page_title="Stark Real Estate - Voice AI",
    layout="wide"
)

#Model Layer
class CallModel:
    @staticmethod
    def read_live_transcript(lines=20):
        if not os.path.exists(LIVE_TRANSCRIPT_FILE):
            return None
        with open(LIVE_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
            return f.readlines()[-lines:]


class MoMModel:
    @staticmethod
    def list_moms():
        files = glob.glob(os.path.join(CALL_LOGS_DIR, "mom_*.txt"))
        files.sort(key=os.path.getmtime, reverse=True)
        return files

    @staticmethod
    def extract_call_id(path):
        name = os.path.basename(path).replace(".txt", "")
        _, call_id, *_ = name.split("_")
        return call_id

    @staticmethod
    def read_mom(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def extract_lead_quality(content):
        for line in content.split("\n"):
            if line.startswith("Lead Quality:"):
                return line.split(":")[1].strip()
        return "Unknown"

#View Layer
class SidebarView:
    @staticmethod
    def render():
        st.sidebar.title("Dashboard")
        return st.sidebar.radio(
            "Navigate",
            [
                "Live Call Details",
                "MoM Details",
                "MoM Analytics"
            ]
        )


class LiveCallView:
    @staticmethod
    def render(transcript):
        st.header("Live Call Details")

        with st.expander("Live Call Monitor", expanded=True):
            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown("### Live Transcript")

            with col2:
                if transcript:
                    st.success("Call in progress")
                    st.code("".join(transcript))
                else:
                    st.info("No active call")


class MoMDetailView:
    @staticmethod
    def render(mom_files):
        st.header("Minutes of Meeting")

        if not mom_files:
            st.info("No MoM files available.")
            return

        #Select MoM
        options = {
            f"{MoMModel.extract_call_id(f)} – "
            f"{datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')}"
            : f
            for f in mom_files
        }

        selected = st.selectbox("Select a call", options.keys())
        content = MoMModel.read_mom(options[selected])

        #Display exact text content
        st.text(content)

        st.markdown("---")

        #Owner actions
        st.subheader("Owner Actions")
        st.button("Call Back")
        st.button("Assign to Sales")
        st.button("Mark Low Priority")



class MoMAnalyticsView:
    @staticmethod
    def render(mom_files):
        st.header(" MoM Analytics")

        if not mom_files:
            st.info("No MoM data available.")
            return

        #Collect Data
        lead_quality = []
        cities = []
        budgets = []
        configurations = []

        for f in mom_files:
            content = MoMModel.read_mom(f)

            for line in content.split("\n"):
                line = line.strip()

                if line.startswith("Lead Quality:"):
                    lead_quality.append(line.split(":", 1)[1].strip())

                elif line.startswith("City:"):
                    cities.append(line.split(":", 1)[1].strip())

                elif line.startswith("Budget:"):
                    raw = line.split(":", 1)[1].strip().upper()
                    try:
                        if "CR" in raw:
                            budgets.append(float(raw.replace("CR", "")) * 100)
                        elif "L" in raw:
                            budgets.append(float(raw.replace("L", "")))
                    except:
                        pass

                elif line.startswith("Configuration:"):
                    configurations.append(line.split(":", 1)[1].strip())

        #Lead Quality Overview
        st.subheader("Lead Quality Overview")

        q_count = Counter(lead_quality)

        col1, col2, col3 = st.columns(3)
        col1.metric("Hot", q_count.get("Hot", 0))
        col2.metric("Warm", q_count.get("Warm", 0))
        col3.metric("Cold", q_count.get("Cold", 0))

        if q_count:
            #Convert to DataFrame
            df_quality = pd.DataFrame(
                q_count.items(),
                columns=["Lead Quality", "Count"]
            ).set_index("Lead Quality")

            st.bar_chart(df_quality)
        else:
            st.info("No lead quality data found.")

        st.markdown("---")
        #City Wise Demand
        st.subheader("City-wise Demand")

        if cities:
            city_count = Counter(cities)
            df_city = pd.DataFrame(
                city_count.items(),
                columns=["City", "Leads"]
            ).set_index("City")

            st.bar_chart(df_city)
        else:
            st.info("No city data available.")

        st.markdown("---")

        #Budget Ditribution
        st.subheader("Budget Distribution (Lakhs)")

        if budgets:
            fig, ax = plt.subplots()
            ax.hist(budgets, bins=6)
            ax.set_xlabel("Budget (Lakhs)")
            ax.set_ylabel("Number of Leads")
            st.pyplot(fig)
            plt.close(fig)  
        else:
            st.info("No budget data available.")

        st.markdown("---")

        #Configuration Demand
        st.subheader("Configuration Demand")

        if configurations:
            config_count = Counter(configurations)
            df_config = pd.DataFrame(
                config_count.items(),
                columns=["Configuration", "Leads"]
            ).set_index("Configuration")

            st.bar_chart(df_config)
        else:
            st.info("No configuration data available.")


#Controller
class DashboardController:
    @staticmethod
    def run():
        st.title("Stark Real Estate - Voice AI Dashboard")

        nav = SidebarView.render()
        mom_files = MoMModel.list_moms()

        if nav == "Live Call Details":
            #Auto refresh every 1.5 seconds
            st_autorefresh(interval=1500, key="live_call_refresh")

            transcript = CallModel.read_live_transcript()
            LiveCallView.render(transcript)

        elif nav == "MoM Details":
            MoMDetailView.render(mom_files)

        elif nav == "MoM Analytics":
            MoMAnalyticsView.render(mom_files)

#Entry point
if __name__ == "__main__":
    DashboardController.run()